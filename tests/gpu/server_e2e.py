"""OpenAI-server end-to-end check (Modal: parity --args "server_e2e.py").

Starts the server in-process, then: chat (non-stream), chat (stream, with a follow-up turn that must
reuse the cached prefix), raw completion, a sampled request, and the models list.
"""

import json
import sys
import threading
import time

import requests
import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.server import Server, build_app  # noqa: E402

path = sys.argv[1]
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))

from transformers import AutoTokenizer  # noqa: E402
import uvicorn  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
eng = Engine(path, max_ctx=32768, spec=True, vision=True)
server = Server(eng, tok, "inkling-2-mini")
app = build_app(server)
cfg = uvicorn.Config(app, host="127.0.0.1", port=8011, log_level="warning")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(100):
    try:
        if requests.get("http://127.0.0.1:8011/health", timeout=1).ok:
            break
    except Exception:
        time.sleep(0.2)
B = "http://127.0.0.1:8011/v1"
ok = True

print("models:", requests.get(B + "/models").json())

msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
r = requests.post(B + "/chat/completions", json={"messages": msgs, "max_tokens": 200, "reasoning_effort": "low"})
if r.status_code != 200:
    print("HTTP", r.status_code, r.text[:2000])
r = r.json()
m = r["choices"][0]["message"]
print("chat reasoning:", repr(m.get("reasoning_content", ""))[:200])
print("chat content:", repr(m["content"]))
print("usage:", r["usage"], "timings:", r["timings"])
ok &= "Paris" in m["content"]

# streaming + follow-up turn
msgs2 = msgs + [{"role": "assistant", "content": m["content"]},
                {"role": "user", "content": "And of Germany?"}]
t0 = time.time()
parts, reasoning, first, final = [], [], None, None
with requests.post(B + "/chat/completions", json={"messages": msgs2, "max_tokens": 200, "stream": True,
                                                  "reasoning_effort": "low"}, stream=True) as resp:
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        d = ev["choices"][0].get("delta", {})
        if first is None and (d.get("content") or d.get("reasoning_content")):
            first = time.time() - t0
        parts.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        if "usage" in ev:
            final = ev
print(f"stream: first delta {first * 1000:.0f} ms, content {''.join(parts)!r}")
print("stream usage:", final["usage"] if final else None)
ok &= "Berlin" in "".join(parts)
ok &= final is not None and final["usage"]["prompt_tokens_details"]["cached_tokens"] > 0

r = requests.post(B + "/completions", json={"prompt": "The three primary colors are", "max_tokens": 24}).json()
print("completion:", repr(r["choices"][0]["text"]))
r = requests.post(B + "/completions", json={"prompt": "Once upon a time", "max_tokens": 32, "temperature": 0.8,
                                           "top_p": 0.95, "seed": 7}).json()
r2 = requests.post(B + "/completions", json={"prompt": "Once upon a time", "max_tokens": 32, "temperature": 0.8,
                                            "top_p": 0.95, "seed": 7}).json()
print("sampled:", repr(r["choices"][0]["text"]), "| same seed reproduces:", r["choices"][0]["text"] == r2["choices"][0]["text"])
ok &= r["choices"][0]["text"] == r2["choices"][0]["text"]
# tool calling round trip
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
tmsgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
r = requests.post(B + "/chat/completions", json={"messages": tmsgs, "tools": tools, "max_tokens": 300,
                                                 "reasoning_effort": "low"}).json()
ch = r["choices"][0]
calls = ch["message"].get("tool_calls") or []
print("tool call:", ch["finish_reason"], [(c["function"]["name"], c["function"]["arguments"]) for c in calls])
ok &= ch["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"     and "paris" in calls[0]["function"]["arguments"].lower()
if calls:
    tmsgs += [ch["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temp_c": 18, "sky": "cloudy"}'}]
    r = requests.post(B + "/chat/completions", json={"messages": tmsgs, "tools": tools, "max_tokens": 300,
                                                     "reasoning_effort": "low"}).json()
    ans = r["choices"][0]["message"]["content"] or ""
    print("after tool result:", repr(ans[:200]), "| cached:", r["usage"]["prompt_tokens_details"]["cached_tokens"])
    ok &= "18" in ans
# streamed tool call
stream_calls = []
with requests.post(B + "/chat/completions", json={"messages": tmsgs[:1], "tools": tools, "max_tokens": 300,
                                                  "stream": True, "reasoning_effort": "low"}, stream=True) as resp:
    fin = None
    for line in resp.iter_lines():
        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
            ev = json.loads(line[6:])
            stream_calls += ev["choices"][0].get("delta", {}).get("tool_calls") or []
            fin = ev["choices"][0].get("finish_reason") or fin
print("streamed tool call:", fin, [c["function"]["name"] for c in stream_calls])
ok &= fin == "tool_calls" and stream_calls and stream_calls[0]["function"]["name"] == "get_weather"

# image input (OpenAI image_url with a base64 data URL)
import base64, io  # noqa: E401,E402
from PIL import Image, ImageDraw  # noqa: E402

im = Image.new("RGB", (320, 240), (30, 160, 40))
ImageDraw.Draw(im).ellipse((100, 60, 220, 180), fill=(250, 250, 0))
buf = io.BytesIO()
im.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
r = requests.post(B + "/chat/completions", json={"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}},
    {"type": "text", "text": "What shape and colors do you see? One sentence."}]}], "max_tokens": 200,
    "reasoning_effort": "none"}).json()
img_ans = r["choices"][0]["message"]["content"] or ""
print("image answer:", repr(img_ans))
ok &= ("circle" in img_ans.lower() or "yellow" in img_ans.lower())

# client disconnect mid-stream must stop generation and free the slot
with requests.post(B + "/completions", json={"prompt": "Write a very long story about a dragon.", "max_tokens": 3000,
                                             "stream": True}, stream=True) as resp:
    for i, line in enumerate(resp.iter_lines()):
        if i > 10:
            break
t0 = time.time()
while (server.sched.active or server.sched.prefilling) and time.time() - t0 < 10:
    time.sleep(0.05)
freed = not server.sched.active and not server.sched.prefilling
print(f"disconnect: slot freed {freed} after {time.time() - t0:.2f}s")
ok &= freed
# stop sequences: plain and streaming, a stop string spanning tokens, the stop text itself not returned
body = {"prompt": "Count: 1, 2, 3, 4, 5,", "max_tokens": 40, "stop": [" 9", "never-appears"]}
r = requests.post(B + "/completions", json=body).json()
txt, fin = r["choices"][0]["text"], r["choices"][0]["finish_reason"]
print("stop:", repr(txt), fin)
ok &= fin == "stop" and " 9" not in txt and "8" in txt
parts, sfin = [], None
with requests.post(B + "/completions", json={**body, "stream": True}, stream=True) as rs:
    for line in rs.iter_lines():
        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
            ch = json.loads(line[6:])["choices"][0]
            parts.append(ch.get("text", ""))
            sfin = ch.get("finish_reason") or sfin
print("stop stream:", repr("".join(parts)), sfin)
ok &= "".join(parts) == txt and sfin == "stop"
print("SERVER", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
