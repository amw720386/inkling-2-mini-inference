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
eng = Engine(path, max_ctx=32768, spec=True)
app = build_app(Server(eng, tok, "inkling-2-mini"))
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
print("SERVER", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
