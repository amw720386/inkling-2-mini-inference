"""Server soak test: mixed concurrent load (Modal: parity --args "soak.py [n_requests] [clients]").

Random mix of chat / completion, greedy / sampled, streaming (some dropped mid-stream), short and ~6k-token
prompts, tool calls. Checks every request finishes without error, the scheduler drains, and GPU memory does
not grow between waves.
"""

import json
import random
import sys
import threading
import time

import requests
import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.server import Server, build_app  # noqa: E402

path = sys.argv[1]
n_req = int(sys.argv[2]) if len(sys.argv) > 2 else 60
clients = int(sys.argv[3]) if len(sys.argv) > 3 else 6
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))

from transformers import AutoTokenizer  # noqa: E402
import uvicorn  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
eng = Engine(path, max_ctx=16384, spec=True, slots=4, vision=False)
server = Server(eng, tok, "inkling-2-mini")
srv = uvicorn.Server(uvicorn.Config(build_app(server), host="127.0.0.1", port=8012, log_level="warning"))
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(100):
    try:
        if requests.get("http://127.0.0.1:8012/health", timeout=1).ok:
            break
    except Exception:
        time.sleep(0.2)
B = "http://127.0.0.1:8012/v1"
words = "the quick brown fox jumps over a lazy dog while engineers debug kernels on blackwell gpus".split()
tools = [{"type": "function", "function": {"name": "get_time", "description": "Current time in a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
lock = threading.Lock()
stats = {"ok": 0, "err": 0, "dropped": 0, "tokens": 0, "errors": []}


def one(rng: random.Random):
    kind = rng.choice(["chat", "chat", "completion", "tool", "long"])
    stream = rng.random() < 0.5
    drop = stream and rng.random() < 0.2
    body = {"max_tokens": rng.choice([16, 64, 200]), "stream": stream}
    if rng.random() < 0.4:
        body.update(temperature=rng.choice([0.7, 1.0]), top_p=0.95, seed=rng.randint(0, 999))
    if kind == "completion":
        url = B + "/completions"
        body["prompt"] = " ".join(rng.choice(words) for _ in range(rng.randint(5, 60)))
    else:
        url = B + "/chat/completions"
        n = 3000 if kind == "long" else rng.randint(5, 80)
        body["messages"] = [{"role": "user", "content": " ".join(rng.choice(words) for _ in range(n))
                             + (" What time is it in Tokyo? Use the tool." if kind == "tool" else " Summarize.")}]
        body["reasoning_effort"] = "low"
        if kind == "tool":
            body["tools"] = tools
    try:
        if stream:
            with requests.post(url, json=body, stream=True, timeout=300) as r:
                r.raise_for_status()
                n_ev = 0
                for line in r.iter_lines():
                    if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                        ev = json.loads(line[6:])
                        n_ev += 1
                        if drop and n_ev > 5:
                            with lock:
                                stats["dropped"] += 1
                            return
                        if "usage" in ev:
                            with lock:
                                stats["tokens"] += ev["usage"]["completion_tokens"]
        else:
            r = requests.post(url, json=body, timeout=300)
            r.raise_for_status()
            with lock:
                stats["tokens"] += r.json()["usage"]["completion_tokens"]
        with lock:
            stats["ok"] += 1
    except Exception as e:  # noqa: BLE001
        with lock:
            stats["err"] += 1
            stats["errors"].append(f"{kind}: {e!r}"[:200])


def wave(n, seed):
    rngs = [random.Random(seed * 1000 + c) for c in range(clients)]
    per = [n // clients + (1 if c < n % clients else 0) for c in range(clients)]
    th = [threading.Thread(target=lambda c=c: [one(rngs[c]) for _ in range(per[c])]) for c in range(clients)]
    t0 = time.time()
    for x in th:
        x.start()
    for x in th:
        x.join()
    return time.time() - t0


mem = []
for w in range(2):
    dt = wave(n_req // 2, w)
    t0 = time.time()
    while (server.sched.active or server.sched.prefilling or server.sched.waiting) and time.time() - t0 < 30:
        time.sleep(0.1)
    torch.cuda.synchronize()
    mem.append(torch.cuda.memory_allocated() / 2**30)
    print(f"wave {w}: {dt:.1f}s, ok {stats['ok']} err {stats['err']} dropped {stats['dropped']} "
          f"tokens {stats['tokens']}, scheduler idle: {not server.sched.active and not server.sched.prefilling}, "
          f"allocated {mem[-1]:.2f} GiB")
for e in stats["errors"][:10]:
    print("ERROR", e)
ok = stats["err"] == 0 and abs(mem[1] - mem[0]) < 0.25 and not server.sched.active
print("SOAK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
