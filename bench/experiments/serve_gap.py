"""Server overhead at k concurrent requests: the same prompt sets through the HTTP server (in this process, the
serve_bench client) and directly through the scheduler, interleaved, plus the scheduler's step counters.

Modal: parity --args "serve_gap.py [k] [reps]"
"""

import importlib.util
import json
import sys
import threading
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402

k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))

import uvicorn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from inkling_engine.server import Server, build_app, generation_defaults  # noqa: E402

spec = importlib.util.spec_from_file_location("client", "/root/bench/serve_bench.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

model = sys.argv[1]
eng = Engine(model, max_ctx=32768, spec=True, slots=4, vision=False)
srv = Server(eng, AutoTokenizer.from_pretrained(model), "m", generation_defaults(model))
cfg = uvicorn.Config(build_app(srv), host="127.0.0.1", port=8011, log_level="warning")
us = uvicorn.Server(cfg)
threading.Thread(target=us.run, daemon=True).start()
while not us.started:
    time.sleep(0.1)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
prompts = [it["input_ids"] for it in items[:6]]
sched = srv.sched


def direct(rep):
    res = {}
    th = [threading.Thread(target=lambda i=i: res.__setitem__(
        i, sched.generate(prompts[(i + rep) % 6], 256, set(), Sampling()))) for i in range(k)]
    t0 = time.perf_counter()
    [x.start() for x in th]
    [x.join() for x in th]
    return sum(len(res[i][0]) for i in res) / (time.perf_counter() - t0)


def http(rep):
    """The serve_bench client in its own process (as in production), one rotation."""
    import subprocess
    out = "/tmp/serve_gap_client.json"
    subprocess.run([sys.executable, "/root/bench/serve_bench.py", "--base", "http://127.0.0.1:8011/v1", "--model", "m",
                    "--prompts", "/root/tests/reference/sglang_ref.json", "--skip", "decode,ttft,gsm8k",
                    "--conc", str(k), "--reps", str(rep + 1), "--out", out], check=True, capture_output=True)
    return json.load(open(out))["concurrency"][str(k)]["runs"][rep]


http(0)
direct(0)
modes = [("http_one", http, False), ("http_all", http, True), ("direct", direct, True)]
for rep in range(reps):
    for name, fn, si in modes:
        sched.prefill_all = si
        sched.counters = {kk: ([0, 0.0] if kk != "batch_sizes" else {}) for kk in sched.counters}
        t0 = time.perf_counter()
        r = fn(rep)
        wall = time.perf_counter() - t0
        busy = sum(v[1] for kk, v in sched.counters.items() if kk != "batch_sizes")
        c = {kk: (v if kk == "batch_sizes" else (v[0], round(v[1], 3))) for kk, v in sched.counters.items()}
        print(f"rep{rep} {name:6s}: {r:.0f} tok/s | wall {wall:.2f}s, scheduler busy {busy:.2f}s | {c}", flush=True)
