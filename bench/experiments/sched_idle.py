"""GPU idle inside a real scheduler run (k concurrent greedy 256-token requests, no stop tokens): CUDA trace of the
whole run, kernel intervals merged (PDL kernels overlap), idle = wall - busy, plus the largest gaps.

Modal: parity --args "sched_idle.py [k] [adaptive|plain]"
"""

import json
import os
import sys
import tempfile
import threading
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402

k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
mode = sys.argv[3] if len(sys.argv) > 3 else "adaptive"
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4)
sched = Scheduler(eng)
sched.batched_spec = mode == "adaptive"
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
prompts = [it["input_ids"] for it in items[:6]]


def run():
    res = {}
    th = [threading.Thread(target=lambda i=i: res.__setitem__(i, sched.generate(prompts[i], 256, set(), Sampling())))
          for i in range(k)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    return sum(len(res[i][0]) for i in res), time.time() - t0


run()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    toks, dt = run()
print(f"k={k} {mode}: {toks} tokens in {dt:.2f}s -> {toks / dt:.0f} tok/s")
path = os.path.join(tempfile.mkdtemp(), "trace.json")
prof.export_chrome_trace(path)
ev = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev)
busy, gaps, cur_s, cur_e = 0.0, [], iv[0][0], iv[0][1]
for s, e in iv[1:]:
    if s > cur_e:
        busy += cur_e - cur_s
        gaps.append(s - cur_e)
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
span = iv[-1][1] - iv[0][0]
gaps.sort(reverse=True)
big = [g for g in gaps if g > 50]
print(f"kernels {len(iv)}, span {span / 1e3:.0f} ms, busy {busy / 1e3:.0f} ms, idle {(span - busy) / 1e3:.0f} ms "
      f"({100 * (span - busy) / span:.1f}%); gaps > 50 us: {len(big)} totalling {sum(big) / 1e3:.0f} ms; "
      f"largest {[round(g) for g in gaps[:8]]} us")
hist = {">1ms": sum(g for g in gaps if g > 1000), "200us-1ms": sum(g for g in gaps if 200 < g <= 1000),
        "50-200us": sum(g for g in gaps if 50 < g <= 200), "<50us": sum(g for g in gaps if g <= 50)}
print("idle by gap size (ms):", {kk: round(v / 1e3, 1) for kk, v in hist.items()})
