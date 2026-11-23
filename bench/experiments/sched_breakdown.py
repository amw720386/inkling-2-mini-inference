"""Where a 4-request server run's time goes: scheduler step counts and time per kind
(Modal: parity --args "sched_breakdown.py").

Replays the serve_bench concurrency case in process: k requests of 256 tokens, no stop tokens, started together,
greedy. Prints wall time, tokens/s and the scheduler's per-kind step counters, once with adaptive batched
speculation and once with plain batched steps only.
"""

import json
import sys
import threading
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4)
sched = Scheduler(eng)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
prompts = [it["input_ids"] for it in items[:6]]

for k in (2, 4):
    for mode in ("adaptive", "plain"):
        sched.batched_spec = mode == "adaptive"
        for rep in range(2):
            sched.counters = {kk: ([0, 0.0] if kk != "batch_sizes" else {}) for kk in sched.counters}
            res = {}
            th = [threading.Thread(target=lambda i=i: res.__setitem__(
                i, sched.generate(prompts[(i + rep) % 6], 256, set(), Sampling()))) for i in range(k)]
            t0 = time.time()
            [x.start() for x in th]
            [x.join() for x in th]
            dt = time.time() - t0
            toks = sum(len(res[i][0]) for i in res)
            c = {kk: (v if kk == "batch_sizes" else (v[0], round(v[1], 3))) for kk, v in sched.counters.items()}
            busy = sum(v[1] for kk, v in sched.counters.items() if kk != "batch_sizes")
            print(f"k={k} {mode:8s} rep{rep}: {toks} tokens in {dt:.2f}s -> {toks / dt:.0f} tok/s | "
                  f"scheduler busy {busy:.2f}s | {c}")
