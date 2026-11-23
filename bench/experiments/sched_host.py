"""Host-side profile of the scheduler's decode steps (k concurrent greedy requests of 256 tokens, no stop tokens):
cProfile around each step inside the scheduler thread. Time in .tolist() / synchronize is GPU wait; the rest is
host work the GPU may idle through. (Modal: parity --args "sched_host.py [k] [adaptive|plain]")
"""

import cProfile
import json
import pstats
import sys
import threading
import time

import torch

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
prof = cProfile.Profile()
PROFILE = [False]
for name in ("_batched_spec_step", "_batched_step", "_spec_step"):
    orig = getattr(Scheduler, name)

    def wrapped(self, *a, _orig=orig):
        if not PROFILE[0]:
            return _orig(self, *a)
        prof.enable()
        try:
            return _orig(self, *a)
        finally:
            prof.disable()
    setattr(Scheduler, name, wrapped)


def run():
    sched.counters = {kk: ([0, 0.0] if kk != "batch_sizes" else {}) for kk in sched.counters}
    res = {}
    th = [threading.Thread(target=lambda i=i: res.__setitem__(i, sched.generate(prompts[i], 256, set(), Sampling())))
          for i in range(k)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    dt = time.time() - t0
    toks = sum(len(res[i][0]) for i in res)
    return toks, dt


r2 = lambda d: {kk: round(v, 2) for kk, v in d.items()}
print("calibrated verify costs", r2(eng.verify_costs), "decode", r2(eng.decode_costs), "draft", r2(eng.draft_multi_costs))
run()  # warm-up: multi-draft graph capture for this slot set
PROFILE[0] = True
toks, dt = run()
print(f"k={k} {mode}: {toks} tokens in {dt:.2f}s -> {toks / dt:.0f} tok/s | counters {sched.counters}")
print("online verify costs", r2(eng.verify_costs), "decode", r2(eng.decode_costs), "draft", r2(eng.draft_multi_costs))
st = pstats.Stats(prof)
st.sort_stats("tottime").print_stats(25)
st.sort_stats("cumulative").print_stats(30)
