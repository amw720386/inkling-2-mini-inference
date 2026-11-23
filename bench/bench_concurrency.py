"""Scheduler: 1/2/4 concurrent requests, aggregate throughput and per-request correctness.

Modal: modal run modal/engine_app.py::parity --args "bench_concurrency.py"
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

path = sys.argv[1]
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=16384, spec=True, slots=4)
print(f"engine (4 slots, 16k ctx, DSpark): allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
sched = Scheduler(eng)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
stop = {eng.cfg.eos_token_id}
N = 256

# reference: each prompt alone (greedy) through the scheduler
alone = {}
for i in range(4):
    out, st = sched.generate(items[i]["input_ids"], N, stop)
    alone[i] = out
    print(f"alone[{i}]: {len(out)} tok, {st.decode_tok_s:.1f} tok/s (spec rounds {st.rounds})")

for conc in (2, 4):
    res = {}

    def worker(i):
        res[i] = sched.generate(items[i]["input_ids"], N, stop)

    t0 = time.time()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.time() - t0
    toks = sum(len(res[i][0]) for i in res)
    match = [next((k for k, (a, b) in enumerate(zip(res[i][0], alone[i])) if a != b), min(len(res[i][0]), len(alone[i])))
             for i in range(conc)]
    print(f"{conc} concurrent: {toks} tokens in {dt:.2f}s -> {toks / dt:.0f} tok/s aggregate | "
          f"greedy prefix equal to the solo run: {match}")

# sampled + greedy mixed
res = {}
th = [threading.Thread(target=lambda i=i: res.__setitem__(i, sched.generate(
    items[i]["input_ids"], 64, stop, Sampling(temperature=0.8, seed=i) if i % 2 else None))) for i in range(4)]
for t in th:
    t.start()
for t in th:
    t.join()
print("mixed greedy/sampled 4 concurrent:", {i: len(res[i][0]) for i in res}, "OK")
