"""4 concurrent greedy 256-token requests (no stop tokens) through the scheduler: the width policy vs forced
common verify widths, and batched speculation vs plain batched steps (Modal: parity --args "sched_widths.py [k]").
"""

import json
import sys
import threading
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import VERIFY_WIDTHS, Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402

k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4)
sched = Scheduler(eng)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
prompts = [it["input_ids"] for it in items[:6]]
policy = eng._pick_width_batched


def run(rep):
    res = {}
    th = [threading.Thread(target=lambda i=i: res.__setitem__(
        i, sched.generate(prompts[(i + rep) % 6], 256, set(), Sampling()))) for i in range(k)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    dt = time.time() - t0
    return sum(len(res[i][0]) for i in res) / dt


run(0)
for name in ["m=2", "m=4", "m=8"]:
    sched.batched_spec = name != "plain"
    eng._pick_width_batched = (lambda confs, g, m=int(name[2:]): m) if name.startswith("m=") else policy
    pays = Scheduler._spec_pays
    if name != "policy":
        sched._spec_pays = lambda reqs: True
    else:
        sched._spec_pays = lambda reqs, s=sched: pays(s, reqs)
    rates = [run(rep) for rep in range(2)]
    print(f"k={k} {name:12s}: {' / '.join(f'{r:.0f}' for r in rates)} tok/s | batched_spec rounds "
          f"{sched.counters['batched_spec'][0]} batched {sched.counters['batched'][0]}")
    sched.counters = {kk: ([0, 0.0] if kk != "batch_sizes" else {}) for kk in sched.counters}
print("online costs", {kk: round(v, 2) for kk, v in eng.verify_costs.items() if kk[0] == k},
      "decode", {kk: round(v, 2) for kk, v in eng.decode_costs.items()})
