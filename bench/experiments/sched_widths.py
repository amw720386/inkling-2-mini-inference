"""k concurrent greedy 256-token requests (no stop tokens) through the scheduler: the default batched verify widths
vs forced widths, and batched speculation vs plain batched steps
(Modal: parity --args "sched_widths.py [k] [default,m=2,m=4,m=8,plain]").
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
default_pick = eng._pick_width_batched


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
names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["default", "m=2", "m=4", "m=8", "plain"]
for name in names:
    sched.batched_spec = name != "plain"
    eng._pick_width_batched = (lambda reqs, m=int(name[2:]): m) if name.startswith("m=") else default_pick
    if name.startswith("lead="):  # progress balancing lead (tokens), "lead=off" disables
        eng.BALANCE_LEAD = None if name[5:] == "off" else int(name[5:])
    if name.startswith("wide="):  # threshold of mean recent acceptance for width 8 on a 2-request batch
        eng.WIDE_PAIR_ACCEPT = float(name[5:])
    rates = [run(rep) for rep in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2)]
    print(f"k={k} {name:8s}: {' / '.join(f'{r:.0f}' for r in rates)} tok/s | batched_spec rounds "
          f"{sched.counters['batched_spec'][0]} batched {sched.counters['batched'][0]}", flush=True)
    sched.counters = {kk: ([0, 0.0] if kk != "batch_sizes" else {}) for kk in sched.counters}
print("online costs", {kk: round(v, 2) for kk, v in eng.verify_costs.items() if kk[0] == k},
      "decode", {kk: round(v, 2) for kk, v in eng.decode_costs.items()})
