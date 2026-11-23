"""Acceptance per request: alone (spec_round) vs in a batched round (batched_spec_round), same prompts, same
verify width, greedy, 256 tokens, no stop tokens (Modal: parity --args "batched_accept.py [m]").
"""

import json
import sys
import threading

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402

m = int(sys.argv[2]) if len(sys.argv) > 2 else 4
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4)
sched = Scheduler(eng)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
prompts = [it["input_ids"] for it in items[:4]]
eng._pick_width = lambda conf, gamma=None, seeded=False: m
eng._pick_width_batched = lambda reqs: m
sched._spec_pays = lambda reqs: True

alone = []
for p in prompts:
    out, st = sched.generate(p, 256, set(), Sampling())
    alone.append((out, (st.accepted + st.rounds) / max(st.rounds, 1)))
res = {}
th = [threading.Thread(target=lambda i=i: res.__setitem__(i, sched.generate(prompts[i], 256, set(), Sampling())))
      for i in range(4)]
[x.start() for x in th]
[x.join() for x in th]
for i in range(4):
    out, st = res[i]
    same = next((j for j, (a, b) in enumerate(zip(out, alone[i][0])) if a != b), min(len(out), len(alone[i][0])))
    print(f"prompt {i}: alone accept len {alone[i][1]:.2f} | concurrent accept len "
          f"{(st.accepted + st.rounds) / max(st.rounds, 1):.2f} over {st.rounds} rounds, widths {st.widths} | "
          f"same tokens for {same}/{len(out)}")
print("scheduler", sched.counters)
