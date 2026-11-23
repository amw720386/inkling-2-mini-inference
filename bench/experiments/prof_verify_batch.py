"""Per-kernel profile of real batched DSpark rounds (k distinct greedy sequences, width m): draft_multi, verify(k, m),
commits. Modal: parity --args "INKLING_PDL=0 prof_verify_batch.py [k] [m]"
"""

import json
import sys
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine, GenStats  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402

k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
m = int(sys.argv[3]) if len(sys.argv) > 3 else 4
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4)
eng._pick_width_batched = lambda reqs: m
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
reqs = []
for s in range(k):
    ids = items[s]["input_ids"]
    lg = eng.prefill(ids)[0] if s == 0 else eng.prefill_slot(s, ids)
    tok = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
    reqs.append(SimpleNamespace(slot=s, P=len(ids), out=[tok], sp=Sampling(), gen=None, stats=GenStats()))


def round_():
    for r, (new, commit) in zip(reqs, eng.batched_spec_round(reqs)):
        r.P += commit
        r.out += new


for _ in range(5):
    round_()
torch.cuda.synchronize()
import time  # noqa: E402

t0 = time.perf_counter()
for _ in range(20):
    round_()
torch.cuda.synchronize()
print(f"k={k} m={m}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms per round wall; costs "
      f"verify {eng.verify_costs[(k, m)]:.2f} draft {eng.draft_multi_costs[k]:.2f}")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        round_()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=40))
