"""Kernel breakdown of one DSpark draft (the captured graph) (Modal: parity --args "prof_draft.py")."""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=32768, spec=True)
d = eng.drafter
eng.prefill(list(range(1000, 1000 + 3000)))
g = d.graphs[(d.buckets[0], 0)]
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(50):
    g.replay()
e1.record()
torch.cuda.synchronize()
print(f"draft graph: {e0.elapsed_time(e1) / 50:.3f} ms")
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))
