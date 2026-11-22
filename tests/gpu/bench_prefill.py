"""Prefill throughput and kernel breakdown (Modal: parity --args "bench_prefill.py [tokens] [chunk]")."""

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import Batch, SeqChunk  # noqa: E402
from inkling_engine.cache import CacheSpec, InklingCache  # noqa: E402
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.fused_model import FusedModel  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
cfg = InklingConfig.from_pretrained(path)
w = load_weights(cfg, path)
cache = InklingCache(cfg, CacheSpec(slots=1, max_ctx=n + 64, ring=cfg.sliding_window + 64))
model = FusedModel(cfg, w)
ws = model.workspace(chunk, 1)
ids = torch.randint(1000, 60000, (n,)).tolist()


def run():
    cache.reset_slot(0)
    for s in range(0, n, chunk):
        part = ids[s:s + chunk]
        model.forward(Batch.build([SeqChunk(0, s, len(part), "last")], [part]), cache, ws)


run()
torch.cuda.synchronize()
t0 = time.time()
run()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"prefill {n} tokens, chunk {chunk}: {dt * 1000:.0f} ms -> {n / dt:.0f} tok/s")
from torch.profiler import ProfilerActivity, profile  # noqa: E402

ops = len(sys.argv) > 4 and sys.argv[4] == "ops"  # per aten op and input shape instead of per kernel
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU] if ops else [ProfilerActivity.CUDA],
             record_shapes=ops) as prof:
    run()
    torch.cuda.synchronize()
if ops:
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=30,
                                                             max_name_column_width=40, max_shapes_column_width=70))
else:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=50))
