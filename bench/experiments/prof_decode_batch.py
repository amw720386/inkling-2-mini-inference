"""Per-kernel profile of the batched decode graph on k real, different sequences (greedy, own tokens fed back),
so each row routes to its own experts (bench_fused.py feeds one token to every row).

Modal: parity --args "INKLING_PDL=0 prof_decode_batch.py [k]"
"""

import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=8192, spec=False, slots=4)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
P, toks = [], []
for s in range(k):
    ids = items[s]["input_ids"]
    lg = eng.prefill(ids)[0] if s == 0 else eng.prefill_slot(s, ids)
    P.append(len(ids))
    toks.append(int(lg.reshape(-1, lg.shape[-1])[-1].argmax()))
g = eng.decode_graphs[k]


def step():
    global toks
    lg = g.step(list(range(k)), P, toks)
    toks = lg[:k].argmax(-1).tolist()
    for i in range(k):
        P[i] += 1


for _ in range(5):
    step()
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(20):
    g.graph.replay()  # same inputs, for the pure graph time (rows still distinct)
e1.record()
torch.cuda.synchronize()
print(f"decode graph k={k}: {e0.elapsed_time(e1) / 20:.2f} ms per replay")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=40))
