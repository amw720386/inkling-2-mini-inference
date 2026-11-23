"""Fused forward: correctness vs the reference model, then CUDA-graph decode timing.

Modal: modal run modal/engine_app.py::parity --args "bench_fused.py [ctx] [profile]"
Memory capped to 32 GiB on an RTX PRO 6000.
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import Batch, SeqChunk  # noqa: E402
from inkling_engine.cache import CacheSpec, InklingCache  # noqa: E402
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.fused_model import FusedDecodeGraph, FusedModel  # noqa: E402
from inkling_engine.model import InklingModel, RefBackend  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

path = sys.argv[1]
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
do_prof = len(sys.argv) > 3 and sys.argv[3] == "profile"
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
print(torch.cuda.get_device_name(0), "| memory capped to 32 GiB")

cfg = InklingConfig.from_pretrained(path)
w = load_weights(cfg, path)
cache = InklingCache(cfg, CacheSpec(slots=4, max_ctx=max(8192, ctx + 1024), ring=cfg.sliding_window + 64))
fused = FusedModel(cfg, w)
graphs = {bs: FusedDecodeGraph(fused, cache, bs) for bs in (1, 4)}
print(f"weights+cache+graphs {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
refm = InklingModel(cfg, w, RefBackend(cfg))

torch.manual_seed(0)
prompt = torch.randint(1000, 60000, (ctx,)).tolist()
pws = fused.workspace(2048, 1)


def kl(a, b):
    a, b = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    return (b.exp() * (b - a)).sum(-1)


def prefill_fused(slot):
    cache.reset_slot(slot)
    for s in range(0, ctx, 2048):
        part = prompt[s:s + 2048]
        out = fused.forward(Batch.build([SeqChunk(slot, s, len(part), "last")], [part]), cache, pws).clone()
    return out


# --- correctness on a short prefix (reference model is slow)
n_chk = min(ctx, 300)
fused.capture_visits = refm.capture_visits = [1, 2, 5, 6, 12, 18, 30, 47, 48]
cache.reset_slot(2)
cache.reset_slot(3)
lf = fused.forward(Batch.build([SeqChunk(2, 0, n_chk, "all")], [prompt[:n_chk]]), cache, pws).clone()
lr, caps = refm.forward(Batch.build([SeqChunk(3, 0, n_chk, "all")], [prompt[:n_chk]]), cache)
for vi in refm.capture_visits:
    a, b = fused.captures[vi].float(), caps[vi].float()
    print(f"  visit {vi:2d}: rel err {((a - b).norm() / b.norm()).item():.2e}")
k = kl(lf, lr)
print(f"prefill fused-vs-ref: KL mean {k.mean().item():.2e} max {k.max().item():.2e} "
      f"top1 {(lf.argmax(-1) == lr.argmax(-1)).float().mean().item():.3f}")
fused.capture_visits = []
# decode 12 steps: fused graph vs reference, teacher-forced on the reference's tokens
tok = int(lr[-1].argmax())
kls, agree = [], 0
for i in range(12):
    p = n_chk + i
    gf = graphs[1].step([2], [p], [tok]).clone()
    gr, _ = refm.forward(Batch.build([SeqChunk(3, p, 1)], [[tok]]), cache)
    kls.append(kl(gf, gr).item())
    agree += int(gf.argmax() == gr.argmax())
    tok = int(gr.argmax())
print(f"decode fused-graph-vs-ref: KL mean {sum(kls) / len(kls):.2e} max {max(kls):.2e}, argmax agree {agree}/12")

# --- prefill speed
for rep in range(2):
    torch.cuda.synchronize()
    t0 = time.time()
    prefill_fused(0)
    torch.cuda.synchronize()
    dt = time.time() - t0
print(f"prefill {ctx} tok: {dt * 1000:.0f} ms ({ctx / dt:.0f} tok/s)")

# --- decode timing
for bs, gr in graphs.items():
    slots = list(range(bs))
    for s in slots[1:]:
        prefill_fused(s) if s < 2 else None
    pos = [ctx] * bs
    gr.step(slots, pos, [1000] * bs)
    torch.cuda.synchronize()
    n = 200
    t0 = time.time()
    for _ in range(n):
        gr.graph.replay()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"decode bs={bs} ctx~{ctx}: {dt * 1000:.2f} ms/step -> {bs / dt:.0f} tok/s aggregate, {1 / dt:.0f} tok/s/stream")

if do_prof:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            graphs[1].graph.replay()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=50))
