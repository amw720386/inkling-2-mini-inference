"""Decode benchmark + fast-vs-reference check (Modal: parity --args "bench_decode.py [ctx]").

GPU memory is capped to 32 GiB to emulate an RTX 5090. Numbers from the RTX PRO 6000
(same GB202 / sm_120 chip, more SMs, similar bandwidth) are a proxy, not 5090 results.
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import Batch, SeqChunk  # noqa: E402
from inkling_engine.cache import CacheSpec, InklingCache  # noqa: E402
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.fast import FastBackend  # noqa: E402
from inkling_engine.model import InklingModel, RefBackend  # noqa: E402
from inkling_engine.runner import DecodeGraph  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

path = sys.argv[1]
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
print(torch.cuda.get_device_name(0), f"SMs={torch.cuda.get_device_properties(0).multi_processor_count}",
      "memory capped to 32 GiB")

cfg = InklingConfig.from_pretrained(path)
w = load_weights(cfg, path)
cache = InklingCache(cfg, CacheSpec(slots=4, max_ctx=max(8192, ctx + 1024), ring=cfg.sliding_window + 64))
fast = InklingModel(cfg, w, FastBackend(cfg))
print(f"weights+cache {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

graphs = {bs: DecodeGraph(fast, cache, bs) for bs in (1, 4)}
print(f"after graph capture {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

torch.manual_seed(0)
prompt = torch.randint(1000, 60000, (ctx,)).tolist()


def prefill(model, slot):
    cache.reset_slot(slot)
    t0 = time.time()
    out = []
    for s in range(0, ctx, 2048):
        part = prompt[s:s + 2048]
        out.append(model.forward(Batch.build([SeqChunk(slot, s, len(part), "last")], [part]), cache)[0])
    torch.cuda.synchronize()
    return out[-1], time.time() - t0


# fast vs reference on the last prompt token (bf16 noise floor ~ KL 5e-2)
lf, tf = prefill(fast, 0)
print(f"fast prefill {ctx} tok: {tf * 1000:.0f} ms ({ctx / tf:.0f} tok/s, includes compile on first call)")
lf2, tf2 = prefill(fast, 0)
print(f"fast prefill {ctx} tok (warm): {tf2 * 1000:.0f} ms ({ctx / tf2:.0f} tok/s)")
if ctx <= 2048:
    refm = InklingModel(cfg, w, RefBackend(cfg))
    lr, _ = prefill(refm, 1)
    a, b = torch.log_softmax(lf.float(), -1), torch.log_softmax(lr.float(), -1)
    print(f"fast-vs-ref last token: KL={(b.exp() * (b - a)).sum().item():.2e} "
          f"argmax {'match' if lf.argmax() == lr.argmax() else 'DIFF'}")

# eager fast decode vs graph decode: identical kernels -> identical logits
tok = int(lf2[-1].argmax())
snap = [t[:, 0].clone() for t in (cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv)]
eager = fast.forward(Batch.build([SeqChunk(0, ctx, 1)], [[tok]]), cache)[0].clone()
for t, s in zip((cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv), snap):
    t[:, 0].copy_(s)
g = graphs[1].step([0], [ctx], [tok]).clone()
print(f"graph-vs-eager decode max|d| = {(g - eager).abs().max().item():.3e}")

# timing
for bs, gr in graphs.items():
    slots = list(range(bs))
    for s in slots[1:]:
        cache.reset_slot(s)
    pos = [ctx] * bs
    for _ in range(5):
        gr.step(slots, pos, [tok] * bs)
    torch.cuda.synchronize()
    n = 100
    t0 = time.time()
    for i in range(n):
        gr.graph.replay()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"decode bs={bs} ctx={ctx}: {dt * 1000:.2f} ms/step -> {bs / dt:.0f} tok/s aggregate, "
          f"{1 / dt:.0f} tok/s per stream")

# kernel breakdown for one bs=1 step
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        graphs[1].graph.replay()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))
