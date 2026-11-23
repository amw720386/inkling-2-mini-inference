"""Would NVFP4 dense weights (attention projections, output projection, LM head) keep quality? Simulation
(Modal: parity --args "fp4_dense_sim.py [tokens]").

Rounds those bf16 weights through NVFP4 (round-to-nearest, e4m3 scale per 16 values with the 4-or-6 choice, fp32
per-row outer scale) and back to bf16, then compares prefill logits on natural text with the checkpoint's, next to
the same measurement for row-scaled FP8 (what dense_fp8 does) and the bf16 noise floor.
"""

import argparse
import collections
import email
import json
import re
import sys
import textwrap

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import Batch, SeqChunk  # noqa: E402
from inkling_engine.cache import CacheSpec, InklingCache  # noqa: E402
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.fused_model import FusedModel  # noqa: E402
from inkling_engine.ops.moe_fp4 import nvfp4_dequantize, nvfp4_quantize  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 48 * 2**30 / total))
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
text = "\n\n".join(textwrap.dedent(m.__doc__ or "") for m in (argparse, collections, email, json, re, textwrap))
text += "\n\n" + "\n\n".join(textwrap.dedent(f.__doc__ or "") for f in vars(re).values() if callable(f))
ids = tok.encode(text)
while len(ids) < n:
    ids += ids
ids = ids[:n]
cfg = InklingConfig.from_pretrained(path)
w = load_weights(cfg, path)
cache = InklingCache(cfg, CacheSpec(slots=1, max_ctx=n + 64, ring=cfg.sliding_window + 64))
model = FusedModel(cfg, w)
ws = model.workspace(1024, 1024)


def run(chunk=1024):
    cache.reset_slot(0)
    out = []
    for s in range(0, n, chunk):
        part = ids[s:s + chunk]
        logits = model.forward(Batch.build([SeqChunk(0, s, len(part), "all")], [part]), cache, ws)
        out.append(logits[:len(part)].float().cpu())
    return torch.cat(out)


def compare(name, a, b):
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (lb.exp() * (lb - la)).sum(-1)
    nll = torch.nn.functional.cross_entropy(a[:-1], torch.tensor(ids[1:])).item()
    print(f"[{name}] top1={top1:.4f} KL(mean)={kl.mean():.2e} KL(p99)={kl.quantile(0.99):.2e} NLL {nll:.4f}")


def fp4_roundtrip(t):
    out = torch.empty_like(t)
    for s in range(0, t.shape[0], 4096):
        q, sc, o = nvfp4_quantize(t[s:s + 4096].contiguous())
        out[s:s + 4096] = nvfp4_dequantize(q, sc, o).to(t.dtype)
    return out


def fp8_roundtrip(t):
    amax = t.float().abs().amax(1, keepdim=True).clamp(min=1e-12)
    return ((t.float() / (amax / 448)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * (amax / 448)).to(t.dtype)


targets = [model.w_proj[i] for i in range(len(model.w_proj))] + [lw.wo for lw in w.layers] + [w.unembed]
orig = [t.clone() for t in targets]
base = run()
base512 = run(512)
compare("bf16 chunk512 vs chunk1024 (noise floor)", base512, base)
compare("bf16 reference", base, base)
for name, fn in (("FP8 rows (dense_fp8)", fp8_roundtrip), ("NVFP4 (simulated)", fp4_roundtrip)):
    for t, o in zip(targets, orig):
        t.copy_(fn(o))
    compare(name, run(), base)
    for t, o in zip(targets, orig):
        t.copy_(o)
