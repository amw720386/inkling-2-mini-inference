"""End-to-end effect of fp4_prefill on the fused model (Modal: parity --args "fp4_e2e.py [tokens]").

Prefills natural text (stdlib docstrings) with all logits and compares:
  bf16 chunk 1024 vs bf16 chunk 512    the bf16 noise floor (batch-shape differences only)
  fp4 vs bf16 (both chunk 1024)        what NVFP4 activations change
Reports top-1 agreement, KL, and the mean next-token NLL of each run on the text itself.
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
model = FusedModel(cfg, w, fp4_prefill=False)
ws = model.workspace(1024, 1024)


def run(chunk, fp4):
    model.fp4_prefill = fp4
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
    print(f"[{name}] top1={top1:.4f} KL(mean)={kl.mean():.2e} KL(p99)={kl.quantile(0.99):.2e}")


def nll(a):
    tgt = torch.tensor(ids[1:])
    return torch.nn.functional.cross_entropy(a[:-1], tgt).item()


base = run(1024, False)
base512 = run(512, False)
fp4 = run(1024, True)
print(f"{n} tokens of natural text")
compare("bf16 chunk1024 vs chunk512 (noise floor)", base512, base)
compare("fp4 vs bf16", fp4, base)
print(f"NLL bf16 {nll(base):.4f} | bf16 chunk512 {nll(base512):.4f} | fp4 {nll(fp4):.4f}")
