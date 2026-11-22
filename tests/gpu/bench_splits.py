"""Decode graph time vs attention key splits (local / global)."""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.fused_model import FusedDecodeGraph  # noqa: E402

eng = Engine(sys.argv[1], max_ctx=16384, spec=False)
eng.prefill(list(range(1000, 1000 + int(sys.argv[2]) if len(sys.argv) > 2 else 2000)))


def t(fn, n=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


for nl, ng in ((4, 16), (8, 16), (16, 16), (8, 32), (16, 32)):
    g = FusedDecodeGraph(eng.model, eng.cache, 1, n_split_local=nl, n_split_global=ng)
    print(f"splits local {nl:2d} global {ng:2d}: decode {t(g.graph.replay):.3f} ms")
