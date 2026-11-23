"""Verify-sized MoE (T = 8..32 tokens, one NVFP4 layer of 128 experts, random routing per step so weights stream
from DRAM): bf16-dequant grouped path (moe_grouped_small) vs NVFP4 x NVFP4 tensor-core path (moe_prefill_fp4).

Modal: parity --args "bench_verify_moe_fp4.py"
"""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.ops import moe_triton as mt  # noqa: E402
from inkling_engine.ops.moe_fp4 import moe_prefill_fp4  # noqa: E402
from inkling_engine.quant import ExpertWeight  # noqa: E402

E, H, I, TOPK, STEPS = 128, 2304, 1152, 8, 24
torch.manual_seed(0)
print(torch.cuda.get_device_name(0))


def nvfp4(e, n, k):
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    s = (torch.rand(e, n, k // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
    return ExpertWeight("nvfp4", w, s, torch.full((e,), 0.01, device="cuda"))


w13, w2 = nvfp4(E, 2 * I, H), nvfp4(E, H, I)


def timed(fn):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 5 / STEPS * 1e3


for t in (4, 8, 16, 32):
    ids = [torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(t)]).int() for _ in range(STEPS)]
    wts = torch.rand(t, TOPK, device="cuda")
    x = (torch.randn(t, H, device="cuda") * 0.5).to(torch.bfloat16)
    sc = mt.GroupedScratch(t, TOPK, E, H, I)
    rows = torch.empty(t * TOPK, H, device="cuda")
    bf = timed(lambda: [mt.moe_grouped_small(x, i, wts, w13, w2, 4.0, 25.0, sc, rows) for i in ids])
    r_bf = mt.moe_grouped_small(x, ids[0], wts, w13, w2, 4.0, 25.0, sc, rows).clone()
    res = [f"T={t:2d}: bf16 grouped {bf:7.1f} us"]
    for bm in (16, 32, 64):
        try:
            f4 = timed(lambda: [moe_prefill_fp4(x, i, wts, w13, w2, 4.0, 25.0, bm=bm, rows_out=rows) for i in ids])
            r4 = moe_prefill_fp4(x, ids[0], wts, w13, w2, 4.0, 25.0, bm=bm, rows_out=rows)
            rel = ((r4 - r_bf).norm() / r_bf.norm()).item()
            res.append(f"fp4 bm={bm} {f4:7.1f} us (rel diff {rel:.3f})")
        except Exception as e:
            res.append(f"fp4 bm={bm} failed: {type(e).__name__}: {str(e)[:80]}")
    print(" | ".join(res))
