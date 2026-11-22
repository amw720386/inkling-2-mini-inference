"""Experiment: one persistent launch for the decode MoE (w13 + SwiGLU, grid barrier, w2) vs two launches.

Modal: parity --args "bench_moe_fused.py". NVFP4 layer, bs 1, 8 experts, weights rotated past L2.
"""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root")
from inkling_engine.ops.moe_triton import (_e2m1x8_to_f16, _f16x2_hi, _f16x2_lo, _swiglu_pairs_1d,  # noqa: E402
                                           moe_decode)
from inkling_engine.quant import ExpertWeight  # noqa: E402


@triton.jit
def _dot_words(W32, w_base, n, nmask, KW: tl.constexpr, k0, x_ptr, BN: tl.constexpr, BK: tl.constexpr,
               CG: tl.constexpr):
    NW: tl.constexpr = BK // 8
    j = tl.arange(0, NW)
    wv = tl.load(W32 + w_base + n[:, None] * KW + (k0 // 8 + j)[None, :], mask=nmask[:, None], other=0,
                 eviction_policy="evict_first")
    c0, c1, c2, c3 = _e2m1x8_to_f16(wv)
    xb = x_ptr + k0 + 8 * j
    cm: tl.constexpr = ".cg" if CG else ""
    acc = _f16x2_lo(c0) * tl.load(xb, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_hi(c0) * tl.load(xb + 1, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_lo(c1) * tl.load(xb + 2, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_hi(c1) * tl.load(xb + 3, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_lo(c2) * tl.load(xb + 4, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_hi(c2) * tl.load(xb + 5, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_lo(c3) * tl.load(xb + 6, cache_modifier=cm).to(tl.float32)[None, :]
    acc += _f16x2_hi(c3) * tl.load(xb + 7, cache_modifier=cm).to(tl.float32)[None, :]
    return tl.sum(tl.reshape(acc, (BN, BK // 16, 2)), axis=2)


@triton.jit
def _gemv_block(W, S, F, e, n, nmask, N: tl.constexpr, K: tl.constexpr, x_ptr, BN: tl.constexpr, BK: tl.constexpr,
                CG: tl.constexpr):
    KS: tl.constexpr = K // 16
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        js = tl.arange(0, BK // 16)
        sc = tl.load(S + e * N * KS + n[:, None] * KS + (k0 // 16 + js)[None, :], mask=nmask[:, None], other=0.0,
                     eviction_policy="evict_first")
        p = _dot_words(W.to(tl.pointer_type(tl.int32), bitcast=True), e * N * (K // 8), n, nmask, K // 8, k0, x_ptr,
                       BN, BK, CG)
        acc += tl.sum(p * sc.to(tl.float32), axis=1)
    return acc * tl.load(F + e)


@triton.jit
def _moe_fused_kernel(X, W13, S13, F13, W2, S2, F2, Ids, Act, Out, Wts, Bar, NPROG,
                      P: tl.constexpr, H: tl.constexpr, I: tl.constexpr, GATE_CAP: tl.constexpr,
                      UP_CAP: tl.constexpr, BN1: tl.constexpr, BK1: tl.constexpr, BN2: tl.constexpr,
                      BK2: tl.constexpr):
    pid = tl.program_id(0)
    N13: tl.constexpr = 2 * I
    NB1: tl.constexpr = N13 // BN1
    for item in range(pid, P * NB1, NPROG):
        r = item // NB1
        pn = item % NB1
        e = tl.load(Ids + r).to(tl.int64)
        n = pn * BN1 + tl.arange(0, BN1)
        acc = _gemv_block(W13, S13, F13, e, n, n < N13, N13, H, X, BN1, BK1, False)
        a = _swiglu_pairs_1d(acc, BN1, GATE_CAP, UP_CAP)
        no = pn * (BN1 // 2) + tl.arange(0, BN1 // 2)
        tl.store(Act + r * I + no, a.to(tl.bfloat16))
    # grid barrier (sense reversal; every program is resident: NPROG <= SMs)
    gen = tl.atomic_add(Bar + 1, 0, sem="acquire")
    c = tl.atomic_add(Bar, 1, sem="acq_rel")
    if c == NPROG - 1:
        tl.atomic_xchg(Bar, 0)
        tl.atomic_add(Bar + 1, 1, sem="release")
    else:
        cur = tl.atomic_add(Bar + 1, 0, sem="acquire")
        while cur == gen:
            cur = tl.atomic_add(Bar + 1, 0, sem="acquire")
    NB2: tl.constexpr = H // BN2
    for item in range(pid, P * NB2, NPROG):
        r = item // NB2
        pn = item % NB2
        e = tl.load(Ids + r).to(tl.int64)
        n = pn * BN2 + tl.arange(0, BN2)
        acc = _gemv_block(W2, S2, F2, e, n, n < H, H, I, Act + r * I, BN2, BK2, True)
        acc = acc.to(tl.bfloat16).to(tl.float32) * tl.load(Wts + r)
        tl.store(Out + r * H + n, acc)


def moe_fused(x, ids, wts, w13, w2, act, rows, bar, nprog, cfg):
    p = ids.numel()
    bn1, bk1, bn2, bk2, nw = cfg
    _moe_fused_kernel[(nprog,)](x, w13.weight, w13.scale, w13.expert_factor, w2.weight, w2.scale, w2.expert_factor,
                                ids.reshape(-1).int(), act, rows, wts.reshape(-1).float(), bar, nprog,
                                P=p, H=2304, I=1152, GATE_CAP=4.0, UP_CAP=25.0, BN1=bn1, BK1=bk1, BN2=bn2, BK2=bk2,
                                num_warps=nw)


def timeit(fn, n=64):
    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1e3


e = 128
torch.manual_seed(0)
w13 = ExpertWeight("nvfp4", torch.randint(0, 255, (e, 2304, 1152), dtype=torch.uint8, device="cuda"),
                   (torch.rand(e, 2304, 144, device="cuda") * 0.01).to(torch.float8_e4m3fn),
                   torch.full((e,), 0.5, device="cuda"))
w2 = ExpertWeight("nvfp4", torch.randint(0, 255, (e, 2304, 576), dtype=torch.uint8, device="cuda"),
                  (torch.rand(e, 2304, 72, device="cuda") * 0.01).to(torch.float8_e4m3fn),
                  torch.full((e,), 0.5, device="cuda"))
x = torch.randn(1, 2304, device="cuda").to(torch.bfloat16)
perm = torch.randperm(128, device="cuda").view(16, 8).int()
wts = torch.rand(1, 8, device="cuda")
act = torch.empty(8, 1152, dtype=torch.bfloat16, device="cuda")
rows = torch.empty(8, 2304, dtype=torch.float32, device="cuda")
act2 = torch.empty_like(act)
rows2 = torch.empty_like(rows)
bar = torch.zeros(2, dtype=torch.int32, device="cuda")
sms = torch.cuda.get_device_properties(0).multi_processor_count
per = sum(t[0].numel() * t.element_size() for t in (w13.weight, w2.weight, w13.scale, w2.scale))

moe_decode(x, perm[0].view(1, 8), wts, w13, w2, 4.0, 25.0, act, rows)
base = rows.clone()
us = timeit(lambda i: moe_decode(x, perm[i % 16].view(1, 8), wts, w13, w2, 4.0, 25.0, act, rows))
print(f"two launches: {us:.1f} us -> {8 * per / us / 1e3:.0f} GB/s")
for cfg in [(64, 256, 32, 128, 4), (128, 256, 64, 128, 16), (128, 512, 128, 128, 16), (256, 256, 128, 128, 32),
            (64, 512, 64, 128, 16)]:
    for mult in (1,):  # all programs must be resident for the grid barrier
        nprog = sms * mult
        try:
            moe_fused(x, perm[0].view(1, 8), wts, w13, w2, act2, rows2, bar, nprog, cfg)
            torch.cuda.synchronize()
            err = ((rows2 - base).norm() / base.norm()).item()
            us = timeit(lambda i: moe_fused(x, perm[i % 16].view(1, 8), wts, w13, w2, act2, rows2, bar, nprog, cfg))
            print(f"fused {cfg} nprog {nprog}: {us:.1f} us -> {8 * per / us / 1e3:.0f} GB/s (rel err {err:.1e})")
        except Exception as exc:  # noqa: BLE001
            print(f"fused {cfg} nprog {nprog}: failed {type(exc).__name__}: {str(exc)[:120]}")
