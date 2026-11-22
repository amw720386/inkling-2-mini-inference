"""NVFP4 x NVFP4 grouped expert GEMMs on the FP4 tensor cores (sm_120 `tl.dot_scaled`), for prefill.

The bf16 grouped path (moe_triton.moe_prefill) dequantizes the expert weights and runs bf16 MMAs with fp32
accumulation, which GeForce Blackwell (RTX 5090) executes at half rate. Here the activations are quantized to
NVFP4 as well, so both operands feed the block-scaled e2m1 MMA directly. This follows vLLM's sm_120 Inkling 2
path (vllm/models/inkling2/nvidia/ops, Apache-2.0): a per-row outer scale amax / 1536, e4m3 scales per 16
values, and per block the better of a 6.0- and a 4.0-based scale ("four over six"). The quantizer below is a
trimmed port of its `nvfp4_quantize` (row-major scales, hardware e2m1 conversion only).

Activation quantization changes the numerics (it is what SGLang / vLLM do on this model, not what the bf16
checkpoint math does), so the path is opt-in. Only the NVFP4 layers use it; MXFP8 layers stay on bf16.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..quant import ExpertWeight
from .moe_triton import _swiglu_pairs_2d, plan_tiles

def check_support():
    major, minor = (int(v) for v in triton.__version__.split(".")[:2])
    if (major, minor) < (3, 6):
        raise RuntimeError(f"fp4_prefill needs Triton >= 3.6 (NVFP4 tl.dot_scaled on sm_120), found {triton.__version__}")


FP32_MAX = tl.constexpr(3.4028234663852886e38)


# ------------------------------------------------------------------ activation quantizer
@triton.jit
def _pack_roundtrip(x):
    """x [..., 2] fp32 -> (e2m1x2 bytes [...], the rounded values [..., 2]) via cvt.rn.satfinite."""
    low, high = tl.split(x)
    packed, halves = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
        "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
        "=r,=r,f,f",
        [low, high],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    lo = (halves & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return packed.to(tl.uint8), tl.join(lo, hi).to(tl.float32)


@triton.jit
def _block_error(x, rec):
    groups: tl.constexpr = x.shape[0]
    err = tl.abs(rec - x)
    for width in tl.static_range(3, -1, -1):
        lo, hi = tl.split(tl.reshape(err, (groups, 1 << width, 2)))
        err = lo + hi
    return tl.reshape(err, (groups,))


@triton.jit
def _quantize_block(x, enc, dec, outer, BLOCK: tl.constexpr):
    x = tl.reshape(x, (BLOCK // 16, 16))
    bmax = tl.max(tl.abs(x), 1)
    s6 = tl.minimum(bmax * (enc * (1.0 / 6.0)), 448.0).to(tl.float8e4nv).to(tl.float32)
    s4 = tl.minimum(bmax * (enc * 0.25), 448.0).to(tl.float8e4nv).to(tl.float32)
    i6 = tl.minimum(tl.div_rn(1.0, s6 * dec), FP32_MAX)
    i4 = tl.minimum(tl.div_rn(1.0, s4 * dec), FP32_MAX)
    p6, r6 = _pack_roundtrip(tl.reshape(x * i6[:, None], (BLOCK // 16, 8, 2)))
    p4, r4 = _pack_roundtrip(tl.reshape(x * i4[:, None], (BLOCK // 16, 8, 2)))
    e6 = _block_error(x, tl.reshape(r6, (BLOCK // 16, 16)) * s6[:, None] * outer)
    e4 = _block_error(x, tl.reshape(r4, (BLOCK // 16, 16)) * s4[:, None] * outer)
    use4 = e4 < e6
    packed = tl.reshape(tl.where(use4[:, None], p4, p6), (BLOCK // 2,))
    return packed, tl.where(use4, s4, s6).to(tl.float8e4nv)


@triton.jit
def _nvfp4_quant_kernel(X, x_stride, Values, Scales, Outer, M, K: tl.constexpr, BLOCK: tl.constexpr):
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        r = row.to(tl.int64)
        amax = tl.zeros((BLOCK,), tl.float32)
        for start in range(0, K, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            amax = tl.maximum(amax, tl.abs(tl.load(X + r * x_stride + cols).to(tl.float32)))
        amax = tl.max(amax, 0)
        enc = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
        dec = tl.div_rn(1.0, enc)
        outer = amax * (1.0 / 1536.0)
        tl.store(Outer + r, outer)
        for start in range(0, K, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(X + r * x_stride + cols).to(tl.float32)
            packed, scales = _quantize_block(x, enc, dec, outer, BLOCK)
            tl.store(Values + r * (K // 2) + start // 2 + tl.arange(0, BLOCK // 2), packed)
            tl.store(Scales + r * (K // 16) + start // 16 + tl.arange(0, BLOCK // 16), scales)


def nvfp4_quantize(x: torch.Tensor):
    """Rows of bf16 x [M, K] -> (e2m1 bytes u8 [M, K/2], e4m3 block scales [M, K/16], fp32 outer scale [M])."""
    m, k = x.shape
    assert k % 16 == 0 and x.stride(1) == 1
    values = torch.empty(m, k // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(m, k // 16, dtype=torch.float8_e4m3fn, device=x.device)
    outer = torch.empty(m, dtype=torch.float32, device=x.device)
    if m == 0:
        return values, scales, outer
    block, warps = (512, 4) if m <= 512 else (256, 1)
    block = math.gcd(k, block)
    _nvfp4_quant_kernel[(min(m, 8192),)](x, x.stride(0), values, scales, outer, m, K=k, BLOCK=block,
                                         num_warps=warps, enable_fp_fusion=False)
    return values, scales, outer


def nvfp4_dequantize(values, scales, outer):
    """Reference inverse of nvfp4_quantize (fp32), for tests."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=values.device)
    v = torch.stack([lut[(values & 15).long()], lut[(values >> 4).long()]], -1).reshape(values.shape[0], -1)
    return (v.view(v.shape[0], -1, 16) * scales.float()[:, :, None]).reshape(v.shape) * outer[:, None]


# ------------------------------------------------------------------------ grouped GEMM
_FP4_CONFIGS = [
    triton.Config({"BN": bn, "BK": bk}, num_warps=nw, num_stages=ns)
    for bn in (64, 128)
    for bk in (128, 256, 512)
    for nw in (4, 8)
    for ns in (2, 3)
]


def _prune_fp4(configs, named_args, **kw):
    k = named_args["K"]
    return [c for c in configs if k % c.kwargs["BK"] == 0]


@triton.autotune(cache_results=True, configs=_FP4_CONFIGS, key=["N", "K", "ACT", "BM"],
                 prune_configs_by={"early_config_prune": _prune_fp4})
@triton.jit
def _grouped_fp4_kernel(
    A, AS, AO, RowSrc, a_row_div,  # sorted row r reads quantized row RowSrc[r] // a_row_div
    W, WS, F, TileExpert, TileStart, TileEnd,
    Out, out_stride, Wts, OutRow,
    N, K: tl.constexpr,
    ACT: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile = tl.program_id(0)
    pn = tl.program_id(1)
    start = tl.load(TileStart + tile)
    end = tl.load(TileEnd + tile)
    if start >= end:
        return
    e = tl.load(TileExpert + tile).to(tl.int64)
    rm = start + tl.arange(0, BM)
    mmask = rm < end
    src = tl.load(RowSrc + rm, mask=mmask, other=0)
    arow = (src // a_row_div).to(tl.int64)
    n = pn * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        kp = k0 // 2 + tl.arange(0, BK // 2)
        g = k0 // 16 + tl.arange(0, BK // 16)
        a = tl.load(A + arow[:, None] * (K // 2) + kp[None, :], mask=mmask[:, None], other=0)
        a_s = tl.load(AS + arow[:, None] * (K // 16) + g[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(W + e * N * (K // 2) + n[None, :] * (K // 2) + kp[:, None], mask=nmask[None, :], other=0,
                    eviction_policy="evict_first")
        w_s = tl.load(WS + e * N * (K // 16) + n[:, None] * (K // 16) + g[None, :], mask=nmask[:, None], other=0.0)
        acc = tl.dot_scaled(a, a_s, "e2m1", w, w_s, "e2m1", acc)
    acc = acc * tl.load(F + e) * tl.load(AO + arow, mask=mmask, other=0.0)[:, None]
    if ACT == 1:
        act = _swiglu_pairs_2d(acc, BM, BN, GATE_CAP, UP_CAP)
        no = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(Out + rm.to(tl.int64)[:, None] * out_stride + no[None, :], act.to(tl.bfloat16),
                 mask=mmask[:, None] & (no < N // 2)[None, :])
    else:
        orow = tl.load(OutRow + rm, mask=mmask, other=0)
        wt = tl.load(Wts + orow, mask=mmask, other=0.0)
        acc = acc.to(tl.bfloat16).to(tl.float32) * wt[:, None]
        tl.store(Out + orow.to(tl.int64)[:, None] * out_stride + n[None, :], acc, mask=mmask[:, None] & nmask[None, :])


def moe_prefill_fp4(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, w13: ExpertWeight, w2: ExpertWeight,
                    gate_cap: float, up_cap: float, bm: int = 128, rows_out: torch.Tensor | None = None):
    """NVFP4 layers only. Same contract as moe_prefill: fp32 per-pair weighted rows [T*k, H], (token, k) order."""
    assert w13.fmt == "nvfp4" and w2.fmt == "nvfp4"
    t, hdim = x.shape
    k = ids.shape[1]
    e, n13, _ = w13.shape
    inter = n13 // 2
    row_src, tile_expert, tile_start, tile_end, n_tiles = plan_tiles(ids, e, bm)
    wts_f = wts.reshape(-1).float().contiguous()
    rows = rows_out if rows_out is not None else torch.empty(t * k, hdim, dtype=torch.float32, device=x.device)

    xq, xs, xo = nvfp4_quantize(x)
    act = torch.empty(t * k, inter, dtype=torch.bfloat16, device=x.device)
    _grouped_fp4_kernel[lambda m: (n_tiles, triton.cdiv(n13, m["BN"]))](
        xq, xs, xo, row_src, k, w13.weight, w13.scale, w13.expert_factor, tile_expert, tile_start, tile_end,
        act, act.stride(0), wts_f, row_src, n13, hdim, 1, gate_cap, up_cap, BM=bm,
    )
    aq, as_, ao = nvfp4_quantize(act)
    del act
    ident = torch.arange(t * k, device=x.device, dtype=torch.int32)
    _grouped_fp4_kernel[lambda m: (n_tiles, triton.cdiv(hdim, m["BN"]))](
        aq, as_, ao, ident, 1, w2.weight, w2.scale, w2.expert_factor, tile_expert, tile_start, tile_end,
        rows, rows.stride(0), wts_f, row_src, hdim, inter, 0, gate_cap, up_cap, BM=bm,
    )
    return rows
