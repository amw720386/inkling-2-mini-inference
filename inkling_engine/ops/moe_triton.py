"""Weight-only quantized MoE kernels (Triton) for sm_120.

Activations stay bf16. Expert weights are decoded in registers:
  NVFP4: u8 [E, N, K/2] (low nibble = even element), e4m3 scale per 16, fp32 factor[E]
  MXFP8: e4m3 [E, N, K], u8 exponent per 32 (2 ** (s - 127))

decode path  (few tokens): one program per (token, expert) pair x N-block, GEMV style.
prefill path (many tokens): rows sorted by expert, grouped tl.dot over tiles.

w13 output columns interleave (gate, up), so the softcapped SwiGLU is fused into the
w13 epilogue. w2 writes per-pair rows scaled by the routing weight; the caller (or the
fused conv epilogue) sums the top-k rows of each token.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..quant import ExpertWeight


@triton.jit
def _e2m1_to_f32(c):
    """c: int32 nibble codes 0..15 -> fp32 values, via bit construction."""
    s = (c >> 3) & 1
    e = (c >> 1) & 3
    m = c & 1
    bits = tl.where(e == 0, m * 0x3F000000, ((e + 126) << 23) | (m << 22))
    bits = bits | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _e8m0_to_f32(s):
    return (s.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _tanh(x):
    # tanh(x) = 2 * sigmoid(2x) - 1, stable for large |x|
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _swiglu(g, u, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    return GATE_CAP * _tanh(g / GATE_CAP) * tl.sigmoid(g) * (UP_CAP * _tanh(u / UP_CAP))


@triton.jit
def _swiglu_pairs_1d(acc, BN: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    """acc [BN] interleaved (gate, up) -> [BN // 2] activation (fp32)."""
    acc = acc.to(tl.bfloat16).to(tl.float32)  # mirror the bf16 gate_up GEMM output
    g, u = tl.split(tl.reshape(acc, (BN // 2, 2)))
    return _swiglu(g, u, GATE_CAP, UP_CAP)


@triton.jit
def _swiglu_pairs_2d(acc, BM: tl.constexpr, BN: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    acc = acc.to(tl.bfloat16).to(tl.float32)
    g, u = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
    return _swiglu(g, u, GATE_CAP, UP_CAP)


# --------------------------------------------------------------------------- decode
_GEMV_CONFIGS = [
    triton.Config({"BN": bn, "BK": bk}, num_warps=nw, num_stages=ns)
    for bn in (16, 32, 64)
    for bk in (128, 256, 512)
    for nw in (4, 8)
    for ns in (2, 3)
]


def _prune(configs, named_args, **kw):
    k = named_args["K"]
    return [c for c in configs if k % c.kwargs["BK"] == 0]


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K", "FMT", "ACT"], restore_value=["Out"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemv_kernel(
    X, x_row_div, x_stride,  # activations: row of pair r is r // x_row_div
    W, S, F, Ids,  # expert weights, scales, nvfp4 factor, expert id per pair
    Out, out_stride, Wts,
    N, K: tl.constexpr,
    FMT: tl.constexpr,  # 0 nvfp4, 1 mxfp8
    ACT: tl.constexpr,  # 1: fused swiglu (w13), 0: scale by routing weight (w2)
    GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    tl.static_assert(K % BK == 0)
    r = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(Ids + r).to(tl.int64)
    n = pn * BN + tl.arange(0, BN)
    nmask = n < N
    x_ptr = X + (r // x_row_div).to(tl.int64) * x_stride
    acc = tl.zeros((BN,), dtype=tl.float32)
    if FMT == 0:
        KB: tl.constexpr = K // 2
        KS: tl.constexpr = K // 16
        w_base = W + e * N * KB
        s_base = S + e * N * KS
        for k0 in tl.range(0, K, BK):
            j = tl.arange(0, BK // 2)
            xe = tl.load(x_ptr + k0 + 2 * j).to(tl.float32)
            xo = tl.load(x_ptr + k0 + 2 * j + 1).to(tl.float32)
            wb = tl.load(w_base + n[:, None] * KB + (k0 // 2 + j)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            p = _e2m1_to_f32(wb & 15) * xe[None, :] + _e2m1_to_f32(wb >> 4) * xo[None, :]
            p = tl.sum(tl.reshape(p, (BN, BK // 16, 8)), axis=2)
            js = tl.arange(0, BK // 16)
            sc = tl.load(s_base + n[:, None] * KS + (k0 // 16 + js)[None, :], mask=nmask[:, None], other=0.0)
            acc += tl.sum(p * sc.to(tl.float32), axis=1)
        acc = acc * tl.load(F + e)
    else:
        KS2: tl.constexpr = K // 32
        w_base = W + e * N * K
        s_base = S + e * N * KS2
        for k0 in tl.range(0, K, BK):
            j = tl.arange(0, BK)
            xv = tl.load(x_ptr + k0 + j).to(tl.float32)
            wv = tl.load(w_base + n[:, None] * K + (k0 + j)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
            p = tl.sum(tl.reshape(wv * xv[None, :], (BN, BK // 32, 32)), axis=2)
            js = tl.arange(0, BK // 32)
            sc = tl.load(s_base + n[:, None] * KS2 + (k0 // 32 + js)[None, :], mask=nmask[:, None], other=127)
            acc += tl.sum(p * _e8m0_to_f32(sc), axis=1)
    if ACT == 1:
        a = _swiglu_pairs_1d(acc, BN, GATE_CAP, UP_CAP)
        no = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(Out + r.to(tl.int64) * out_stride + no, a.to(tl.bfloat16), mask=no < N // 2)
    else:
        acc = acc.to(tl.bfloat16).to(tl.float32) * tl.load(Wts + r)
        tl.store(Out + r.to(tl.int64) * out_stride + n, acc, mask=nmask)


def _fmt_id(w: ExpertWeight) -> int:
    return {"nvfp4": 0, "mxfp8": 1}[w.fmt]


def _scale_view(w: ExpertWeight):
    return w.scale if w.fmt == "nvfp4" else w.scale.view(torch.uint8)


def moe_decode(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, w13: ExpertWeight, w2: ExpertWeight,
               gate_cap: float, up_cap: float, act_buf: torch.Tensor | None = None,
               rows_buf: torch.Tensor | None = None) -> torch.Tensor:
    """x [T, H] bf16; ids/wts [T, k]. Returns fp32 rows [T*k, H] (weighted, not summed)."""
    t, hdim = x.shape
    k = ids.shape[1]
    e, n13, _ = w13.shape
    inter = n13 // 2
    pairs = t * k
    ids_f = ids.reshape(-1).to(torch.int32)
    wts_f = wts.reshape(-1).float()
    act = act_buf if act_buf is not None else torch.empty(pairs, inter, dtype=torch.bfloat16, device=x.device)
    rows = rows_buf if rows_buf is not None else torch.empty(pairs, hdim, dtype=torch.float32, device=x.device)
    f13 = w13.expert_factor if w13.fmt == "nvfp4" else wts_f
    f2 = w2.expert_factor if w2.fmt == "nvfp4" else wts_f
    _gemv_kernel[lambda m: (pairs, triton.cdiv(n13, m["BN"]))](
        x, k, x.stride(0), w13.weight, _scale_view(w13), f13, ids_f, act, act.stride(0), wts_f,
        n13, hdim, _fmt_id(w13), 1, gate_cap, up_cap,
    )
    _gemv_kernel[lambda m: (pairs, triton.cdiv(hdim, m["BN"]))](
        act, 1, act.stride(0), w2.weight, _scale_view(w2), f2, ids_f, rows, rows.stride(0), wts_f,
        hdim, inter, _fmt_id(w2), 0, gate_cap, up_cap,
    )
    return rows


# -------------------------------------------------------------------------- prefill
@triton.jit
def _grouped_kernel(
    X, x_stride, RowSrc, x_row_div,  # sorted row r reads X[RowSrc[r] // x_row_div]
    W, S, F, TileExpert, TileStart, TileEnd,
    Out, out_stride, Wts,  # output row = sorted row r (w13: act buffer; w2: rows scaled by Wts[RowSrc[r]])
    N, K: tl.constexpr,
    FMT: tl.constexpr, ACT: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr,
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
    x_rows = (src // x_row_div).to(tl.int64)
    n = pn * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        xt = tl.load(X + x_rows[:, None] * x_stride + kk[None, :], mask=mmask[:, None], other=0.0)
        if FMT == 0:
            KB: tl.constexpr = K // 2
            KS: tl.constexpr = K // 16
            j = tl.arange(0, BK // 2)
            wb = tl.load(W + e * N * KB + n[:, None] * KB + (k0 // 2 + j)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            wv = tl.reshape(tl.join(_e2m1_to_f32(wb & 15), _e2m1_to_f32(wb >> 4)), (BN, BK))
            js = tl.arange(0, BK // 16)
            sc = tl.load(S + e * N * KS + n[:, None] * KS + (k0 // 16 + js)[None, :], mask=nmask[:, None], other=0.0)
            wv = tl.reshape(tl.reshape(wv, (BN, BK // 16, 16)) * sc.to(tl.float32)[:, :, None], (BN, BK))
        else:
            KS2: tl.constexpr = K // 32
            wv = tl.load(W + e * N * K + n[:, None] * K + kk[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
            js = tl.arange(0, BK // 32)
            sc = tl.load(S + e * N * KS2 + n[:, None] * KS2 + (k0 // 32 + js)[None, :], mask=nmask[:, None], other=127)
            wv = tl.reshape(tl.reshape(wv, (BN, BK // 32, 32)) * _e8m0_to_f32(sc)[:, :, None], (BN, BK))
        acc = tl.dot(xt, tl.trans(wv.to(tl.bfloat16)), acc)
    if FMT == 0:
        acc = acc * tl.load(F + e)
    if ACT == 1:
        a = _swiglu_pairs_2d(acc, BM, BN, GATE_CAP, UP_CAP)
        no = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(Out + rm.to(tl.int64)[:, None] * out_stride + no[None, :], a.to(tl.bfloat16),
                 mask=mmask[:, None] & (no < N // 2)[None, :])
    else:
        wt = tl.load(Wts + src, mask=mmask, other=0.0)
        acc = acc.to(tl.bfloat16).to(tl.float32) * wt[:, None]
        tl.store(Out + rm.to(tl.int64)[:, None] * out_stride + n[None, :], acc, mask=mmask[:, None] & nmask[None, :])


def moe_prefill(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, w13: ExpertWeight, w2: ExpertWeight,
                gate_cap: float, up_cap: float, bm: int = 64) -> torch.Tensor:
    """Grouped path. Returns bf16 [T, H] (top-k summed in fp32)."""
    t, hdim = x.shape
    k = ids.shape[1]
    e, n13, _ = w13.shape
    inter = n13 // 2
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=e)
    # tiles: per expert ceil(count / bm)
    tiles_per = (counts + bm - 1) // bm
    n_tiles = int(tiles_per.sum())
    starts_e = torch.cumsum(counts, 0) - counts
    tile_expert = torch.repeat_interleave(torch.arange(e, device=x.device), tiles_per)
    tile_rank = torch.arange(n_tiles, device=x.device) - (torch.cumsum(tiles_per, 0) - tiles_per)[tile_expert]
    tile_start = (starts_e[tile_expert] + tile_rank * bm).to(torch.int32)
    tile_end = torch.minimum(tile_start + bm, (starts_e + counts)[tile_expert].to(torch.int32))
    tile_expert = tile_expert.to(torch.int32)
    row_src = order.to(torch.int32)  # sorted row -> flat pair index
    wts_f = wts.reshape(-1).float().contiguous()
    act = torch.empty(t * k, inter, dtype=torch.bfloat16, device=x.device)
    rows = torch.empty(t * k, hdim, dtype=torch.float32, device=x.device)
    f13 = w13.expert_factor if w13.fmt == "nvfp4" else wts_f
    f2 = w2.expert_factor if w2.fmt == "nvfp4" else wts_f
    bn = 64
    _grouped_kernel[(n_tiles, triton.cdiv(n13, bn))](
        x, x.stride(0), row_src, k, w13.weight, _scale_view(w13), f13, tile_expert, tile_start, tile_end,
        act, act.stride(0), wts_f, n13, hdim, _fmt_id(w13), 1, gate_cap, up_cap,
        BM=bm, BN=bn, BK=64, num_warps=4, num_stages=3,
    )
    ident = torch.arange(t * k, device=x.device, dtype=torch.int32)
    _grouped_kernel[(n_tiles, triton.cdiv(hdim, bn))](
        act, act.stride(0), ident, 1, w2.weight, _scale_view(w2), f2, tile_expert, tile_start, tile_end,
        rows, rows.stride(0), wts_f[order].contiguous(), hdim, inter, _fmt_id(w2), 0, gate_cap, up_cap,
        BM=bm, BN=bn, BK=64, num_warps=4, num_stages=3,
    )
    out = torch.zeros(t * k, hdim, dtype=torch.float32, device=x.device)
    out[order] = rows
    return out.view(t, k, hdim).sum(1).to(torch.bfloat16)
