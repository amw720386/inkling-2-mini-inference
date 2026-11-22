"""Low-M BF16 GEMV for sm_120 (idea from SGLang's sm12x_bf16_gemv: one CTA streams a slab
of output columns, M padded to 16 for tl.dot). Autotuned per (N, K); fp32 accumulation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pdl import PDL_ON, enable as _pdl_enable

MAX_M = 32

_CONFIGS = [
    triton.Config({"BN": bn, "BK": bk}, num_warps=nw, num_stages=ns)
    for bn in (16, 32, 64)
    for bk in (128, 256, 512)
    for nw in (4, 8)
    for ns in (3, 4)
]


@triton.autotune(cache_results=True, configs=_CONFIGS, key=["N", "K", "MP"])
@triton.jit
def _dense_gemv_kernel(X, W, Y, M, N, K, SX, SW, SY, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in tl.range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], mask=(cols[None, :] < N) & (ks[:, None] < K),
                    other=0.0)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        acc = tl.dot(x, w, acc)
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def gemv(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[M, N] = x[M, K] @ w[N, K]^T, M <= 32. Unit last strides required."""
    m, k = x.shape
    n = w.shape[0]
    assert m <= MAX_M and x.stride(1) == 1 and w.stride(1) == 1 and out.stride(1) == 1
    _dense_gemv_kernel[lambda meta: (triton.cdiv(n, meta["BN"]),)](
        x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0), MP=16 if m <= 16 else 32,
    )
    return out


def mm(x: torch.Tensor, w, out: torch.Tensor) -> torch.Tensor:
    """Dense projection: Triton GEMV for decode-sized M (bf16 or Fp8Weight); for larger M, cuBLAS (bf16) or the
    row-scaled FP8 GEMM (Fp8Weight)."""
    if isinstance(w, Fp8Weight):
        if x.shape[0] <= MAX_M:
            return fp8_gemv(x, w, out)
        return fp8_mm(x, w, out)
    if x.shape[0] <= MAX_M and x.stride(1) == 1 and out.stride(1) == 1 and out.is_contiguous():
        return gemv(x, w, out)
    return torch.mm(x, w.t(), out=out)


# ------------------------------------------------------------------ FP8 GEMM (prefill)
@triton.jit
def _quant_rows_fp8_kernel(X, sx, Q, S, K: tl.constexpr, BK: tl.constexpr):
    """Row r of bf16 X -> E4M3 Q[r] = X[r] / S[r], S[r] = max|X[r]| / 448."""
    r = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BK)
    km = k < K
    x = tl.load(X + r * sx + k, mask=km, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0), 1e-12) / 448.0
    tl.store(Q + r * K + k, (x / s).to(tl.float8e4nv), mask=km)
    tl.store(S + r, s)


_MM_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=nw, num_stages=ns)
    for bm in (64, 128)
    for bn in (64, 128, 256)
    for bk in (64, 128)
    for nw in (4, 8)
    for ns in (3, 4)
    if not (bm == 128 and bn == 256 and nw == 4)
]


def _prune_mm(configs, named_args, **kw):
    k = named_args["K"]
    return [c for c in configs if k % c.kwargs["BK"] == 0]


@triton.autotune(cache_results=True, configs=_MM_CONFIGS, key=["N", "K"], prune_configs_by={"early_config_prune": _prune_mm})
@triton.jit
def _fp8_mm_kernel(A, As, W, Ws, Out, M, N, K: tl.constexpr, so, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr = 8):
    """Out[M, N] (bf16) = (A8 @ W8^T) * As[row] * Ws[col]: row-scaled E4M3 operands, fp32 accumulation."""
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    group = pid // (GROUP * n_n)
    first = group * GROUP
    size = tl.minimum(n_m - first, GROUP)
    pm = first + (pid % (GROUP * n_n)) % size
    pn = (pid % (GROUP * n_n)) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm_ = rm < M
    nm = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm.to(tl.int64)[:, None] * K + kk[None, :], mask=mm_[:, None], other=0.0)
        w = tl.load(W + rn.to(tl.int64)[None, :] * K + kk[:, None], mask=nm[None, :], other=0.0)
        acc = tl.dot(a, w, acc)
    acc = acc * tl.load(As + rm, mask=mm_, other=0.0)[:, None] * tl.load(Ws + rn, mask=nm, other=0.0)[None, :]
    tl.store(Out + rm.to(tl.int64)[:, None] * so + rn[None, :], acc.to(Out.dtype.element_ty),
             mask=mm_[:, None] & nm[None, :])


def fp8_mm(x: torch.Tensor, w: Fp8Weight, out: torch.Tensor) -> torch.Tensor:
    """out[M, N] = x[M, K] @ W^T for prefill-sized M: x quantized per row to E4M3, FP8 tensor-core GEMM (twice the
    bf16 rate; GeForce Blackwell runs bf16 with fp32 accumulation at half rate)."""
    m, k = x.shape
    n = w.w8.shape[0]
    assert x.stride(1) == 1 and out.stride(1) == 1 and w.w8.stride(0) == k
    x8 = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty(m, dtype=torch.float32, device=x.device)
    _quant_rows_fp8_kernel[(m,)](x, x.stride(0), x8, xs, K=k, BK=triton.next_power_of_2(k), num_warps=8)
    _fp8_mm_kernel[lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)](
        x8, xs, w.w8, w.scale, out, m, n, k, out.stride(0),
    )
    return out


@triton.autotune(cache_results=True, configs=_CONFIGS, key=["N", "K", "MP"])
@triton.jit
def _fp8_gemv_kernel(X, W, Sc, Y, M, N, K, SX, SW, SY, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    """Y = (X @ W8^T) * scale[col]; W8 float8_e4m3 [N, K] with one fp32 scale per row."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in tl.range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], mask=(cols[None, :] < N) & (ks[:, None] < K),
                    other=0.0, eviction_policy="evict_first")
        x = tl.load(X + rows[:, None] * SX + ks[None, :], mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        acc = tl.dot(x, w.to(tl.bfloat16), acc)
    sc = tl.load(Sc + cols, mask=cols < N, other=0.0)
    tl.store(Y + rows[:, None] * SY + cols[None, :], (acc * sc[None, :]).to(Y.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


class Fp8Weight:
    """Row-scaled E4M3 copy of a bf16 [N, K] weight: scale[r] = max|w_r| / 448."""

    def __init__(self, w: torch.Tensor | None = None, *, w8=None, scale=None):
        if w is None:  # a view (e.g. rows of another Fp8Weight)
            self.w8, self.scale, self.shape = w8, scale, w8.shape
            return
        amax = w.float().abs().amax(1).clamp(min=1e-12)
        self.scale = (amax / 448.0).contiguous()
        self.w8 = (w.float() / self.scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        self.shape = w.shape

    def rows(self, a: int, b: int) -> "Fp8Weight":
        return Fp8Weight(w8=self.w8[a:b], scale=self.scale[a:b])

    def t(self):  # interface parity with tensors used as dense_mm weights
        raise TypeError("Fp8Weight is consumed by mm()/fp8_gemv only")


def fp8_gemv(x: torch.Tensor, w: Fp8Weight, out: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = w.w8.shape[0]
    assert m <= MAX_M
    _fp8_gemv_kernel[lambda meta: (triton.cdiv(n, meta["BN"]),)](
        x, w.w8, w.scale, out, m, n, k, x.stride(0), w.w8.stride(0), out.stride(0), MP=16 if m <= 16 else 32,
    )
    return out



@triton.autotune(cache_results=True, configs=[c for c in _CONFIGS if c.kwargs["BN"] <= 64], key=["N", "K", "MP", "FP8"])
@triton.jit
def _seg_gemv_kernel(X, x_stride_in, x_stride_m, W, Sc, Y, M, N, K, SW, SY, B1, B2, B3, B4,
                     BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr, FP8: tl.constexpr,
                     PDL: tl.constexpr = PDL_ON):
    """Y[:, n] = X[seg(n)] @ W[n] for column segments [0,B1) -> input 0, [B1,B2) -> 1, [B2,B3) -> 2,
    [B3,B4) -> 1, [B4,N) -> 2 (boundaries are multiples of BN). X: [3, M, K] (input stride may be 0)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    c0 = pid * BN
    inp = tl.where(c0 < B1, 0, tl.where(c0 < B2, 1, tl.where(c0 < B3, 2, tl.where(c0 < B4, 1, 2))))
    xb = X + inp.to(tl.int64) * x_stride_in
    cols = c0 + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in tl.range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], mask=(cols[None, :] < N) & (ks[:, None] < K),
                    other=0.0, eviction_policy="evict_first")
        x = tl.load(xb + rows[:, None] * x_stride_m + ks[None, :], mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        acc = tl.dot(x, w.to(tl.bfloat16), acc)
    if FP8:
        acc = acc * tl.load(Sc + cols, mask=cols < N, other=0.0)[None, :]
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def seg_gemv(x3: torch.Tensor, w, out: torch.Tensor, bounds: tuple[int, int, int, int]) -> torch.Tensor:
    """Multi-input projection in one launch. x3 [3, M, K] (or an expanded view with input stride 0);
    w bf16 [N, K] or Fp8Weight; out [M, N]; bounds (B1..B4) multiples of 64."""
    _, m, k = x3.shape
    fp8 = isinstance(w, Fp8Weight)
    wt, sc = (w.w8, w.scale) if fp8 else (w, out)
    n = wt.shape[0]
    assert m <= MAX_M and x3.stride(2) == 1 and out.stride(1) == 1 and all(b % 64 == 0 for b in bounds)
    _seg_gemv_kernel[lambda meta: (triton.cdiv(n, meta["BN"]),)](
        x3, x3.stride(0), x3.stride(1), wt, sc, out, m, n, k, wt.stride(0), out.stride(0), *bounds,
        MP=16 if m <= 16 else 32, FP8=fp8,
    )
    return out


_pdl_enable(_seg_gemv_kernel, _dense_gemv_kernel, _fp8_gemv_kernel)
