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
    """Dense projection: Triton GEMV for decode-sized M (bf16 or Fp8Weight), cuBLAS otherwise."""
    if isinstance(w, Fp8Weight):
        if x.shape[0] <= MAX_M:
            return fp8_gemv(x, w, out)
        return torch.mm(x, (w.w8.to(torch.bfloat16) * w.scale[:, None].to(torch.bfloat16)).t(), out=out)
    if x.shape[0] <= MAX_M and x.stride(1) == 1 and out.stride(1) == 1 and out.is_contiguous():
        return gemv(x, w, out)
    return torch.mm(x, w.t(), out=out)


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

    def __init__(self, w: torch.Tensor):
        amax = w.float().abs().amax(1).clamp(min=1e-12)
        self.scale = (amax / 448.0).contiguous()
        self.w8 = (w.float() / self.scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        self.shape = w.shape

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
