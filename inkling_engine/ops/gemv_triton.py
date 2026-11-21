"""Low-M BF16 GEMV for sm_120 (idea from SGLang's sm12x_bf16_gemv: one CTA streams a slab
of output columns, M padded to 16 for tl.dot). Autotuned per (N, K); fp32 accumulation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_M = 16

_CONFIGS = [
    triton.Config({"BN": bn, "BK": bk}, num_warps=nw, num_stages=ns)
    for bn in (16, 32, 64)
    for bk in (128, 256, 512)
    for nw in (4, 8)
    for ns in (3, 4)
]


@triton.autotune(configs=_CONFIGS, key=["N", "K"])
@triton.jit
def _dense_gemv_kernel(X, W, Y, M, N, K, SX, SW, SY, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
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
    """out[M, N] = x[M, K] @ w[N, K]^T, M <= 16. Unit last strides required."""
    m, k = x.shape
    n = w.shape[0]
    assert m <= MAX_M and x.stride(1) == 1 and w.stride(1) == 1 and out.stride(1) == 1
    _dense_gemv_kernel[lambda meta: (triton.cdiv(n, meta["BN"]),)](
        x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0), MP=16,
    )
    return out


def mm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Dense projection: Triton GEMV for decode-sized M, cuBLAS otherwise."""
    if x.shape[0] <= MAX_M and x.stride(1) == 1 and out.stride(1) == 1 and out.is_contiguous():
        return gemv(x, w, out)
    return torch.mm(x, w.t(), out=out)
