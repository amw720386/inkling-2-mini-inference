"""Fused glue for the DSpark drafter's block layers (Triton): residual add + RMS norm, per-head q/k RMS norm +
rotary, SiLU-gated product. Each replaces 4-30 eager torch ops (one launch each inside the draft graph) and keeps
their bf16 rounding points."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_kernel(Res, A, ResOut, W, Out, eps, D: tl.constexpr, BD: tl.constexpr, ADD: tl.constexpr):
    """ResOut = bf16(Res + A) (ADD); Out = bf16(rms(ResOut) * W)."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BD)
    cm = c < D
    x = tl.load(Res + r * D + c, mask=cm, other=0.0).to(tl.float32)
    if ADD:
        x = (x + tl.load(A + r * D + c, mask=cm, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(ResOut + r * D + c, x.to(tl.bfloat16), mask=cm)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
    y = x * rinv * tl.load(W + c, mask=cm, other=0.0).to(tl.float32)
    tl.store(Out + r * D + c, y.to(tl.bfloat16), mask=cm)


def add_rms(res: torch.Tensor, a: torch.Tensor | None, w: torch.Tensor, eps: float = 1e-6):
    """(new residual, normed) for bf16 rows [n, D]: residual = bf16(res + a) (res itself when a is None)."""
    n, d = res.shape
    out = torch.empty_like(res)
    new = torch.empty_like(res) if a is not None else res
    _add_rms_kernel[(n,)](res, a if a is not None else res, new, w, out, eps, D=d, BD=triton.next_power_of_2(d),
                          ADD=a is not None, num_warps=4)
    return new, out


@triton.jit
def _qk_rope_kernel(QKV, stride, Pos, QW, KW, QO, KO, eps, log2_theta, NQ: tl.constexpr, NK: tl.constexpr,
                    HD: tl.constexpr):
    """Head h of row r: rms over the head (weight), bf16, then neox rotary at the row's position."""
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    j = tl.arange(0, HALF)
    if h < NQ:
        src = QKV + r * stride + h * HD
        w = QW
        dst = QO + (r * NQ + h) * HD
    else:
        src = QKV + r * stride + NQ * HD + (h - NQ) * HD
        w = KW
        dst = KO + (r * NK + (h - NQ)) * HD
    x1 = tl.load(src + j).to(tl.float32)
    x2 = tl.load(src + HALF + j).to(tl.float32)
    rinv = 1.0 / tl.sqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / HD + eps)
    y1 = (x1 * rinv * tl.load(w + j).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * rinv * tl.load(w + HALF + j).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv = tl.exp2(-(2.0 * j.to(tl.float32) / HD) * log2_theta)
    ang = tl.load(Pos + r).to(tl.float32) * inv
    c, s = tl.cos(ang), tl.sin(ang)
    tl.store(dst + j, (y1 * c - y2 * s).to(tl.bfloat16))
    tl.store(dst + HALF + j, (y2 * c + y1 * s).to(tl.bfloat16))


def qk_norm_rope(qkv: torch.Tensor, positions: torch.Tensor, q_w, k_w, nq: int, nk: int, hd: int, theta: float,
                 eps: float = 1e-6):
    """q [n, nq, hd], k [n, nk, hd] (bf16) from the q / k columns of qkv rows [n, (nq + 2 nk) hd]."""
    n = qkv.shape[0]
    q = torch.empty(n, nq, hd, dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty(n, nk, hd, dtype=torch.bfloat16, device=qkv.device)
    _qk_rope_kernel[(n, nq + nk)](qkv, qkv.stride(0), positions, q_w, k_w, q, k, eps, math.log2(theta),
                                  NQ=nq, NK=nk, HD=hd, num_warps=1)
    return q, k


@triton.jit
def _silu_mul_kernel(GU, Out, N: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    cm = c < N
    g = tl.load(GU + r * 2 * N + c, mask=cm, other=0.0).to(tl.float32)
    u = tl.load(GU + r * 2 * N + N + c, mask=cm, other=0.0).to(tl.float32)
    tl.store(Out + r * N + c, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=cm)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """bf16(silu(gate) * up) of rows [n, 2N] = [gate | up]."""
    n, n2 = gu.shape
    half = n2 // 2
    out = torch.empty(n, half, dtype=torch.bfloat16, device=gu.device)
    bn = 1024
    _silu_mul_kernel[(n, triton.cdiv(half, bn))](gu, out, N=half, BN=bn, num_warps=4)
    return out


@triton.jit
def _ctx_mix_kernel(F, Alpha, Wn, Out, n, eps, D: tl.constexpr, BD: tl.constexpr, NT: tl.constexpr):
    """Out[j, r] = bf16(rms(sum_t alpha[j, t] * feats[r, t]) * w[j]): the drafter's context mixes (3 layers + f)."""
    j = tl.program_id(0)
    r = tl.program_id(1).to(tl.int64)
    c = tl.arange(0, BD)
    cm = c < D
    m = tl.zeros((BD,), tl.float32)
    for t in tl.static_range(NT):
        m += tl.load(F + (r * NT + t) * D + c, mask=cm, other=0.0).to(tl.float32) * tl.load(Alpha + j * NT + t).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(m * m, 0) / D + eps)
    y = m * rinv * tl.load(Wn + j * D + c, mask=cm, other=0.0).to(tl.float32)
    tl.store(Out + (j * n + r) * D + c, y.to(tl.bfloat16), mask=cm)


def ctx_mix(feats: torch.Tensor, alpha: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """feats [n, T, D] bf16, alpha [J, T], w [J, D] -> [J, n, D] bf16."""
    n, nt, d = feats.shape
    j = alpha.shape[0]
    out = torch.empty(j, n, d, dtype=torch.bfloat16, device=feats.device)
    _ctx_mix_kernel[(j, n)](feats, alpha, w, out, n, eps, D=d, BD=triton.next_power_of_2(d), NT=nt, num_warps=4)
    return out


@triton.jit
def _ctx_kv_store_kernel(KV, kv_stride, Pos, KW, KC, VC, CP, n, size, eps, log2_theta,
                         KVH: tl.constexpr, HD: tl.constexpr, HAS_POS: tl.constexpr):
    """Row r, KV head h: k = rope(bf16(rms(k) * w)), v as is, into ring slot pos % size (only the last `size`
    rows of the chunk are kept); local layers also record the slot's position."""
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    if r < n - size:
        return
    HALF: tl.constexpr = HD // 2
    j = tl.arange(0, HALF)
    pos = tl.load(Pos + r)
    slot = (pos % size).to(tl.int64)
    src = KV + r * kv_stride + h * HD
    x1 = tl.load(src + j).to(tl.float32)
    x2 = tl.load(src + HALF + j).to(tl.float32)
    rinv = 1.0 / tl.sqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / HD + eps)
    y1 = (x1 * rinv * tl.load(KW + j).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * rinv * tl.load(KW + HALF + j).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    ang = pos.to(tl.float32) * tl.exp2(-(2.0 * j.to(tl.float32) / HD) * log2_theta)
    cs, sn = tl.cos(ang), tl.sin(ang)
    dst = (slot * KVH + h) * HD
    tl.store(KC + dst + j, (y1 * cs - y2 * sn).to(tl.bfloat16))
    tl.store(KC + dst + HALF + j, (y2 * cs + y1 * sn).to(tl.bfloat16))
    d = tl.arange(0, HD)
    tl.store(VC + dst + d, tl.load(KV + r * kv_stride + KVH * HD + h * HD + d))
    if HAS_POS:
        if h == 0:
            tl.store(CP + slot, pos)


def ctx_kv_store(kv, positions, k_w, k_ctx, v_ctx, ctx_pos, size: int, kvh: int, hd: int, theta: float,
                 eps: float = 1e-6):
    n = kv.shape[0]
    _ctx_kv_store_kernel[(n, kvh)](kv, kv.stride(0), positions, k_w, k_ctx, v_ctx,
                                   ctx_pos if ctx_pos is not None else positions, n, size, eps, math.log2(theta),
                                   KVH=kvh, HD=hd, HAS_POS=ctx_pos is not None, num_warps=1)
