"""Short causal convolutions with per-slot state (Triton), varlen over a batch.

out[t] = (sum_j w[:, j] * x[t - (K-1) + j] + x[t]) * scale, with x before the chunk read
from the slot's state (the last K-1 inputs). Dynamic variant (global K/V): taps are
static[lag] + dyn[t, head, lag] with lag 0 the current token.

Two kernels: outputs (token x channel tiles, parallel), then the state update (one
program per seq x channel tile), so no program reads state another one overwrites.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pdl import PDL_ON, enable as _pdl_enable


@triton.jit
def _conv_out_kernel(
    X, x_stride, Out, out_stride, State, st_slot, st_row,  # State [slots, K-1, C] (slot/row strides)
    Wt, Static, Dyn, dyn_stride,  # Wt [C, K] (static conv) | Static [K, C] + Dyn [T, NH*K] (dynamic)
    TokSeq, SeqSlot, SeqStart, SeqOff,  # per-token seq id; per-seq slot / start pos / token offset
    T, C: tl.constexpr, K: tl.constexpr, scale,
    DYN: tl.constexpr, CH_PER_HEAD: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr,
    PDL: tl.constexpr = PDL_ON,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    tmask = t < T
    sq = tl.load(TokSeq + t, mask=tmask, other=0)
    off = tl.load(SeqOff + sq, mask=tmask, other=0)
    slot = tl.load(SeqSlot + sq, mask=tmask, other=0).to(tl.int64)
    i = t - off  # index within the chunk
    m2 = tmask[:, None] & cm[None, :]
    xc = tl.load(X + t.to(tl.int64)[:, None] * x_stride + c[None, :], mask=m2, other=0.0).to(tl.float32)
    acc = xc  # residual
    for j in tl.static_range(K):
        lag = K - 1 - j  # tap j multiplies x[i - lag]
        src = i - lag
        in_chunk = src >= 0
        xv_chunk = tl.load(X + (t - lag).to(tl.int64)[:, None] * x_stride + c[None, :],
                           mask=(tmask & in_chunk)[:, None] & cm[None, :], other=0.0)
        srow = (K - 1) + src  # row in state when src < 0
        xv_state = tl.load(State + slot[:, None] * st_slot + srow.to(tl.int64)[:, None] * st_row + c[None, :],
                           mask=(tmask & ~in_chunk)[:, None] & cm[None, :], other=0.0)
        xv = (xv_chunk + xv_state).to(tl.float32)
        if DYN:
            w = tl.load(Static + lag * C + c, mask=cm, other=0.0).to(tl.float32)[None, :]
            head = c // CH_PER_HEAD
            dw = tl.load(Dyn + t.to(tl.int64)[:, None] * dyn_stride + head[None, :] * K + lag,
                         mask=m2, other=0.0).to(tl.float32)
            acc += xv * (w + dw)
        else:
            w = tl.load(Wt + c * K + j, mask=cm, other=0.0).to(tl.float32)
            acc += xv * w[None, :]
    tl.store(Out + t.to(tl.int64)[:, None] * out_stride + c[None, :], (acc * scale).to(Out.dtype.element_ty),
             mask=m2)


@triton.jit
def _conv_state_kernel(X, x_stride, State, st_slot, st_row, SeqSlot, SeqOff, SeqN,
                       C: tl.constexpr, K: tl.constexpr, BC: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    sq = tl.program_id(0)
    pc = tl.program_id(1)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    slot = tl.load(SeqSlot + sq).to(tl.int64)
    off = tl.load(SeqOff + sq)
    n = tl.load(SeqN + sq)
    base = State + slot * st_slot
    tl.static_assert(K == 4, "state kernel is written for kernel size 4")
    # gather the new last 3 inputs first (they may come from the old state), then store
    v0 = _new_state_row(X, x_stride, base, st_row, off, n, 0, c, cm)
    v1 = _new_state_row(X, x_stride, base, st_row, off, n, 1, c, cm)
    v2 = _new_state_row(X, x_stride, base, st_row, off, n, 2, c, cm)
    tl.store(base + c, v0, mask=cm)
    tl.store(base + st_row + c, v1, mask=cm)
    tl.store(base + 2 * st_row + c, v2, mask=cm)


@triton.jit
def _new_state_row(X, x_stride, base, st_row, off, n, r: tl.constexpr, c, cm):
    src = n - 3 + r  # index within chunk
    from_x = src >= 0
    vx = tl.load(X + (off + src).to(tl.int64) * x_stride + c, mask=from_x & cm, other=0.0)
    vs = tl.load(base + (3 + src).to(tl.int64) * st_row + c, mask=(~from_x) & cm, other=0.0)
    return tl.where(from_x, vx, vs)


def short_conv(x, state, meta, tok_seq, weight=None, static=None, dyn=None, ch_per_head=1, scale=1.0,
               out=None, max_seqs=None):
    """x [T, C] bf16 (rows contiguous in C); state [slots, K-1, C]; meta: AttnMeta-like with
    seq_slot/seq_start/seq_off/seq_n; tok_seq [T] int32. Updates state in place."""
    t, c = x.shape
    km1 = state.shape[1]
    k = km1 + 1
    out = out if out is not None else torch.empty_like(x)
    dynamic = dyn is not None
    bt, bc = (16, 256) if t >= 16 else (1, 512)
    _conv_out_kernel[(triton.cdiv(t, bt), triton.cdiv(c, bc))](
        x, x.stride(0), out, out.stride(0), state, state.stride(0), state.stride(1),
        weight if weight is not None else x, static if dynamic else x, dyn if dynamic else x,
        dyn.stride(0) if dynamic else 0,
        tok_seq, meta.seq_slot, meta.seq_start, meta.seq_off,
        t, c, k, scale, DYN=dynamic, CH_PER_HEAD=ch_per_head, BT=bt, BC=bc, num_warps=4,
    )
    ns = max_seqs if max_seqs is not None else meta.seq_slot.shape[0]
    _conv_state_kernel[(ns, triton.cdiv(c, 512))](
        x, x.stride(0), state, state.stride(0), state.stride(1), meta.seq_slot, meta.seq_off, meta.seq_n,
        c, k, BC=512, num_warps=4,
    )
    return out


@triton.jit
def _conv_cand_kernel(X, x_stride, State, st_slot, st_row, Cand, cand_stride_b, cand_stride_c, cand_row,
                      SeqSlot, SeqOff, C: tl.constexpr, BC: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    """Speculative verify: for sequence b of the batch (slot SeqSlot[b], rows SeqOff[b]..), Cand[b, c] = the
    conv state after committing its first c + 1 rows (last 3 of [old state, x_0..x_c]); State is untouched."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    bq = tl.program_id(0)
    cidx = tl.program_id(1)
    pc = tl.program_id(2)
    ch = pc * BC + tl.arange(0, BC)
    cm = ch < C
    slot = tl.load(SeqSlot + bq).to(tl.int64)
    off = tl.load(SeqOff + bq).to(tl.int64)
    n = cidx + 1
    for r in tl.static_range(3):
        src = n - 3 + r
        from_x = src >= 0
        vx = tl.load(X + (off + src).to(tl.int64) * x_stride + ch, mask=from_x & cm, other=0.0)
        vs = tl.load(State + slot * st_slot + (3 + src).to(tl.int64) * st_row + ch, mask=(~from_x) & cm, other=0.0)
        tl.store(Cand + bq * cand_stride_b + cidx * cand_stride_c + r * cand_row + ch, tl.where(from_x, vx, vs),
                 mask=cm)


def conv_candidates(x, state, cand, seq_slot, seq_off, n_seq: int, m: int):
    """x [n_seq * m, C] chunk inputs (sequence b at rows seq_off[b]..+m); state [slots, 3, C];
    cand [>= n_seq, >= m, 3, C]."""
    c = x.shape[1]
    _conv_cand_kernel[(n_seq, m, triton.cdiv(c, 512))](
        x, x.stride(0), state, state.stride(0), state.stride(1), cand, cand.stride(0), cand.stride(1),
        cand.stride(2), seq_slot, seq_off, c, BC=512, num_warps=4)


_pdl_enable(_conv_out_kernel, _conv_state_kernel, _conv_cand_kernel)
