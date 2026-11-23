"""Fused per-token glue kernels for the fast forward (Triton).

The residual stream lives in a block buffer `blk` [NB, T, H] (bf16): slots 0..nb-1 are
the attention-residual blocks and slot nb holds the current partial. Depth attention
reads a contiguous slot range plus an optional fp32 source (the mid-layer residual).

Kernels (one launch each, grid over tokens):
  depth_attn      block attention residual (+ optional RMSNorm), single online pass
  qkv_post        q/r clip + log-scale, relative logits, K/V short conv + clip / rms,
                  optional cache write and (decode) conv-state update
  post_attn       attention-output conv + residual + MLP-input depth attention + norm
  router_logits / topk
  post_mlp        top-k row sum + MLP-output conv + residual -> next partial
  msg_gate_down / msg_gate_up   loop message controllers

"decode state" mode (every sequence has exactly one new token) lets the per-token
program read and rewrite its own conv state; otherwise a separate state kernel runs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pdl import PDL_ON, enable as _pdl_enable, prefetch_rows_l2

# persistent (graph-safe) scratch for message-norm block sums, keyed by shape
_SS: dict = {}


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _rms_inv(ss, n: tl.constexpr, eps):
    return 1.0 / tl.sqrt(ss / n + eps)


# ----------------------------------------------------------------------- depth attn
@triton.jit
def _depth_attn_row_loop(Src, src_stride_s, src_row, ns, Extra, extra_row, HAS_EXTRA: tl.constexpr,
                         q, hmask, H: tl.constexpr, BH: tl.constexpr, eps, xe_reg, EXTRA_REG: tl.constexpr):
    h = tl.arange(0, BH)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((BH,), tl.float32)
    for s in range(ns):
        x = tl.load(Src + s * src_stride_s + src_row + h, mask=hmask, other=0.0).to(tl.float32)
        rinv = _rms_inv(tl.sum(x * x, 0), H, eps)
        logit = tl.sum(x * q, 0) * rinv
        m_new = tl.maximum(m_i, logit)
        a = tl.exp(m_i - m_new)
        p = tl.exp(logit - m_new)
        l_i = l_i * a + p
        acc = acc * a + (p * rinv) * x
        m_i = m_new
    if HAS_EXTRA:
        if EXTRA_REG:
            x = xe_reg
        else:
            x = tl.load(Extra + extra_row + h, mask=hmask, other=0.0).to(tl.float32)
        rinv = _rms_inv(tl.sum(x * x, 0), H, eps)
        logit = tl.sum(x * q, 0) * rinv
        m_new = tl.maximum(m_i, logit)
        a = tl.exp(m_i - m_new)
        p = tl.exp(logit - m_new)
        l_i = l_i * a + p
        acc = acc * a + (p * rinv) * x
    return acc / l_i


@triton.jit
def _depth_attn_row(Src, src_stride_s, src_row, ns, Extra, extra_row, HAS_EXTRA: tl.constexpr,
                    q, hmask, H: tl.constexpr, BH: tl.constexpr, eps, NS: tl.constexpr = 8, xe_reg=None,
                    EXTRA_REG: tl.constexpr = False, TILE: tl.constexpr = False):
    """Softmax attention of one token row over its sources (at most NS, plus an optional extra row); returns fp32
    [BH]. TILE (decode-sized steps): all sources load as one [NS, BH] tile, so their loads overlap and the
    reductions run together. Otherwise (prefill, thousands of programs) an online softmax over the sources one at
    a time: the tile's registers would cut occupancy (prefill depth attention 6.8 -> 16.5 ms with it)."""
    if not TILE:
        return _depth_attn_row_loop(Src, src_stride_s, src_row, ns, Extra, extra_row, HAS_EXTRA, q, hmask, H, BH,
                                    eps, xe_reg, EXTRA_REG)
    h = tl.arange(0, BH)
    sv = tl.arange(0, NS)
    sm = sv < ns
    x = tl.load(Src + sv[:, None] * src_stride_s + src_row + h[None, :], mask=sm[:, None] & hmask[None, :],
                other=0.0).to(tl.float32)
    rinv = _rms_inv(tl.sum(x * x, 1), H, eps)  # [NS]
    logit = tl.where(sm, tl.sum(x * q[None, :], 1) * rinv, float("-inf"))
    m = tl.max(logit, 0)
    if HAS_EXTRA:
        if EXTRA_REG:  # the caller's row, already in registers
            xe = xe_reg
        else:
            xe = tl.load(Extra + extra_row + h, mask=hmask, other=0.0).to(tl.float32)
        rinv_e = _rms_inv(tl.sum(xe * xe, 0), H, eps)
        logit_e = tl.sum(xe * q, 0) * rinv_e
        m = tl.maximum(m, logit_e)
    p = tl.where(sm, tl.exp(logit - m), 0.0)
    l = tl.sum(p, 0)
    acc = tl.sum((p * rinv)[:, None] * x, 0)
    if HAS_EXTRA:
        pe = tl.exp(logit_e - m)
        l += pe
        acc += (pe * rinv_e) * xe
    return acc / l


@triton.jit
def _depth_attn_kernel(Src, src_stride_s, src_stride_t, ns, Extra, extra_stride_t, Q, Wn,
                       Out, out_stride_q, out_stride_t, eps,
                       HAS_EXTRA: tl.constexpr, NORM: tl.constexpr, OUT_F32: tl.constexpr,
                       H: tl.constexpr, BH: tl.constexpr, TILE: tl.constexpr = False, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    qi = tl.program_id(1)
    h = tl.arange(0, BH)
    hm = h < H
    q = tl.load(Q + qi * H + h, mask=hm, other=0.0)
    o = _depth_attn_row(Src, src_stride_s, t * src_stride_t, ns, Extra, t * extra_stride_t, HAS_EXTRA,
                        q, hm, H, BH, eps, TILE=TILE)
    if NORM:
        o = o * _rms_inv(tl.sum(o * o, 0), H, eps) * tl.load(Wn + qi * H + h, mask=hm, other=0.0).to(tl.float32)
    dst = Out + qi * out_stride_q + t * out_stride_t + h
    if OUT_F32:
        tl.store(dst, o, mask=hm)
    else:
        tl.store(dst, o.to(tl.bfloat16), mask=hm)


@triton.jit
def _depth_attn2_kernel(SrcA, a_stride_s, a_stride_t, nsa, QA, OutA, oa_stride_q, oa_stride_t, NQA: tl.constexpr,
                        SrcB, b_stride_s, b_stride_t, nsb, QB, OutB, ob_stride_q, ob_stride_t, eps,
                        H: tl.constexpr, BH: tl.constexpr, TILE: tl.constexpr = False, PDL: tl.constexpr = PDL_ON):
    """Two unnormalized depth attentions in one launch: queries 0..NQA-1 over source set A, the rest over B
    (a loop visit's attention input and its messages). fp32 outputs."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    qi = tl.program_id(1)
    h = tl.arange(0, BH)
    hm = h < H
    if qi < NQA:
        q = tl.load(QA + qi * H + h, mask=hm, other=0.0)
        o = _depth_attn_row(SrcA, a_stride_s, t * a_stride_t, nsa, SrcA, 0, False, q, hm, H, BH, eps, TILE=TILE)
        tl.store(OutA + qi * oa_stride_q + t * oa_stride_t + h, o, mask=hm)
    else:
        qb = qi - NQA
        q = tl.load(QB + qb * H + h, mask=hm, other=0.0)
        o = _depth_attn_row(SrcB, b_stride_s, t * b_stride_t, nsb, SrcB, 0, False, q, hm, H, BH, eps, TILE=TILE)
        tl.store(OutB + qb * ob_stride_q + t * ob_stride_t + h, o, mask=hm)


def depth_attn2(src_a, ns_a: int, q_a, out_a, src_b, ns_b: int, q_b, out_b, eps=1e-6):
    _, t, h = src_a.shape
    nqa, nqb = q_a.shape[0], q_b.shape[0]
    _depth_attn2_kernel[(t, nqa + nqb)](
        src_a, src_a.stride(0), src_a.stride(1), ns_a, q_a, out_a, out_a.stride(0), out_a.stride(1), nqa,
        src_b, src_b.stride(0), src_b.stride(1), ns_b, q_b, out_b, out_b.stride(0), out_b.stride(1), eps,
        H=h, BH=triton.next_power_of_2(h), TILE=t <= 32, num_warps=8,
    )


def depth_attn(src, ns: int, queries, out, *, extra=None, norm_w=None, out_f32=False, eps=1e-6):
    """src: [S, T, H] bf16 (first ns slots used); queries fp32 [Q, H] (pre-scaled);
    out: [Q, T, H] (bf16, or fp32 when out_f32). norm_w: [Q, H] bf16 or None."""
    s_, t, h = src.shape
    nq = queries.shape[0]
    _depth_attn_kernel[(t, nq)](
        src, src.stride(0), src.stride(1), ns, extra if extra is not None else src,
        extra.stride(0) if extra is not None else 0, queries, norm_w if norm_w is not None else queries,
        out, out.stride(0), out.stride(1), eps,
        HAS_EXTRA=extra is not None, NORM=norm_w is not None, OUT_F32=out_f32, H=h,
        BH=triton.next_power_of_2(h), TILE=t <= 32, num_warps=8,
    )
    return out


# ------------------------------------------------------------------------- qkv post
@triton.jit
def _tap_w(Wt, Static, static_stride, Dyn, dyn_off, ch, cm, lag: tl.constexpr, K: tl.constexpr, DYN: tl.constexpr):
    """Weight of the tap `lag` tokens back (fp32 [len(ch)])."""
    if DYN:
        return (tl.load(Static + lag * static_stride + ch, mask=cm, other=0.0).to(tl.float32)
                + tl.load(Dyn + dyn_off + lag).to(tl.float32))
    else:
        return tl.load(Wt + ch * K + (K - 1 - lag), mask=cm, other=0.0).to(tl.float32)


@triton.jit
def _conv_prev(X, x_stride, t, i, State, st_base, st_row, ch, cm, Wt, Static, static_stride, Dyn, dyn_off,
               K: tl.constexpr, DYN: tl.constexpr):
    """sum over lags 1..K-1 of tap * x[i - lag] (chunk rows or conv state), fp32."""
    acc = tl.zeros(ch.shape, tl.float32)
    for lag in tl.static_range(1, K):
        src = i - lag
        inc = src >= 0
        xv = tl.load(X + (t - lag) * x_stride + ch, mask=cm & inc, other=0.0)
        sv = tl.load(State + st_base + (K - 1 + src) * st_row + ch, mask=cm & (src < 0), other=0.0)
        acc += (xv + sv).to(tl.float32) * _tap_w(Wt, Static, static_stride, Dyn, dyn_off, ch, cm, lag, K, DYN)
    return acc


@triton.jit
def _write_decode_state(X, x_stride, t, State, st_base, st_row, ch, cm):
    """One new token: state rows <- [row1, row2, x]."""
    r1 = tl.load(State + st_base + st_row + ch, mask=cm)
    r2 = tl.load(State + st_base + 2 * st_row + ch, mask=cm)
    tl.store(State + st_base + ch, r1, mask=cm)
    tl.store(State + st_base + st_row + ch, r2, mask=cm)
    tl.store(State + st_base + 2 * st_row + ch, tl.load(X + t * x_stride + ch, mask=cm), mask=cm)


@triton.jit
def _write_cand(X, x_stride, t, i, State, st_base, st_row, Cand, cand_base, cand_row, ch, cm):
    """Verify: the conv state after committing rows 0..i of this sequence = last 3 of [old state, x_0..x_i]."""
    for r in tl.static_range(3):
        src = i - 2 + r
        inx = src >= 0
        xv = tl.load(X + (t - 2 + r) * x_stride + ch, mask=cm & inx, other=0.0)
        sv = tl.load(State + st_base + (3 + src) * st_row + ch, mask=cm & (src < 0), other=0.0)
        tl.store(Cand + cand_base + r * cand_row + ch, tl.where(inx, xv, sv), mask=cm)


@triton.jit
def _l2clip_vec(x, thr):
    n = tl.sqrt(tl.sum(x * x, axis=-1, keep_dims=True))
    return x * tl.minimum(thr / (n + 1e-6), 1.0)


@triton.jit
def _qkv_post_kernel(
    QP, q_stride,  # q | gate | r | sink row (q at col 0, r at col R_COL)
    KL, VL, kv_stride,  # k_lin, v_lin rows
    DynK, DynV, dyn_stride,  # [T, KVH*K] dyn conv weights for K and V (global only)
    Qs, Rs, Proj,  # q_scale [H, D], rel_scale [H, DR], rel_proj [DR, RE]
    KW, VW, KStatic, VStatic,  # local conv weights [KVD, K] | global static [K, KVD]
    KState, VState, st_slot, st_row,  # [slots, K-1, KVD]
    QOut, ROut, KOut, VOut,  # [T,H,D], [T,H,RE], [T,KVH,D], [T,KVH,D]
    KCache, VCache, c_slot, c_pos,  # cache for direct write (WRITE_CACHE)
    TokSeq, SeqSlot, SeqStart, SeqOff, Pos,
    qk_thr, rel_thr, log_alpha, log_floor, conv_scale, eps,
    H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, DR: tl.constexpr, RE: tl.constexpr,
    R_COL: tl.constexpr, K: tl.constexpr,
    GLOBAL: tl.constexpr, WRITE_CACHE: tl.constexpr, RING: tl.constexpr, DECODE_STATE: tl.constexpr,
    KV_FP8: tl.constexpr = False,
    CandK=None, CandV=None, cand_sb=0, cand_sc=0, cand_sr=0, CAND: tl.constexpr = False,
    PDL: tl.constexpr = PDL_ON,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1)
    part = tl.program_id(2)  # 0..GRP-1: one q/rel head of the group; GRP: the K/V convs
    GRP: tl.constexpr = H // KVH
    sq = tl.load(TokSeq + t)
    slot = tl.load(SeqSlot + sq).to(tl.int64)
    off = tl.load(SeqOff + sq)
    i = t - off
    pos = tl.load(Pos + t)
    tau = 1.0 + log_alpha * tl.log(tl.maximum((pos + 1).to(tl.float32) / log_floor, 1.0))

    # one q head of this group (part < GRP) ---------------------------------------
    d = tl.arange(0, D)
    if part < GRP:
        hh = kvh * GRP + part
        x = tl.load(QP + t * q_stride + hh * D + d).to(tl.float32) * tl.load(Qs + hh * D + d).to(tl.float32)
        x = _l2clip_vec(x, qk_thr).to(tl.bfloat16)
        if GLOBAL:
            x = (x.to(tl.float32) * tau).to(tl.bfloat16)
        tl.store(QOut + (t * H + hh) * D + d, x)
        # relative states -> logits over distance
        dr = tl.arange(0, DR)
        r = tl.load(QP + t * q_stride + R_COL + hh * DR + dr).to(tl.float32) * tl.load(Rs + hh * DR + dr).to(tl.float32)
        r = _l2clip_vec(r, rel_thr).to(tl.bfloat16)
        if GLOBAL:
            r = (r.to(tl.float32) * tau).to(tl.bfloat16)
        e = tl.arange(0, RE)
        pt = tl.load(Proj + dr[:, None] * RE + e[None, :]).to(tl.float32)  # [DR, RE]
        logits = tl.sum(r.to(tl.float32)[:, None] * pt, axis=0)
        tl.store(ROut + (t * H + hh) * RE + e, logits.to(tl.bfloat16))

    if part < GRP:
        return
    # k / v: short conv over this head's channels ------------------------------
    KVD: tl.constexpr = KVH * D
    ch = kvh * D + d
    cm = ch < KVD
    st_base = slot * st_slot
    dyn_k = t * dyn_stride + kvh * K
    dyn_v = t * dyn_stride + kvh * K
    xk = tl.load(KL + t * kv_stride + ch).to(tl.float32)
    xv = tl.load(VL + t * kv_stride + ch).to(tl.float32)
    ka = _conv_prev(KL, kv_stride, t, i, KState, st_base, st_row, ch, cm, KW, KStatic, KVD, DynK, dyn_k, K, GLOBAL)
    va = _conv_prev(VL, kv_stride, t, i, VState, st_base, st_row, ch, cm, VW, VStatic, KVD, DynV, dyn_v, K, GLOBAL)
    ka += xk * _tap_w(KW, KStatic, KVD, DynK, dyn_k, ch, cm, 0, K, GLOBAL)
    va += xv * _tap_w(VW, VStatic, KVD, DynV, dyn_v, ch, cm, 0, K, GLOBAL)
    k = ((ka + xk) * conv_scale).to(tl.bfloat16).to(tl.float32)
    v = ((va + xv) * conv_scale).to(tl.bfloat16).to(tl.float32)
    k = _l2clip_vec(k, qk_thr).to(tl.bfloat16)
    v = (v * _rms_inv(tl.sum(v * v, 0), D, eps)).to(tl.bfloat16)
    if KV_FP8:  # every later read sees the E4M3 value, this call's own keys included
        k = tl.minimum(tl.maximum(k.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv).to(tl.bfloat16)
        v = tl.minimum(tl.maximum(v.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv).to(tl.bfloat16)
    tl.store(KOut + (t * KVH + kvh) * D + d, k)
    tl.store(VOut + (t * KVH + kvh) * D + d, v)
    if WRITE_CACHE:
        cp = (pos % RING) if RING > 0 else pos
        dst = slot * c_slot + cp.to(tl.int64) * c_pos + kvh * D + d
        tl.store(KCache + dst, k.to(KCache.dtype.element_ty))
        tl.store(VCache + dst, v.to(VCache.dtype.element_ty))
    if DECODE_STATE:
        _write_decode_state(KL, kv_stride, t, KState, st_base, st_row, ch, cm)
        _write_decode_state(VL, kv_stride, t, VState, st_base, st_row, ch, cm)
    if CAND:
        cb = sq.to(tl.int64) * cand_sb + i * cand_sc
        _write_cand(KL, kv_stride, t, i, KState, st_base, st_row, CandK, cb, cand_sr, ch, cm)
        _write_cand(VL, kv_stride, t, i, VState, st_base, st_row, CandV, cb, cand_sr, ch, cm)


def qkv_post(qp, r_col, k_lin, v_lin, dyn_kv, lw, kstate, vstate, q_out, r_out, k_out, v_out, meta, tok_seq,
             positions, cfg, *, is_global, cache_kv=None, ring=0, decode_state=False, kv_fp8=False, cand=None):
    """dyn_kv: (dyn_k, dyn_v) [T, KVH*K] views with equal row stride (global layers)."""
    t = q_out.shape[0]
    kvh, d = cfg.num_kv_heads, cfg.head_dim
    assert k_lin.stride(0) == v_lin.stride(0)
    kc, vc = cache_kv if cache_kv is not None else (k_out, k_out)
    _qkv_post_kernel[(t, kvh, cfg.num_heads // kvh + 1)](
        qp, qp.stride(0), k_lin, v_lin, k_lin.stride(0),
        dyn_kv[0] if dyn_kv is not None else qp, dyn_kv[1] if dyn_kv is not None else qp,
        dyn_kv[0].stride(0) if dyn_kv is not None else 0,
        lw.q_scale, lw.rel_scale, lw.rel_proj,
        lw.k_conv_w if lw.k_conv_w is not None else lw.q_scale, lw.v_conv_w if lw.v_conv_w is not None else lw.q_scale,
        lw.k_static if lw.k_static is not None else lw.q_scale, lw.v_static if lw.v_static is not None else lw.q_scale,
        kstate, vstate, kstate.stride(0), kstate.stride(1), q_out, r_out, k_out, v_out,
        kc, vc, kc.stride(0) if cache_kv is not None else 0, kc.stride(1) if cache_kv is not None else 0,
        tok_seq, meta.seq_slot, meta.seq_start, meta.seq_off, positions,
        cfg.qk_l2_clip, cfg.rel_l2_clip, cfg.log_scaling_alpha, cfg.log_scaling_n_floor,
        cfg.dyn_conv_out_scale if is_global else cfg.conv_out_scale, cfg.rms_eps,
        H=cfg.num_heads, KVH=kvh, D=d, DR=cfg.d_rel, RE=cfg.rel_extent, R_COL=r_col, K=cfg.sconv_kernel,
        GLOBAL=is_global, WRITE_CACHE=cache_kv is not None, RING=ring, DECODE_STATE=decode_state,
        KV_FP8=kv_fp8,
        CandK=cand[0] if cand else qp, CandV=cand[1] if cand else qp,
        cand_sb=cand[0].stride(0) if cand else 0, cand_sc=cand[0].stride(1) if cand else 0,
        cand_sr=cand[0].stride(2) if cand else 0, CAND=cand is not None,
        num_warps=4,
    )


# ------------------------------------------------------------------------ post attn
@triton.jit
def _post_attn_kernel(
    O, o_stride, CW, State, st_slot, st_row,  # attn output, conv weight [H, K], state [slots, K-1, H]
    Src, src_stride_s, src_stride_t, ns, part_slot,  # block buffer; partial at slot part_slot
    Resid,  # fp32 [T, H] out
    Q, Wn, Out, out_stride_t,  # mlp query fp32 [H], norm weight; out bf16 normed (or fp32 when not NORM)
    TokSeq, SeqSlot, SeqOff, conv_scale, eps,
    BOUNDARY: tl.constexpr, NORM: tl.constexpr, DECODE_STATE: tl.constexpr,
    H: tl.constexpr, BH: tl.constexpr, K: tl.constexpr,
    Cand=None, cand_sb=0, cand_sc=0, cand_sr=0, CAND: tl.constexpr = False,
    TILE: tl.constexpr = False,
    PDL: tl.constexpr = PDL_ON,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    hm = h < H
    sq = tl.load(TokSeq + t)
    slot = tl.load(SeqSlot + sq).to(tl.int64)
    i = t - tl.load(SeqOff + sq)
    st_base = slot * st_slot
    x = tl.load(O + t * o_stride + h, mask=hm, other=0.0).to(tl.float32)
    a = _conv_prev(O, o_stride, t, i, State, st_base, st_row, h, hm, CW, CW, 0, CW, 0, K, False)
    a += x * _tap_w(CW, CW, 0, CW, 0, h, hm, 0, K, False)
    a = ((a + x) * conv_scale).to(tl.bfloat16).to(tl.float32)
    if DECODE_STATE:
        _write_decode_state(O, o_stride, t, State, st_base, st_row, h, hm)
    if CAND:
        _write_cand(O, o_stride, t, i, State, st_base, st_row, Cand, sq.to(tl.int64) * cand_sb + i * cand_sc,
                    cand_sr, h, hm)
    if BOUNDARY:
        resid = a
    else:
        resid = tl.load(Src + part_slot * src_stride_s + t * src_stride_t + h, mask=hm, other=0.0).to(tl.float32) + a
    tl.store(Resid + t * H + h, resid, mask=hm)
    q = tl.load(Q + h, mask=hm, other=0.0)
    o = _depth_attn_row(Src, src_stride_s, t * src_stride_t, ns, Resid, t * H, True, q, hm, H, BH, eps,
                        xe_reg=tl.where(hm, resid, 0.0), EXTRA_REG=True, TILE=TILE)
    if NORM:
        o = o * _rms_inv(tl.sum(o * o, 0), H, eps) * tl.load(Wn + h, mask=hm, other=0.0).to(tl.float32)
        tl.store(Out + t * out_stride_t + h, o.to(tl.bfloat16), mask=hm)
    else:
        tl.store(Out + t * out_stride_t + h, o, mask=hm)


def post_attn(o, conv_w, state, blk, ns, part_slot, resid, query, norm_w, out, meta, tok_seq, cfg, *,
              boundary, decode_state, cand=None):
    t, h = o.shape
    _post_attn_kernel[(t,)](
        o, o.stride(0), conv_w, state, state.stride(0), state.stride(1), blk, blk.stride(0), blk.stride(1), ns,
        part_slot, resid, query, norm_w if norm_w is not None else query, out, out.stride(0),
        tok_seq, meta.seq_slot, meta.seq_off, cfg.conv_out_scale, cfg.rms_eps,
        BOUNDARY=boundary, NORM=norm_w is not None, DECODE_STATE=decode_state, H=h,
        BH=triton.next_power_of_2(h), K=cfg.sconv_kernel,
        Cand=cand if cand is not None else o, cand_sb=cand.stride(0) if cand is not None else 0,
        cand_sc=cand.stride(1) if cand is not None else 0, cand_sr=cand.stride(2) if cand is not None else 0,
        CAND=cand is not None, TILE=t <= 32, num_warps=8,
    )


# --------------------------------------------------------------------------- router
@triton.jit
def _select_topk(lg, LRow, bias, ea, E: tl.constexpr, TOPK: tl.constexpr):
    """Top-k of sigmoid(lg) + bias (ties -> lowest expert), weights sigmoid(lg) of the picks. Each pick is one
    max over int64 keys (order-preserving score bits << 32 | E-1-expert) instead of max + argmin + gather, so a
    multi-warp program pays one cross-warp reduction per pick; the picks' weights are re-read from LRow."""
    sel = tl.sigmoid(lg) + bias
    b = sel.to(tl.int32, bitcast=True)
    key = ((b ^ ((b >> 31) & 0x7FFFFFFF)).to(tl.int64) << 32) | (E - 1 - ea).to(tl.int64)
    ks = tl.arange(0, TOPK)
    ids = tl.zeros((TOPK,), tl.int32)
    for j in tl.static_range(TOPK):
        mk = tl.max(key, 0)
        ids = tl.where(ks == j, (E - 1 - (mk & 0xFFFFFFFF)).to(tl.int32), ids)
        key = tl.where(key == mk, -9223372036854775807 - 1, key)
    ws = tl.sigmoid(tl.load(LRow + ids, volatile=True))
    return ids, ws


@triton.jit
def _router_topk_kernel(X, x_stride, W, Bias, Logits, Counter, Ids, Wts, scale,
                        H: tl.constexpr, BH: tl.constexpr, E: tl.constexpr, BE: tl.constexpr, TOPK: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    """Each program computes BE expert logits of one token with one load of its weight rows; the program
    that finishes last for the token (atomic counter) selects the top-k and resets the counter."""
    if PDL:
        pe0 = tl.program_id(1) * BE + tl.arange(0, BE)
        prefetch_rows_l2(W, pe0, pe0 < E, H * 2, 8192)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    pe = tl.program_id(1)
    h = tl.arange(0, BH)
    hm = h < H
    e = pe * BE + tl.arange(0, BE)
    x = tl.load(X + t * x_stride + h, mask=hm, other=0.0).to(tl.float32)
    w = tl.load(W + e[:, None] * H + h[None, :], mask=hm[None, :], other=0.0, eviction_policy="evict_last")
    tl.store(Logits + t * E + e, tl.sum(w.to(tl.float32) * x[None, :], 1))
    NPROG: tl.constexpr = E // BE
    done = tl.atomic_add(Counter + t, 1, sem="acq_rel")
    if done == NPROG - 1:
        ea = tl.arange(0, E)
        lg = tl.load(Logits + t * E + ea, volatile=True)
        ids, ws = _select_topk(lg, Logits + t * E, tl.load(Bias + ea), ea, E, TOPK)
        ks = tl.arange(0, TOPK)
        ws = ws / (tl.sum(ws, 0) + 1e-20) * scale
        tl.store(Ids + t * TOPK + ks, ids)
        tl.store(Wts + t * TOPK + ks, ws)
        tl.atomic_xchg(Counter + t, 0)


_ROUTER_COUNTERS: dict = {}


@triton.jit
def _topk_kernel(Logits, l_stride, Bias, Ids, Wts, E: tl.constexpr, TOPK: tl.constexpr, scale, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    ea = tl.arange(0, E)
    lg = tl.load(Logits + t * l_stride + ea)
    ids, ws = _select_topk(lg, Logits + t * l_stride, tl.load(Bias + ea), ea, E, TOPK)
    ks = tl.arange(0, TOPK)
    ws = ws / (tl.sum(ws, 0) + 1e-20) * scale
    tl.store(Ids + t * TOPK + ks, ids)
    tl.store(Wts + t * TOPK + ks, ws)


@triton.jit
def _router_gemm_kernel(X, x_stride, W, Out, T, K: tl.constexpr, E: tl.constexpr, BM: tl.constexpr,
                        BK: tl.constexpr, BE: tl.constexpr = 0, PREC: tl.constexpr = "tf32x3"):
    """Prefill router logits X @ W^T at about fp32 accuracy on the TF32 tensor cores (3xTF32 split products)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < T
    r64 = rows.to(tl.int64)
    NE: tl.constexpr = BE if BE > 0 else E
    e = tl.program_id(1) * NE + tl.arange(0, NE)
    acc = tl.zeros((BM, NE), tl.float32)
    for k0 in tl.range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(X + r64[:, None] * x_stride + kk[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        w = tl.load(W + e[:, None] * K + kk[None, :])
        acc = tl.dot(x, tl.trans(w), acc, input_precision=PREC)
    tl.store(Out + r64[:, None] * E + e[None, :], acc, mask=rm[:, None])


def route(x, w_bf16, bias, logits_buf, ids, wts, top_k, scale, w_f32=None):
    t, h = x.shape
    e = w_bf16.shape[0]
    if t >= PREFILL_GEMM_MIN_TOKENS and w_f32 is not None and x.dtype == torch.bfloat16:
        logits = torch.empty(t, e, dtype=torch.float32, device=x.device)
        _router_gemm_kernel[(triton.cdiv(t, 16), e // 64)](x, x.stride(0), w_f32, logits, t, K=h, E=e, BM=16,
                                                           BK=64, BE=64, num_warps=4, num_stages=3)
        _topk_kernel[(t,)](logits, logits.stride(0), bias, ids, wts, E=e, TOPK=top_k, scale=scale, num_warps=4)
        return
    key = (x.device.index, logits_buf.data_ptr())
    cnt = _ROUTER_COUNTERS.get(key)
    if cnt is None or cnt.numel() < logits_buf.shape[0]:
        cnt = torch.zeros(logits_buf.shape[0], dtype=torch.int32, device=x.device)
        _ROUTER_COUNTERS[key] = cnt
    be = 1
    _router_topk_kernel[(t, e // be)](x, x.stride(0), w_bf16, bias, logits_buf, cnt, ids, wts, scale,
                                      H=h, BH=triton.next_power_of_2(h), E=e, BE=be, TOPK=top_k, num_warps=4)


# ------------------------------------------------------------------------- post mlp
@triton.jit
def _moe_sum_kernel(Rows, M, H: tl.constexpr, TOPK: tl.constexpr, BH: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    hm = h < H
    acc = tl.zeros((BH,), tl.float32)
    for j in tl.static_range(TOPK):
        acc += tl.load(Rows + (t * TOPK + j) * H + h, mask=hm, other=0.0)
    tl.store(M + t * H + h, acc.to(tl.bfloat16), mask=hm)


@triton.jit
def _post_mlp_kernel(
    M, m_stride, Rows, CW, State, st_slot, st_row, Resid, Dst, dst_stride,
    TokSeq, SeqSlot, SeqOff, conv_scale,
    FROM_ROWS: tl.constexpr, TOPK: tl.constexpr, DECODE_STATE: tl.constexpr,
    H: tl.constexpr, BH: tl.constexpr, K: tl.constexpr,
    Cand=None, cand_sb=0, cand_sc=0, cand_sr=0, CAND: tl.constexpr = False,
    PDL: tl.constexpr = PDL_ON,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BH + tl.arange(0, BH)  # every channel is independent: split over programs
    hm = h < H
    sq = tl.load(TokSeq + t)
    slot = tl.load(SeqSlot + sq).to(tl.int64)
    i = t - tl.load(SeqOff + sq)
    st_base = slot * st_slot
    if FROM_ROWS:  # decode: this token's rows only; neighbors come from state
        acc = tl.zeros((BH,), tl.float32)
        for j in tl.static_range(TOPK):
            acc += tl.load(Rows + (t * TOPK + j) * H + h, mask=hm, other=0.0)
        mb = acc.to(tl.bfloat16)
        tl.store(M + t * m_stride + h, mb, mask=hm)
        x = mb.to(tl.float32)
    else:
        x = tl.load(M + t * m_stride + h, mask=hm, other=0.0).to(tl.float32)
    c = _conv_prev(M, m_stride, t, i, State, st_base, st_row, h, hm, CW, CW, 0, CW, 0, K, False)
    c += x * _tap_w(CW, CW, 0, CW, 0, h, hm, 0, K, False)
    c = ((c + x) * conv_scale).to(tl.bfloat16).to(tl.float32)
    if DECODE_STATE:
        _write_decode_state(M, m_stride, t, State, st_base, st_row, h, hm)
    if CAND:
        _write_cand(M, m_stride, t, i, State, st_base, st_row, Cand, sq.to(tl.int64) * cand_sb + i * cand_sc,
                    cand_sr, h, hm)
    out = tl.load(Resid + t * H + h, mask=hm, other=0.0) + c
    tl.store(Dst + t * dst_stride + h, out.to(tl.bfloat16), mask=hm)


def post_mlp(rows, m_buf, conv_w, state, resid, dst, meta, tok_seq, cfg, *, decode_state, cand=None):
    t, h = resid.shape
    bh = triton.next_power_of_2(h)
    if not decode_state and rows is not None:
        _moe_sum_kernel[(t,)](rows, m_buf, H=h, TOPK=cfg.top_k, BH=bh, num_warps=8)
    bc = 256
    _post_mlp_kernel[(t, triton.cdiv(h, bc))](
        m_buf, m_buf.stride(0), rows if rows is not None else m_buf, conv_w, state, state.stride(0), state.stride(1), resid, dst, dst.stride(0),
        tok_seq, meta.seq_slot, meta.seq_off, cfg.conv_out_scale,
        FROM_ROWS=decode_state, TOPK=cfg.top_k, DECODE_STATE=decode_state, H=h, BH=bc, K=cfg.sconv_kernel,
        Cand=cand if cand is not None else m_buf, cand_sb=cand.stride(0) if cand is not None else 0,
        cand_sc=cand.stride(1) if cand is not None else 0, cand_sr=cand.stride(2) if cand is not None else 0,
        CAND=cand is not None, num_warps=4,
    )


# ------------------------------------------------------------------ loop messages
@triton.jit
def _msg_down_kernel(Hs, hs_stride_g, hs_stride_t, Ms, ms_stride_g, ms_stride_t, Down, G, eps,
                     GATE0: tl.constexpr, NGATE: tl.constexpr, H: tl.constexpr, BH: tl.constexpr,
                     RANK: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    """g[t, j, r] = bf16(down_j @ bf16(cat(rms(h_j), rms(m_j)))) for gates j."""
    if PDL:
        r0 = tl.program_id(2) * BR + tl.arange(0, BR)
        prefetch_rows_l2(Down, (GATE0 + tl.program_id(1)) * RANK + r0, r0 < RANK, 2 * H * 2, 16384)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    pr = tl.program_id(2)
    hrow = Hs + j * hs_stride_g + t * hs_stride_t
    mrow = Ms + j * ms_stride_g + t * ms_stride_t
    hh = tl.arange(0, BH)
    hm = hh < H
    xa = tl.load(hrow + hh, mask=hm, other=0.0)
    ma = tl.load(mrow + hh, mask=hm, other=0.0)
    rx = _rms_inv(tl.sum(xa * xa, 0), H, eps)
    rm = _rms_inv(tl.sum(ma * ma, 0), H, eps)
    r = pr * BR + tl.arange(0, BR)
    base = Down + (GATE0 + j) * RANK * 2 * H
    acc = tl.zeros((BR,), tl.float32)
    for k0 in tl.range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        km = k < H
        xn = (tl.load(hrow + k, mask=km, other=0.0) * rx).to(tl.bfloat16).to(tl.float32)
        mn = (tl.load(mrow + k, mask=km, other=0.0) * rm).to(tl.bfloat16).to(tl.float32)
        w1 = tl.load(base + r[:, None] * (2 * H) + k[None, :], mask=km[None, :], other=0.0).to(tl.float32)
        w2 = tl.load(base + r[:, None] * (2 * H) + H + k[None, :], mask=km[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w1 * xn[None, :] + w2 * mn[None, :], 1)
    tl.store(G + (t * NGATE + j) * RANK + r, acc.to(tl.bfloat16).to(tl.float32))


@triton.jit
def _msg_up_kernel(Hs, hs_stride_g, hs_stride_t, Ms, ms_stride_g, ms_stride_t, G, Gain, Forget,
                   Dst, dst_stride_g, dst_stride_t, SS, NBLK: tl.constexpr,
                   GATE0: tl.constexpr, NGATE: tl.constexpr, NORM: tl.constexpr,
                   H: tl.constexpr, RANK: tl.constexpr, BC: tl.constexpr,
                   Counter=None, Wn=None, wn_stride=0, Out=None, out_stride_g=0, out_stride_t=0, eps=1e-6,
                   BH: tl.constexpr = 4096, PDL: tl.constexpr = PDL_ON):
    """o = (1 - tanh(F g)) * h + tanh(G g) * m for one H block; fp32 to Dst. With NORM, each block also
    writes its sum of squares to SS[t, j, blk], and the block that finishes last for (t, j) (atomic counter)
    applies the RMS norm with weight to the whole row, writing bf16 Out."""
    if PDL:
        c0 = tl.program_id(2) * BC + tl.arange(0, BC)
        prefetch_rows_l2(Forget, (GATE0 + tl.program_id(1)) * H + c0, c0 < H, RANK * 2, 128)
        prefetch_rows_l2(Gain, (GATE0 + tl.program_id(1)) * H + c0, c0 < H, RANK * 2, 128)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    pb = tl.program_id(2)
    rr = tl.arange(0, RANK)
    g = tl.load(G + (t * NGATE + j) * RANK + rr)
    gi = (GATE0 + j) * H * RANK
    c = pb * BC + tl.arange(0, BC)
    cm = c < H
    f = tl.load(Forget + gi + c[:, None] * RANK + rr[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
    gg = tl.load(Gain + gi + c[:, None] * RANK + rr[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
    fg = tl.sum(f * g[None, :], 1)
    ga = tl.sum(gg * g[None, :], 1)
    x = tl.load(Hs + j * hs_stride_g + t * hs_stride_t + c, mask=cm, other=0.0)
    m = tl.load(Ms + j * ms_stride_g + t * ms_stride_t + c, mask=cm, other=0.0)
    o = (1.0 - _tanh(fg)) * x + _tanh(ga) * m
    tl.store(Dst + j * dst_stride_g + t * dst_stride_t + c, o, mask=cm)
    if NORM:
        tl.store(SS + (t * NGATE + j) * NBLK + pb, tl.sum(tl.where(cm, o * o, 0.0), 0))
        done = tl.atomic_add(Counter + t * NGATE + j, 1, sem="acq_rel")
        if done == tl.num_programs(2) - 1:
            b = tl.arange(0, NBLK)
            rinv = _rms_inv(tl.sum(tl.load(SS + (t * NGATE + j) * NBLK + b, volatile=True), 0), H, eps)
            hh = tl.arange(0, BH)
            hm = hh < H
            row = tl.load(Dst + j * dst_stride_g + t * dst_stride_t + hh, mask=hm, other=0.0, volatile=True) * rinv
            row = row * tl.load(Wn + j * wn_stride + hh, mask=hm, other=0.0).to(tl.float32)
            tl.store(Out + j * out_stride_g + t * out_stride_t + hh, row.to(tl.bfloat16), mask=hm)
            tl.atomic_xchg(Counter + t * NGATE + j, 0)


@triton.jit
def _msg_norm_kernel(Tmp, tmp_stride_g, tmp_stride_t, SS, Wn, wn_stride, Out, out_stride_g, out_stride_t, eps,
                     NGATE: tl.constexpr, NBLK: tl.constexpr, H: tl.constexpr, BH: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    b = tl.arange(0, NBLK)
    rinv = _rms_inv(tl.sum(tl.load(SS + (t * NGATE + j) * NBLK + b), 0), H, eps)
    h = tl.arange(0, BH)
    hm = h < H
    o = tl.load(Tmp + j * tmp_stride_g + t * tmp_stride_t + h, mask=hm, other=0.0) * rinv
    o = o * tl.load(Wn + j * wn_stride + h, mask=hm, other=0.0).to(tl.float32)
    tl.store(Out + j * out_stride_g + t * out_stride_t + h, o.to(tl.bfloat16), mask=hm)


PREFILL_GEMM_MIN_TOKENS = 64  # above this, the per-token kernels re-read their weights too often


@triton.jit
def _msg_prep_kernel(Hs, h_stride_t, Ms, m_stride_t, Cat, eps, H: tl.constexpr, BH: tl.constexpr):
    """Prefill: cat[t] = [rmsnorm(h_t), rmsnorm(m_t)] (no weight), bf16 [T, 2H]."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BH)
    cm = c < H
    h = tl.load(Hs + t * h_stride_t + c, mask=cm, other=0.0)
    m = tl.load(Ms + t * m_stride_t + c, mask=cm, other=0.0)
    tl.store(Cat + t * 2 * H + c, (h * _rms_inv(tl.sum(h * h, 0), H, eps)).to(tl.bfloat16), mask=cm)
    tl.store(Cat + t * 2 * H + H + c, (m * _rms_inv(tl.sum(m * m, 0), H, eps)).to(tl.bfloat16), mask=cm)


@triton.jit
def _msg_up_tile_kernel(G, Hs, h_stride_t, Ms, m_stride_t, Wf, Wg, Dst, dst_stride_t, SS, T,
                        NORM: tl.constexpr, H: tl.constexpr, RANK: tl.constexpr, NBLK: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr):
    """Prefill, (BM tokens) x (BN channels) per program: o = (1 - tanh(g Wf^T)) h + tanh(g Wg^T) m, fp32 to Dst
    (3xTF32 dots, about fp32 accuracy); with NORM also the block's sums of squares to SS[t, blk]."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    pb = tl.program_id(1)
    rm = rows < T
    r64 = rows.to(tl.int64)
    kk = tl.arange(0, RANK)
    n = pb * BN + tl.arange(0, BN)
    # g and the gate weights are bf16: a bf16 MMA with fp32 accumulation forms their products exactly
    g = tl.load(G + r64[:, None] * RANK + kk[None, :], mask=rm[:, None], other=0.0)
    wf = tl.load(Wf + n[:, None] * RANK + kk[None, :])
    wg = tl.load(Wg + n[:, None] * RANK + kk[None, :])
    f = _tanh(tl.dot(g, tl.trans(wf)))
    ga = _tanh(tl.dot(g, tl.trans(wg)))
    h = tl.load(Hs + r64[:, None] * h_stride_t + n[None, :], mask=rm[:, None], other=0.0)
    m = tl.load(Ms + r64[:, None] * m_stride_t + n[None, :], mask=rm[:, None], other=0.0)
    o = (1.0 - f) * h + ga * m
    tl.store(Dst + r64[:, None] * dst_stride_t + n[None, :], o, mask=rm[:, None])
    if NORM:
        tl.store(SS + r64 * NBLK + pb, tl.sum(o * o, 1), mask=rm)


@triton.jit
def _msg_norm_tile_kernel(Src, src_stride_t, SS, Wn, Out, out_stride_t, T, eps, H: tl.constexpr,
                          NBLK: tl.constexpr, NBLK_P2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < T
    r64 = rows.to(tl.int64)
    b = tl.arange(0, NBLK_P2)
    ss = tl.sum(tl.load(SS + r64[:, None] * NBLK + b[None, :], mask=rm[:, None] & (b < NBLK)[None, :], other=0.0), 1)
    rinv = _rms_inv(ss, H, eps)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    o = tl.load(Src + r64[:, None] * src_stride_t + n[None, :], mask=rm[:, None], other=0.0)
    o = o * rinv[:, None] * tl.load(Wn + n).to(tl.float32)[None, :]
    tl.store(Out + r64[:, None] * out_stride_t + n[None, :], o.to(tl.bfloat16), mask=rm[:, None])


def _msg_gates_gemm(hs, ms, lw, gate0, n_gate, out, cfg, norm_w, tmp=None):
    """Same math as the per-token kernels, for prefill: a norm/concat kernel, one cuBLAS GEMM for the down
    projection, a tiled kernel for the rank-64 up projections and the gated combine, and a tiled output norm."""
    eps = cfg.rms_eps
    _, t, h = hs.shape
    rank = cfg.gate_rank
    cat = torch.empty(t, 2 * h, dtype=torch.bfloat16, device=hs.device)
    bh = triton.next_power_of_2(h)
    bm, bn = 64, 128
    assert h % bn == 0
    nblk = h // bn
    ss = torch.empty(t, nblk, dtype=torch.float32, device=hs.device) if norm_w is not None else cat
    grid = (triton.cdiv(t, bm), nblk)
    for j in range(n_gate):
        hj, mj = hs[j], ms[j]
        _msg_prep_kernel[(t,)](hj, hj.stride(0), mj, mj.stride(0), cat, eps, H=h, BH=bh, num_warps=8)
        g = cat @ lw.ctl_down[gate0 + j].t()  # [T, rank] bf16
        if norm_w is not None:
            dst = tmp[j] if tmp is not None else torch.empty(t, h, dtype=torch.float32, device=hs.device)
        else:
            dst = out[j]
        _msg_up_tile_kernel[grid](
            g, hj, hj.stride(0), mj, mj.stride(0), lw.ctl_forget[gate0 + j], lw.ctl_gain[gate0 + j],
            dst, dst.stride(0), ss, t, NORM=norm_w is not None, H=h, RANK=rank, NBLK=nblk, BM=bm, BN=bn,
            num_warps=8, num_stages=1,
        )
        if norm_w is not None:
            _msg_norm_tile_kernel[grid](dst, dst.stride(0), ss, norm_w[j], out[j], out[j].stride(0), t, eps, H=h,
                                        NBLK=nblk, NBLK_P2=triton.next_power_of_2(nblk), BM=bm, BN=bn, num_warps=8)


def msg_gates(hs, ms, lw, gate0: int, n_gate: int, g_buf, out, cfg, *, norm_w=None, tmp=None):
    """hs/ms: [n_gate, T, H] fp32 views (gate stride may be 0 to broadcast).
    out: [n_gate, T, H] bf16 (normed with norm_w [n_gate, H]; needs fp32 `tmp` with out's
    strides) or fp32 (no norm)."""
    _, t, h = hs.shape
    if t >= PREFILL_GEMM_MIN_TOKENS:
        return _msg_gates_gemm(hs, ms, lw, gate0, n_gate, out, cfg, norm_w, tmp)
    rank = cfg.gate_rank
    _msg_down_kernel[(t, n_gate, rank // 4)](
        hs, hs.stride(0), hs.stride(1), ms, ms.stride(0), ms.stride(1), lw.ctl_down, g_buf, cfg.rms_eps,
        GATE0=gate0, NGATE=n_gate, H=h, BH=triton.next_power_of_2(h), RANK=rank, BR=4, BK=1024, num_warps=4,
    )
    bc = 128
    nblk = triton.cdiv(h, bc)
    nblk_p2 = triton.next_power_of_2(nblk)
    norm = norm_w is not None
    dst = tmp if norm else out
    if norm:
        assert tmp is not None and tmp.dtype == torch.float32
        key = (hs.device.index, t, n_gate, nblk_p2)
        if key not in _SS:  # persistent (graph-safe) block sums and counters
            _SS[key] = (torch.zeros(t, n_gate, nblk_p2, dtype=torch.float32, device=hs.device),
                        torch.zeros(t, n_gate, dtype=torch.int32, device=hs.device))
        ss, counter = _SS[key]
    else:
        ss, counter = g_buf, g_buf
    _msg_up_kernel[(t, n_gate, nblk)](
        hs, hs.stride(0), hs.stride(1), ms, ms.stride(0), ms.stride(1), g_buf, lw.ctl_gain, lw.ctl_forget,
        dst, dst.stride(0), dst.stride(1), ss, nblk_p2,
        GATE0=gate0, NGATE=n_gate, NORM=norm, H=h, RANK=rank, BC=bc,
        Counter=counter, Wn=norm_w if norm else g_buf, wn_stride=norm_w.stride(0) if norm else 0,
        Out=out, out_stride_g=out.stride(0), out_stride_t=out.stride(1), eps=cfg.rms_eps,
        BH=triton.next_power_of_2(h), num_warps=4,
    )


_pdl_enable(_depth_attn_kernel, _depth_attn2_kernel, _qkv_post_kernel, _post_attn_kernel, _router_topk_kernel, _topk_kernel, _moe_sum_kernel, _post_mlp_kernel, _msg_down_kernel, _msg_up_kernel, _msg_norm_kernel)
