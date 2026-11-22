"""Inkling attention (Triton): GQA + per-query relative bias + sliding window + sink + gate.

One kernel serves prefill chunks and decode/verify steps.

Rows of a tile are (token, head-in-group) pairs so one K/V tile load serves all query
heads of a KV head. Keys come from two sources: the cache (positions before the chunk;
local layers read a ring indexed pos % ring) and the chunk's freshly projected K/V.

Bias: rel[tok, h, d] for backward distance d < R, zero beyond.
Sink: one extra logit per (tok, h) whose value vector is zero.
Output: softmax(...) @ V, multiplied by sigmoid(gate) and written bf16.
Split-K: programs over key-block ranges write (m, l, acc) partials; a combine kernel
merges them with the sink. With one split the kernel finalizes directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .pdl import PDL_ON, enable as _pdl_enable

G4 = 4  # padded GQA group (3 query heads per KV head for Inkling-2-Mini)


@dataclass
class AttnMeta:
    """Device tensors describing the batch's tiles (static buffers under CUDA graphs)."""

    seq_slot: torch.Tensor  # [S] int32
    seq_start: torch.Tensor  # [S] position of first new token
    seq_off: torch.Tensor  # [S] token offset in the flattened batch
    seq_n: torch.Tensor  # [S] new tokens
    tile_seq: torch.Tensor  # [n_tiles] int32
    tile_q0: torch.Tensor  # [n_tiles] first token (within seq) of the tile
    n_tiles: int
    tok_per_tile: int
    n_split: int
    max_keys: int  # upper bound on keys per tile (for split sizing)

    @classmethod
    def build(cls, seqs, offsets, tok_per_tile: int, n_split: int, device="cuda") -> "AttnMeta":
        slot, start, off, n, tseq, tq0 = [], [], [], [], [], []
        max_keys = 1
        for i, s in enumerate(seqs):
            slot.append(s.slot)
            start.append(s.start)
            off.append(offsets[i])
            n.append(s.n)
            for q0 in range(0, s.n, tok_per_tile):
                tseq.append(i)
                tq0.append(q0)
            max_keys = max(max_keys, s.start + s.n)
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
        return cls(i32(slot), i32(start), i32(off), i32(n), i32(tseq), i32(tq0), len(tseq), tok_per_tile,
                   n_split, max_keys)


_ATTN_CONFIGS = [
    triton.Config({"BN": bn}, num_warps=nw, num_stages=ns)
    for bn in (32, 64, 128)
    for nw in (4, 8)
    for ns in (1, 2, 3)
]


@triton.autotune(cache_results=True, configs=_ATTN_CONFIGS, key=["TOK", "NSPLIT", "WINDOW", "RING", "D", "H"])
@triton.jit
def _attn_kernel(
    Q, q_stride_t,  # [T, H, D] (row stride q_stride_t, head stride D)
    KC, VC, kc_stride_t,  # chunk K/V [T, KVH, D]
    KCache, VCache, cache_stride_slot, cache_stride_pos,  # [slots, CAP, KVH, D]
    Rel, rel_stride_t,  # [T, H, R]
    Sinks, sink_stride_t,  # [T, H] (strided view)
    Gate, gate_stride_t,  # [T, H*D] (strided view)
    Out, out_stride_t,  # [T, H*D] bf16
    PartAcc, PartML,  # split partials: [T, H, NSPLIT, D] fp32, [T, H, NSPLIT, 2] fp32
    SeqSlot, SeqStart, SeqOff, SeqN, TileSeq, TileQ0,
    scale,
    H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    WINDOW: tl.constexpr,  # 0 = full causal
    RING: tl.constexpr,  # ring capacity for local layers (0 = contiguous cache)
    TOK: tl.constexpr, BN: tl.constexpr, NSPLIT: tl.constexpr,
    PDL: tl.constexpr = PDL_ON,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    tile = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    GRP: tl.constexpr = H // KVH
    ROWS: tl.constexpr = TOK * 4

    sq = tl.load(TileSeq + tile)
    q0 = tl.load(TileQ0 + tile)
    slot = tl.load(SeqSlot + sq).to(tl.int64)
    start = tl.load(SeqStart + sq)
    off = tl.load(SeqOff + sq)
    n = tl.load(SeqN + sq)

    row = tl.arange(0, ROWS)
    tok = q0 + row // 4  # token index within seq
    hg = row % 4
    row_ok = (tok < n) & (hg < GRP)
    head = kvh * GRP + hg
    t_glob = (off + tok).to(tl.int64)
    qpos = start + tok

    d = tl.arange(0, D)
    q = tl.load(Q + t_glob[:, None] * q_stride_t + head[:, None] * D + d[None, :], mask=row_ok[:, None], other=0.0)

    last_tok = tl.minimum(q0 + TOK, n) - 1
    kmax = start + last_tok  # inclusive
    if WINDOW > 0:
        kmin = tl.maximum(start + q0 - (WINDOW - 1), 0)
    else:
        kmin = 0
    nblk = tl.cdiv(kmax - kmin + 1, BN)
    per = tl.cdiv(nblk, NSPLIT)
    b0 = split * per
    b1 = tl.minimum(b0 + per, nblk)

    m_i = tl.full((ROWS,), float("-inf"), tl.float32)
    l_i = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    kcache_base = KCache + slot * cache_stride_slot + kvh * D
    vcache_base = VCache + slot * cache_stride_slot + kvh * D
    for b in tl.range(b0, b1):
        kpos = kmin + b * BN + tl.arange(0, BN)
        k_ok = kpos <= kmax
        from_cache = kpos < start
        if RING > 0:
            cidx = (kpos % RING).to(tl.int64)
        else:
            cidx = kpos.to(tl.int64)
        crow = (off + kpos - start).to(tl.int64)
        mc = k_ok & from_cache
        mn = k_ok & (kpos >= start)
        k_c = tl.load(kcache_base + cidx[:, None] * cache_stride_pos + d[None, :], mask=mc[:, None], other=0.0).to(tl.bfloat16)
        k_n = tl.load(KC + crow[:, None] * kc_stride_t + kvh * D + d[None, :], mask=mn[:, None], other=0.0)
        k = k_c + k_n
        s = tl.dot(q, tl.trans(k)) * scale
        dist = qpos[:, None] - kpos[None, :]
        valid = (dist >= 0) & k_ok[None, :] & row_ok[:, None]
        if WINDOW > 0:
            valid = valid & (dist < WINDOW)
        in_rel = valid & (dist < R)
        bias = tl.load(Rel + t_glob[:, None] * rel_stride_t + head[:, None] * R + dist, mask=in_rel, other=0.0)
        s = tl.where(valid, s + bias.to(tl.float32), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v_c = tl.load(vcache_base + cidx[:, None] * cache_stride_pos + d[None, :], mask=mc[:, None], other=0.0).to(tl.bfloat16)
        v_n = tl.load(VC + crow[:, None] * kc_stride_t + kvh * D + d[None, :], mask=mn[:, None], other=0.0)
        v = v_c + v_n
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    if NSPLIT == 1:
        sink = tl.load(Sinks + t_glob * sink_stride_t + head, mask=row_ok, other=0.0).to(tl.float32)
        m_f = tl.maximum(m_i, sink)
        a = tl.exp(m_i - m_f)
        denom = l_i * a + tl.exp(sink - m_f)
        o = acc * (a / denom)[:, None]
        g = tl.load(Gate + t_glob[:, None] * gate_stride_t + head[:, None] * D + d[None, :], mask=row_ok[:, None], other=0.0)
        o = o.to(tl.bfloat16).to(tl.float32) * tl.sigmoid(g.to(tl.float32))
        tl.store(Out + t_glob[:, None] * out_stride_t + head[:, None] * D + d[None, :], o.to(tl.bfloat16), mask=row_ok[:, None])
    else:
        pidx = (t_glob * H + head) * NSPLIT + split
        tl.store(PartAcc + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
        tl.store(PartML + pidx * 2, m_i, mask=row_ok)
        tl.store(PartML + pidx * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _combine_kernel(PartAcc, PartML, Sinks, sink_stride_t, Gate, gate_stride_t, Out, out_stride_t,
                    H: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    if PDL:
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    base = (t * H + h) * NSPLIT
    m = tl.load(PartML + (base + sp) * 2)
    l = tl.load(PartML + (base + sp) * 2 + 1)
    sink = tl.load(Sinks + t * sink_stride_t + h).to(tl.float32)
    m_f = tl.maximum(tl.max(m, 0), sink)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_f))
    denom = tl.sum(l * w, 0) + tl.exp(sink - m_f)
    acc = tl.load(PartAcc + (base + sp)[:, None] * D + d[None, :])
    o = tl.sum(acc * w[:, None], 0) / denom
    g = tl.load(Gate + t * gate_stride_t + h * D + d).to(tl.float32)
    o = o.to(tl.bfloat16).to(tl.float32) * tl.sigmoid(g)
    tl.store(Out + t * out_stride_t + h * D + d, o.to(tl.bfloat16))


def inkling_attention(q, k_new, v_new, k_cache, v_cache, rel, sinks, gate, meta: AttnMeta, *,
                      window: int, ring: int, scale: float, out=None, part_acc=None, part_ml=None,
                      bn: int = 64, num_warps: int = 4, num_stages: int = 2):
    """q [T, H, D]; k_new/v_new [T, KVH, D]; k_cache/v_cache [slots, CAP, KVH, D];
    rel [T, H, R]; sinks [T, H]; gate [T, H*D]. Returns gated attention [T, H*D] bf16.

    Does not write the cache; call `store_kv` afterwards.
    """
    t, h, d = q.shape
    kvh = k_new.shape[1]
    r = rel.shape[-1]
    assert h // kvh <= G4 and q.stride(2) == 1 and q.stride(1) == d
    out = out if out is not None else torch.empty(t, h * d, dtype=torch.bfloat16, device=q.device)
    ns = meta.n_split
    if ns > 1:
        part_acc = part_acc if part_acc is not None else torch.empty(t, h, ns, d, dtype=torch.float32, device=q.device)
        part_ml = part_ml if part_ml is not None else torch.empty(t, h, ns, 2, dtype=torch.float32, device=q.device)
    else:
        part_acc = part_ml = out
    _attn_kernel[(meta.n_tiles, kvh, ns)](
        q, q.stride(0), k_new, v_new, k_new.stride(0), k_cache, v_cache, k_cache.stride(0), k_cache.stride(1),
        rel, rel.stride(0), sinks, sinks.stride(0), gate, gate.stride(0), out, out.stride(0), part_acc, part_ml,
        meta.seq_slot, meta.seq_start, meta.seq_off, meta.seq_n, meta.tile_seq, meta.tile_q0, scale,
        H=h, KVH=kvh, D=d, R=r, WINDOW=window, RING=ring, TOK=meta.tok_per_tile, NSPLIT=ns,
    )
    if ns > 1:
        _combine_kernel[(t, h)](part_acc, part_ml, sinks, sinks.stride(0), gate, gate.stride(0), out, out.stride(0),
                                H=h, D=d, NSPLIT=ns, num_warps=4)
    return out


@triton.jit
def _store_kv_kernel(KN, VN, kn_stride_t, KCache, VCache, cache_stride_slot, cache_stride_pos,
                     SeqSlot, SeqStart, SeqOff, SeqN, RING: tl.constexpr, W: tl.constexpr,
                     KVD: tl.constexpr, PDL: tl.constexpr = PDL_ON):
    """One program per (seq, token-in-seq). Ring caches keep only the last RING tokens."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    sq = tl.program_id(0)
    i = tl.program_id(1)
    n = tl.load(SeqN + sq)
    if i >= n:
        return
    if RING > 0:
        if i < n - RING:
            return
    pos = tl.load(SeqStart + sq) + i
    slot = tl.load(SeqSlot + sq).to(tl.int64)
    trow = (tl.load(SeqOff + sq) + i).to(tl.int64)
    cpos = (pos % RING) if RING > 0 else pos
    c = tl.arange(0, KVD)
    dst = slot * cache_stride_slot + cpos.to(tl.int64) * cache_stride_pos + c
    tl.store(KCache + dst, tl.load(KN + trow * kn_stride_t + c).to(KCache.dtype.element_ty))
    tl.store(VCache + dst, tl.load(VN + trow * kn_stride_t + c).to(VCache.dtype.element_ty))


def store_kv(k_new, v_new, k_cache, v_cache, meta: AttnMeta, ring: int, max_new: int):
    kvd = k_new.shape[1] * k_new.shape[2]
    _store_kv_kernel[(meta.seq_slot.shape[0], max_new)](
        k_new, v_new, k_new.stride(0), k_cache, v_cache, k_cache.stride(0), k_cache.stride(1),
        meta.seq_slot, meta.seq_start, meta.seq_off, meta.seq_n, RING=ring, W=0, KVD=kvd, num_warps=4,
    )


_pdl_enable(_attn_kernel, _combine_kernel, _store_kv_kernel)
