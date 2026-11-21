"""Triton attention vs the torch reference (random data; cache prefix + chunk)."""

import pytest
import torch

from inkling_engine.batch import SeqChunk
from inkling_engine.ops import ref
from inkling_engine.ops.attn_triton import AttnMeta, inkling_attention, store_kv

H, KVH, D, R, W = 24, 8, 128, 1024, 1024


def make_case(seqs, cap, ring, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=g)
    t = sum(s.n for s in seqs)
    slots = max(s.slot for s in seqs) + 1
    q = ref.l2_clip(rnd(t, H, D) * 3, 22.6)
    kn = ref.l2_clip(rnd(t, KVH, D) * 3, 22.6)
    vn = rnd(t, KVH, D).to(torch.bfloat16)
    rel = (rnd(t, H, R) * 2).to(torch.bfloat16)
    qgrs = (rnd(t, H * D + H)).to(torch.bfloat16)  # gate + sinks live in one row like the model
    gate, sinks = qgrs[:, : H * D], qgrs[:, H * D:]
    max_pos = max(s.start + s.n for s in seqs)
    # full per-slot history (positions 0..max_pos) for the reference
    hist_k = ref.l2_clip(rnd(slots, max_pos, KVH, D) * 3, 22.6)
    hist_v = rnd(slots, max_pos, KVH, D).to(torch.bfloat16)
    off = 0
    offsets = []
    for s in seqs:
        offsets.append(off)
        hist_k[s.slot, s.start:s.start + s.n] = kn[off:off + s.n]
        hist_v[s.slot, s.start:s.start + s.n] = vn[off:off + s.n]
        off += s.n
    capn = ring if ring else cap
    kc = torch.zeros(slots, capn, KVH, D, dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    for s in seqs:
        for p in range(s.start):
            idx = p % ring if ring else p
            kc[s.slot, idx] = hist_k[s.slot, p]
            vc[s.slot, idx] = hist_v[s.slot, p]
    return q, kn, vn, rel, gate, sinks, kc, vc, hist_k, hist_v, offsets


def reference(seqs, offsets, q, rel, gate, sinks, hist_k, hist_v, window):
    outs = []
    for s, off in zip(seqs, offsets):
        sl = slice(off, off + s.n)
        qp = torch.arange(s.start, s.start + s.n, device="cuda")
        kp = torch.arange(0, s.start + s.n, device="cuda")
        o = ref.attention(q[sl], hist_k[s.slot, :s.start + s.n], hist_v[s.slot, :s.start + s.n], qp, kp,
                          rel[sl], sinks[sl], window, D ** -0.5)
        outs.append((o.float() * torch.sigmoid(gate[sl].float())).to(torch.bfloat16))
    return torch.cat(outs)


CASES = {
    "prefill_fresh": [SeqChunk(0, 0, 300)],
    "prefill_chunk_after_prefix": [SeqChunk(0, 1500, 200)],
    "decode_batch": [SeqChunk(0, 2000, 1), SeqChunk(1, 37, 1), SeqChunk(2, 1100, 1)],
    "verify8": [SeqChunk(0, 3000, 8), SeqChunk(1, 500, 8)],
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("n_split", [1, 4])
def test_attention(case, local, n_split):
    seqs = CASES[case]
    ring = W + 64 if local else 0
    q, kn, vn, rel, gate, sinks, kc, vc, hk, hv, offs = make_case(seqs, cap=4096, ring=ring)
    tok = 16 if max(s.n for s in seqs) > 8 else 4
    meta = AttnMeta.build(seqs, offs, tok_per_tile=tok, n_split=n_split)
    out = inkling_attention(q, kn, vn, kc, vc, rel, sinks, gate, meta, window=W if local else 0, ring=ring,
                            scale=D ** -0.5)
    exp = reference(seqs, offs, q, rel, gate, sinks, hk, hv, W if local else None)
    err = ((out.float() - exp.float()).norm() / exp.float().norm()).item()
    print(case, local, n_split, "rel err", err)
    assert err < 1e-2

    # cache write: every new position must land where later steps read it
    store_kv(kn, vn, kc, vc, meta, ring, max(s.n for s in seqs))
    for s, off in zip(seqs, offs):
        p = s.start + s.n - 1
        idx = p % ring if ring else p
        assert torch.equal(kc[s.slot, idx], kn[off + s.n - 1])
