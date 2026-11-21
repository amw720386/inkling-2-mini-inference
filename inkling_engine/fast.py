"""Fast backend: Triton kernels for the stateful and heavy ops, CUDA-graph friendly.

Per-step metadata lives in device tensors. `begin()` builds them for eager steps;
the decode graph runner owns static copies and only updates their contents.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .batch import Batch
from .cache import InklingCache
from .config import InklingConfig, Visit
from .model import RefBackend
from .ops import ref
from .ops.attn_triton import AttnMeta, inkling_attention, store_kv
from .ops.conv_triton import short_conv
from .ops.moe_triton import moe_decode, moe_prefill

DECODE_MOE_MAX_TOKENS = 16


@dataclass
class StepMeta:
    local: AttnMeta
    glob: AttnMeta
    tok_seq: torch.Tensor  # [T] int32
    max_new: int  # max new tokens of any seq (store_kv grid)
    n_tokens: int


def build_step_meta(batch: Batch, n_split_local: int, n_split_global: int, device="cuda") -> StepMeta:
    max_new = max(s.n for s in batch.seqs)
    tok = 4 if max_new <= 4 else (8 if max_new <= 8 else 16)  # one tile per verify step: KV read once
    loc = AttnMeta.build(batch.seqs, batch.offsets, tok, n_split_local, device)
    glob = AttnMeta(loc.seq_slot, loc.seq_start, loc.seq_off, loc.seq_n, loc.tile_seq, loc.tile_q0, loc.n_tiles,
                    tok, n_split_global, loc.max_keys)
    tok_seq = torch.repeat_interleave(torch.arange(len(batch.seqs), dtype=torch.int32),
                                      torch.tensor([s.n for s in batch.seqs])).to(device)
    return StepMeta(loc, glob, tok_seq, max_new, batch.num_tokens)


def pick_splits(max_keys: int, n_tiles: int, kvh: int, sms: int, window: int) -> tuple[int, int]:
    """Enough programs to fill the GPU at decode; one split for big prefill tiles."""
    def splits(keys):
        blocks = max(1, (keys + 63) // 64)
        want = max(1, (2 * sms) // max(1, n_tiles * kvh))
        s = 1
        while s * 2 <= min(want, blocks, 32):
            s *= 2
        return s

    return splits(min(max_keys, window)), splits(max_keys)


class FastBackend(RefBackend):
    def __init__(self, cfg: InklingConfig):
        super().__init__(cfg)
        self.meta: StepMeta | None = None
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.static_meta = False  # set by the graph runner

    def begin(self, batch: Batch):
        if self.static_meta:
            return
        max_new = max(s.n for s in batch.seqs)
        tok = 4 if max_new <= 4 else (8 if max_new <= 8 else 16)  # one tile per verify step: KV read once
        n_tiles = sum((s.n + tok - 1) // tok for s in batch.seqs)
        max_keys = max(s.start + s.n for s in batch.seqs)
        nl, ng = pick_splits(max_keys, n_tiles, self.cfg.num_kv_heads, self.sms, self.cfg.sliding_window)
        self.meta = build_step_meta(batch, nl, ng)

    # ------------------------------------------------------------------ stateful
    def conv(self, state, x, weight, batch, out_scale):
        m = self.meta
        return short_conv(x, state, m.local, m.tok_seq, weight=weight, scale=out_scale)

    def dyn_conv(self, state, x, static_w, dyn_w, batch, out_scale):
        m = self.meta
        return short_conv(x, state, m.local, m.tok_seq, static=static_w, dyn=dyn_w,
                          ch_per_head=self.cfg.head_dim, scale=out_scale)

    def attention(self, v: Visit, q, k, vv, rel_logits, sinks, batch, cache: InklingCache, gate=None):
        c, m = self.cfg, self.meta
        if v.is_local:
            kc, vc = cache.local_k[v.local_index], cache.local_v[v.local_index]
            meta, window, ring = m.local, c.sliding_window, cache.spec.ring
        else:
            kc, vc = cache.global_k[v.global_index], cache.global_v[v.global_index]
            meta, window, ring = m.glob, 0, 0
        out = inkling_attention(q, k, vv, kc, vc, rel_logits, sinks, gate, meta, window=window, ring=ring,
                                scale=c.head_dim ** -0.5)
        store_kv(k, vv, kc, vc, meta, ring, m.max_new)
        return out

    def moe(self, x, lw):
        c = self.cfg
        wts, idx = ref.route(x, lw.router_w, lw.router_bias, c.top_k, c.route_scale)
        if x.shape[0] <= DECODE_MOE_MAX_TOKENS:
            rows = moe_decode(x, idx, wts, lw.w13, lw.w2, c.mlp_gate_softcap, c.mlp_up_softcap)
            return rows.view(x.shape[0], c.top_k, -1).sum(1).to(torch.bfloat16)
        return moe_prefill(x, idx, wts, lw.w13, lw.w2, c.mlp_gate_softcap, c.mlp_up_softcap)

    fused_gate = True  # attention() applies sigmoid(gate)
