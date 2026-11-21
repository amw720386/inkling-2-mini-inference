"""Static per-slot caches.

Global (full-attention) visits keep a contiguous KV buffer of `max_ctx` tokens per slot.
Local (sliding-window) visits keep a ring of `ring` tokens per slot, indexed pos % ring.
Every visit keeps the last K-1 inputs of its four short convolutions.
All memory is allocated once, so CUDA graphs can capture pointers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import InklingConfig


@dataclass
class CacheSpec:
    slots: int
    max_ctx: int
    ring: int  # >= sliding_window + max tokens appended per step (spec verify)
    kv_dtype: torch.dtype = torch.bfloat16

    def bytes(self, cfg: InklingConfig) -> int:
        per_tok = 2 * cfg.kv_dim * torch.tensor([], dtype=self.kv_dtype).element_size()
        kv = per_tok * self.slots * (cfg.num_global_visits * self.max_ctx + cfg.num_local_visits * self.ring)
        conv = 2 * self.slots * cfg.num_visits * (cfg.sconv_kernel - 1) * (2 * cfg.kv_dim + 2 * cfg.hidden_size)
        return kv + conv


class InklingCache:
    def __init__(self, cfg: InklingConfig, spec: CacheSpec, device="cuda"):
        if spec.ring < cfg.sliding_window:
            raise ValueError("ring must hold the full sliding window")
        self.cfg, self.spec = cfg, spec
        g, l, s = cfg.num_global_visits, cfg.num_local_visits, spec.slots
        kvh, d = cfg.num_kv_heads, cfg.head_dim
        dt = spec.kv_dtype
        self.global_k = torch.zeros(g, s, spec.max_ctx, kvh, d, dtype=dt, device=device)
        self.global_v = torch.zeros_like(self.global_k)
        self.local_k = torch.zeros(l, s, spec.ring, kvh, d, dtype=dt, device=device)
        self.local_v = torch.zeros_like(self.local_k)
        km1, v = cfg.sconv_kernel - 1, cfg.num_visits
        bf = torch.bfloat16
        self.k_conv = torch.zeros(v, s, km1, cfg.kv_dim, dtype=bf, device=device)
        self.v_conv = torch.zeros_like(self.k_conv)
        self.attn_conv = torch.zeros(v, s, km1, cfg.hidden_size, dtype=bf, device=device)
        self.mlp_conv = torch.zeros_like(self.attn_conv)
        self.seq_lens = torch.zeros(s, dtype=torch.int64)  # host-side bookkeeping
        self.cand = None  # speculative verify: candidate conv states, see enable_spec

    def enable_spec(self, max_verify: int, device="cuda"):
        """Buffers for the conv states after committing 1..max_verify rows of a verify call."""
        mk = lambda t: torch.zeros(t.shape[0], max_verify, *t.shape[2:], dtype=t.dtype, device=device)
        self.cand = {name: mk(getattr(self, name)) for name in ("k_conv", "v_conv", "attn_conv", "mlp_conv")}

    def commit_spec(self, slot: int, n_commit: int):
        """After a verify call of the sequence in `slot`, keep the conv states of its first n_commit rows."""
        for name, cand in self.cand.items():
            getattr(self, name)[:, slot].copy_(cand[:, n_commit - 1])

    def reset_slot(self, slot: int) -> None:
        for t in (self.k_conv, self.v_conv, self.attn_conv, self.mlp_conv):
            t[:, slot].zero_()
        self.seq_lens[slot] = 0
