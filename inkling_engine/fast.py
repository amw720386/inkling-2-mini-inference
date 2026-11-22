"""Per-step attention / conv metadata in device tensors (CUDA-graph friendly): eager steps build it per batch,
captured graphs own static copies and only update their contents."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .batch import Batch
from .ops.attn_triton import AttnMeta


@dataclass
class StepMeta:
    local: AttnMeta
    glob: AttnMeta
    tok_seq: torch.Tensor  # [T] int32
    max_new: int  # max new tokens of any seq (store_kv grid)
    n_tokens: int


def build_step_meta(batch: Batch, n_split_local: int, n_split_global: int, device="cuda") -> StepMeta:
    max_new = max(s.n for s in batch.seqs)
    tok = 4 if max_new <= 4 else (8 if max_new <= 8 else (16 if max_new <= 256 else 32))  # one tile per verify step: KV read once
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
