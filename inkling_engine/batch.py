"""Host-side description of one forward step (prefill chunks and/or decode tokens)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SeqChunk:
    slot: int
    start: int  # position of the first new token
    n: int  # number of new tokens
    want_logits: str = "last"  # "last" | "all" | "none"


@dataclass
class Batch:
    seqs: list[SeqChunk]
    input_ids: torch.Tensor  # [T] int64 on device
    positions: torch.Tensor  # [T] int64 on device
    offsets: list[int] = field(default_factory=list)  # token offset of each seq
    logit_rows: torch.Tensor | None = None  # [n_out] indices into T

    @classmethod
    def build(cls, seqs: list[SeqChunk], token_ids: list[list[int]], device="cuda") -> "Batch":
        assert len(seqs) == len(token_ids)
        ids, pos, offs, rows = [], [], [], []
        off = 0
        for s, toks in zip(seqs, token_ids):
            assert len(toks) == s.n
            offs.append(off)
            ids += toks
            pos += range(s.start, s.start + s.n)
            if s.want_logits == "last":
                rows.append(off + s.n - 1)
            elif s.want_logits == "all":
                rows += range(off, off + s.n)
            off += s.n
        return cls(
            seqs=seqs,
            input_ids=torch.tensor(ids, dtype=torch.int64, device=device),
            positions=torch.tensor(pos, dtype=torch.int64, device=device),
            offsets=offs,
            logit_rows=torch.tensor(rows, dtype=torch.int64, device=device),
        )

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    def seq_slice(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i] + self.seqs[i].n)
