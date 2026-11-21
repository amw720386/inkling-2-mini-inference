"""CUDA-graph decode runner: one captured graph per decode batch size.

Each graph runs `bs` sequences, one new token each, in any slots. Per step only the
contents of static tensors change (token ids, positions, slots).
"""

from __future__ import annotations

import torch

from .batch import Batch, SeqChunk
from .fast import FastBackend, build_step_meta
from .model import InklingModel


class DecodeGraph:
    def __init__(self, model: InklingModel, cache, bs: int, n_split_local: int = 4, n_split_global: int = 16,
                 capture: bool = True, pool=None):
        assert isinstance(model.be, FastBackend)
        self.model, self.cache, self.bs = model, cache, bs
        dev = "cuda"
        seqs = [SeqChunk(slot=i, start=0, n=1) for i in range(bs)]
        self.batch = Batch(seqs=seqs, input_ids=torch.zeros(bs, dtype=torch.int64, device=dev),
                           positions=torch.zeros(bs, dtype=torch.int64, device=dev), offsets=list(range(bs)),
                           logit_rows=torch.arange(bs, device=dev))
        self.meta = build_step_meta(self.batch, n_split_local, n_split_global, dev)
        self.graph = None
        self.logits = None
        if capture:
            self._capture(pool)

    def _run(self):
        be = self.model.be
        prev_meta, prev_static = be.meta, be.static_meta
        be.meta, be.static_meta = self.meta, True
        try:
            logits, _ = self.model.forward(self.batch, self.cache)
        finally:
            be.meta, be.static_meta = prev_meta, prev_static
        return logits

    def _capture(self, pool):
        # warm up (compiles kernels) on scratch values, then restore the touched state
        snap = self._snapshot()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.logits = self._run()
        self._restore(snap)

    def _snapshot(self):
        c = self.cache
        sl = slice(0, self.bs)
        return [(t, t[:, sl].clone()) for t in (c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv)]

    def _restore(self, snap):
        sl = slice(0, self.bs)
        for t, v in snap:
            t[:, sl].copy_(v)

    def set_inputs(self, slots: list[int], positions: list[int], tokens: list[int]):
        assert len(slots) == self.bs
        dev = "cuda"
        self.batch.input_ids.copy_(torch.tensor(tokens, dtype=torch.int64), non_blocking=True)
        self.batch.positions.copy_(torch.tensor(positions, dtype=torch.int64), non_blocking=True)
        self.meta.local.seq_slot.copy_(torch.tensor(slots, dtype=torch.int32), non_blocking=True)
        self.meta.local.seq_start.copy_(torch.tensor(positions, dtype=torch.int32), non_blocking=True)
        del dev

    def step(self, slots, positions, tokens) -> torch.Tensor:
        self.set_inputs(slots, positions, tokens)
        if self.graph is None:
            return self._run()
        self.graph.replay()
        return self.logits
