"""Single-GPU generation engine: chunked prefill, CUDA-graph decode, optional DSpark speculation.

Speculative rounds (greedy): the drafter proposes 7 tokens after the anchor; the leading drafts whose
confidence clears `spec_threshold` decide the verify width m in {2, 4, 8} (a wider verify reads more
experts); one captured verify graph scores the anchor + m-1 drafts; the longest prefix matching the target
argmax is accepted. Rejected rows need no KV rollback (the sliding ring has slack and later reads of
positions >= the verify start come from the next chunk itself); conv states take the candidate of the
committed row count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .batch import Batch, SeqChunk
from .cache import CacheSpec, InklingCache
from .config import InklingConfig
from .fast import build_step_meta
from .fused_model import FusedDecodeGraph, FusedModel
from .weights import load_weights

VERIFY_WIDTHS = (2, 4, 8)


@dataclass
class GenStats:
    prompt_tokens: int = 0
    generated: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    rounds: int = 0
    accepted: int = 0
    widths: dict = field(default_factory=dict)

    @property
    def decode_tok_s(self):
        return self.generated / self.decode_s if self.decode_s else 0.0


class VerifyGraph:
    """Captured target forward of one sequence's anchor + (m - 1) drafts, all rows' logits and taps."""

    def __init__(self, model: FusedModel, cache: InklingCache, m: int, slot: int = 0, n_split_global: int = 16):
        self.model, self.cache, self.m, self.slot = model, cache, m, slot
        seqs = [SeqChunk(slot=slot, start=0, n=m, want_logits="all")]
        self.batch = Batch(seqs=seqs, input_ids=torch.zeros(m, dtype=torch.int64, device="cuda"),
                           positions=torch.arange(m, dtype=torch.int64, device="cuda"), offsets=[0],
                           logit_rows=torch.arange(m, device="cuda"))
        self.meta = build_step_meta(self.batch, 4, n_split_global)
        self.ws = model.workspace(m, m, 16)
        self.base_pos = torch.arange(m, dtype=torch.int64, device="cuda")
        snap = [(t, t[:, slot].clone()) for t in (cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._run()
        for t, v in snap:
            t[:, slot].copy_(v)

    def _run(self):
        return self.model.forward(self.batch, self.cache, self.ws, self.meta, verify=True)

    def run(self, tokens: torch.Tensor, P: int):
        self.batch.input_ids.copy_(tokens)
        torch.add(self.base_pos, P, out=self.batch.positions)
        self.meta.local.seq_start.fill_(P)
        self.graph.replay()
        return self.logits, self.ws.taps[:self.m]


class Engine:
    def __init__(self, model_dir: str, max_ctx: int = 32768, spec: bool = True, spec_threshold: float = 0.8,
                 prefill_chunk: int = 2048, graphs: bool = True):
        self.cfg = InklingConfig.from_pretrained(model_dir)
        self.w = load_weights(self.cfg, model_dir)
        ring = self.cfg.sliding_window + 64
        self.cache = InklingCache(self.cfg, CacheSpec(slots=1, max_ctx=max_ctx + 16, ring=ring))
        self.model = FusedModel(self.cfg, self.w)
        self.prefill_chunk = prefill_chunk
        self.pws = self.model.workspace(prefill_chunk, 1)
        self.max_ctx = max_ctx
        self.spec = spec
        self.spec_threshold = spec_threshold
        self.drafter = None
        if spec:
            from .dspark import Drafter
            self.model.write_taps = True
            self.cache.enable_spec(max(VERIFY_WIDTHS))
            self.drafter = Drafter(model_dir, self.w, max_ctx)
        self.decode_graph = FusedDecodeGraph(self.model, self.cache, 1, capture=graphs)
        self.verify = {m: VerifyGraph(self.model, self.cache, m) for m in VERIFY_WIDTHS} if spec and graphs else {}
        if spec and graphs:
            self.drafter.capture()
        self.gamma = 2.0
        self.gamma = 2.0
        self.costs = self._calibrate() if spec and graphs else {m: 1.0 + 0.1 * m for m in VERIFY_WIDTHS}
        self.verify_tokens = {m: torch.zeros(m, dtype=torch.int64, device="cuda") for m in VERIFY_WIDTHS}

    def _calibrate(self) -> dict[int, float]:
        """Measured ms of a round per verify width (draft + verify), for the width policy."""
        def t(fn, n=20):
            fn()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / n

        draft = t(lambda: self.drafter.graphs[min(self.drafter.graphs)].replay())
        snap = [x[:, 0].clone() for x in (self.cache.k_conv, self.cache.v_conv, self.cache.attn_conv, self.cache.mlp_conv)]
        costs = {m: draft + t(lambda g=g: g.graph.replay()) for m, g in self.verify.items()}
        for x, v in zip((self.cache.k_conv, self.cache.v_conv, self.cache.attn_conv, self.cache.mlp_conv), snap):
            x[:, 0].copy_(v)
        self.drafter.reset()
        return costs

    def _expected(self, conf: list[float], m: int) -> float:
        """Expected accepted drafts of a width-m verify: leading products of conf ** gamma."""
        exp, run = 0.0, 1.0
        for c in conf[:m - 1]:
            run *= c ** self.gamma
            exp += run
        return exp

    def _pick_width(self, conf: list[float]) -> int:
        """Width maximizing expected committed tokens per ms. The drafter's confidences overstate
        acceptance; gamma (confidence ** gamma) is tuned online by `_observe`."""
        best, best_rate = VERIFY_WIDTHS[0], -1.0
        for m in VERIFY_WIDTHS:
            rate = (1.0 + self._expected(conf, m)) / self.costs[m]
            if rate > best_rate:
                best, best_rate = m, rate
        return best

    def _observe(self, conf: list[float], m: int, accepted: int):
        err = accepted - self._expected(conf, m)
        self.gamma = min(8.0, max(0.5, self.gamma - 0.05 * err))

    # ------------------------------------------------------------------ prefill
    @torch.inference_mode()
    def prefill(self, ids: list[int]) -> torch.Tensor:
        self.cache.reset_slot(0)
        if self.drafter is not None:
            self.drafter.reset()
        logits = None
        for s in range(0, len(ids), self.prefill_chunk):
            part = ids[s:s + self.prefill_chunk]
            b = Batch.build([SeqChunk(0, s, len(part), "last")], [part])
            logits = self.model.forward(b, self.cache, self.pws)
            if self.drafter is not None:
                self.drafter.commit(self.pws.taps[:len(part)], s)
        return logits[-1]

    # ----------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, ids: list[int], max_new: int, stop_ids=(), spec: bool | None = None) -> tuple[list[int], GenStats]:
        st = GenStats(prompt_tokens=len(ids))
        stop = set(stop_ids)
        max_new = min(max_new, self.max_ctx - len(ids) - max(VERIFY_WIDTHS))
        torch.cuda.synchronize()
        t0 = time.time()
        logits = self.prefill(ids)
        tok = int(logits.argmax())
        torch.cuda.synchronize()
        st.prefill_s = time.time() - t0
        t0 = time.time()
        out = [tok]
        P = len(ids)  # position of the anchor (sampled, not yet processed)
        use_spec = self.spec if spec is None else (spec and self.spec)
        while len(out) < max_new and out[-1] not in stop:
            if not use_spec:
                lg = self.decode_graph.step([0], [P], [out[-1]])
                out.append(int(lg[0].argmax()))
                if self.drafter is not None:  # keep the drafter's context complete
                    self.drafter.commit(self.decode_graph.ws.taps[:1], P)
                P += 1
                continue
            drafts, conf = self.drafter.draft(out[-1], P)
            info = torch.cat([conf, drafts.float()]).tolist()  # one sync
            confs, drafted = info[:7], [int(d) for d in info[7:]]
            m = self._pick_width(confs)
            vt = self.verify_tokens[m]
            vt[0] = out[-1]
            vt[1:].copy_(drafts[:m - 1])
            logits, taps = self.verify[m].run(vt, P)
            target = logits.argmax(-1)
            acc_t = (drafts[:m - 1] == target[:m - 1]).int().cumprod(0).sum(0, keepdim=True)
            acc, *tgt = torch.cat([acc_t, target.int()]).tolist()  # one sync
            commit = acc + 1
            self._observe(confs, m, acc)
            self.cache.commit_spec(0, commit)
            self.drafter.commit(taps[:commit], P)
            new = drafted[:acc] + [tgt[acc]]
            for t in new:
                out.append(t)
                if t in stop or len(out) >= max_new:
                    break
            P += commit
            st.rounds += 1
            st.accepted += acc
            st.widths[m] = st.widths.get(m, 0) + 1
        torch.cuda.synchronize()
        st.decode_s = time.time() - t0
        st.generated = len(out)
        return out, st
