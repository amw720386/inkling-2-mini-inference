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
from .sampling import Sampling, accept_sampled, probs, sample
from .weights import load_weights

VERIFY_WIDTHS = (2, 4, 8)


@dataclass
class GenStats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
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
                 prefill_chunk: int = 2048, graphs: bool = True, dense_fp8: bool = False, kv_fp8: bool = False):
        self.cfg = InklingConfig.from_pretrained(model_dir)
        self.w = load_weights(self.cfg, model_dir)
        ring = self.cfg.sliding_window + 64
        kv_dtype = torch.float8_e4m3fn if kv_fp8 else torch.bfloat16
        self.cache = InklingCache(self.cfg, CacheSpec(slots=1, max_ctx=max_ctx + 16, ring=ring, kv_dtype=kv_dtype))
        self.model = FusedModel(self.cfg, self.w, dense_fp8=dense_fp8)
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
        self.state_ids: list[int] = []  # tokens whose positions the cache holds
        self.snap = None  # state at the end of the last prompt (see _snapshot)
        # compile and autotune the prefill kernels (first and later chunks) now, not in the first long request
        warm = min(2 * prefill_chunk + 100, max_ctx - 16)
        self.prefill(list(range(1000, 1000 + warm)))
        self.cache.reset_slot(0)
        self.state_ids, self.snap = [], None
        if self.drafter is not None:
            self.drafter.reset()

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
    # ----------------------------------------------------------- prefix reuse
    def _snapshot(self, ids: list[int]):
        """State at the end of a prompt: conv states, sliding rings and the drafter's rings. Global KV is
        position-indexed, so positions < len(ids) stay valid without a copy."""
        c = self.cache
        snap = {"ids": list(ids),
                "conv": [t[:, 0].clone() for t in (c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv)],
                "ring": (c.local_k[:, 0].clone(), c.local_v[:, 0].clone())}
        if self.drafter is not None:
            d = self.drafter
            snap["draft"] = ({l: (d.k_ctx[l].clone(), d.v_ctx[l].clone()) for l in (0, 2)},
                             {l: d.ctx_pos[l].clone() for l in (0, 2)}, d.f_rnn.clone(), d.length)
        self.snap = snap

    def _restore(self):
        c, snap = self.cache, self.snap
        for t, v in zip((c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv), snap["conv"]):
            t[:, 0].copy_(v)
        c.local_k[:, 0].copy_(snap["ring"][0])
        c.local_v[:, 0].copy_(snap["ring"][1])
        if self.drafter is not None:
            kv, pos, f, length = snap["draft"]
            d = self.drafter
            for l in (0, 2):
                d.k_ctx[l].copy_(kv[l][0])
                d.v_ctx[l].copy_(kv[l][1])
                d.ctx_pos[l].copy_(pos[l])
            d.f_rnn.copy_(f)
            d.length = length

    @torch.inference_mode()
    def prefill(self, ids: list[int]) -> tuple[torch.Tensor, int]:
        """Bring the cache to `ids`. Reuses (a) everything processed so far when `ids` extends it (a
        continuation), or (b) the previous prompt's snapshot when `ids` extends that prompt (a chat turn whose
        template re-renders the earlier answer). Returns (last logits, reused tokens)."""
        prev = self.state_ids
        start = 0
        drafter_ok = lambda n: self.drafter is None or self.drafter.length == n  # noqa: E731
        if 0 < len(prev) < len(ids) and ids[:len(prev)] == prev and drafter_ok(len(prev)):
            start = len(prev)
        elif self.snap is not None and 0 < len(self.snap["ids"]) < len(ids) and ids[:len(self.snap["ids"])] == self.snap["ids"]:
            self._restore()
            start = len(self.snap["ids"])
        if start == 0:
            self.cache.reset_slot(0)
            if self.drafter is not None:
                self.drafter.reset()
        logits = None
        for s in range(start, len(ids), self.prefill_chunk):
            part = ids[s:s + self.prefill_chunk]
            b = Batch.build([SeqChunk(0, s, len(part), "last")], [part])
            logits = self.model.forward(b, self.cache, self.pws)
            if self.drafter is not None:
                self.drafter.commit(self.pws.taps[:len(part)], s)
        self.state_ids = list(ids)
        self._snapshot(ids)
        return logits[-1], start

    # ----------------------------------------------------------------- generate
    _probs = staticmethod(probs)

    def _sample(self, logits, sp: Sampling, gen) -> int:
        return sample(logits, sp, gen)

    @torch.inference_mode()
    def stream(self, ids: list[int], max_new: int, stop_ids=(), sampling: "Sampling | None" = None,
               spec: bool | None = None, stats: "GenStats | None" = None):
        """Yield generated token ids, with DSpark speculation when enabled (greedy: longest matching prefix;
        sampled: rejection sampling, exact for the target distribution)."""
        sp = sampling or Sampling()
        st = stats if stats is not None else GenStats()
        st.prompt_tokens = len(ids)
        stop = set(stop_ids)
        max_new = min(max_new, self.max_ctx - len(ids) - max(VERIFY_WIDTHS))
        if max_new <= 0:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit max_ctx {self.max_ctx}")
        gen = None
        if sp.seed is not None:
            gen = torch.Generator(device="cuda")
            gen.manual_seed(sp.seed)
            if self.drafter is not None:  # the drafter's Gumbel noise too, so a seeded request reproduces
                self.drafter.seed.fill_(sp.seed & 0x7FFFFFFF)
        torch.cuda.synchronize()
        t0 = time.time()
        logits, st.reused_tokens = self.prefill(ids)
        tok = self._sample(logits, sp, gen)
        torch.cuda.synchronize()
        st.prefill_s = time.time() - t0
        t0 = time.time()
        out = [tok]
        P = len(ids)  # position of the anchor (sampled, not yet processed)
        use_spec = self.spec if spec is None else (spec and self.spec)
        try:
            yield tok
            while len(out) < max_new and out[-1] not in stop:
                if not use_spec:
                    lg = self.decode_graph.step([0], [P], [out[-1]])
                    if self.drafter is not None:  # keep the drafter's context complete
                        self.drafter.commit(self.decode_graph.ws.taps[:1], P)
                    P += 1
                    t = self._sample(lg[0], sp, gen)
                    out.append(t)
                    yield t
                    continue
                drafts, conf = self.drafter.draft(out[-1], P, max(sp.temperature, 0.0))
                info = torch.cat([conf, drafts.float()]).tolist()  # one sync
                confs, drafted = info[:7], [int(d) for d in info[7:]]
                m = self._pick_width(confs)
                vt = self.verify_tokens[m]
                vt[0] = out[-1]
                vt[1:].copy_(drafts[:m - 1])
                logits, taps = self.verify[m].run(vt, P)
                if sp.temperature <= 0:
                    target = logits.argmax(-1)
                    acc_t = (drafts[:m - 1] == target[:m - 1]).int().cumprod(0).sum(0, keepdim=True)
                    acc, *tgt = torch.cat([acc_t, target.int()]).tolist()  # one sync
                else:
                    acc, nxt = self._accept_sampled(drafts[:m - 1], logits, sp, gen)
                    tgt = [0] * acc + [nxt]
                commit = acc + 1
                self._observe(confs, m, acc)
                self.cache.commit_spec(0, commit)
                self.drafter.commit(taps[:commit], P)
                P += commit
                st.rounds += 1
                st.accepted += acc
                st.widths[m] = st.widths.get(m, 0) + 1
                for t in drafted[:acc] + [tgt[acc]]:
                    out.append(t)
                    yield t
                    if t in stop or len(out) >= max_new:
                        break
        finally:
            # the cache holds positions [0, P): those tokens can be reused by the next request
            self.state_ids = (list(ids) + out)[:P]
            torch.cuda.synchronize()
            st.decode_s = time.time() - t0
            st.generated = len(out)

    def _accept_sampled(self, drafts, target_logits, sp, gen):
        cand, q = self.drafter.draft_distribution(drafts.shape[0], sp.temperature)
        return accept_sampled(drafts, target_logits, cand, q, sp, gen)

    def generate(self, ids: list[int], max_new: int, stop_ids=(), spec: bool | None = None,
                 sampling: "Sampling | None" = None) -> tuple[list[int], GenStats]:
        st = GenStats()
        out = list(self.stream(ids, max_new, stop_ids, sampling, spec, st))
        return out, st
