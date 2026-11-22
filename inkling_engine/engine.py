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
GAMMA = 7  # drafted tokens per DSpark round
SEEDED_GAMMA = 2.0  # fixed confidence calibration for seeded requests (reproducible width choices)
IMAGE_TOKEN_ID = 200054


def _drain(gen):
    """Run a generator to completion and return its return value."""
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


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
    """Captured target forward of k sequences, each its anchor + (m - 1) drafts, in any slots: all rows'
    logits and taps. Conv-state candidates go to cache.cand at batch positions 0..k-1."""

    def __init__(self, model: FusedModel, cache: InklingCache, k: int, m: int, n_split_global: int = 16, pool=None):
        self.model, self.cache, self.k, self.m = model, cache, k, m
        seqs = [SeqChunk(slot=i, start=0, n=m, want_logits="all") for i in range(k)]
        t = k * m
        self.batch = Batch(seqs=seqs, input_ids=torch.zeros(t, dtype=torch.int64, device="cuda"),
                           positions=torch.arange(t, dtype=torch.int64, device="cuda") % m,
                           offsets=[i * m for i in range(k)], logit_rows=torch.arange(t, device="cuda"))
        self.meta = build_step_meta(self.batch, 4, n_split_global)
        self.ws = model.workspace(t, t, 16)
        self.base_pos = torch.arange(m, dtype=torch.int64)
        sl = slice(0, k)
        snap = [(x, x[:, sl].clone()) for x in (cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.logits = self._run()
        for x, v in snap:
            x[:, sl].copy_(v)

    def _run(self):
        return self.model.forward(self.batch, self.cache, self.ws, self.meta, verify=True)

    def run(self, tokens: torch.Tensor, slots: list[int], positions: list[int]):
        """tokens [k * m] (sequence b at rows b*m..); slots / positions (of each anchor) per sequence."""
        self.batch.input_ids.copy_(tokens)
        pos = (torch.tensor(positions, dtype=torch.int64)[:, None] + self.base_pos[None]).reshape(-1)
        self.batch.positions.copy_(pos, non_blocking=True)
        self.meta.local.seq_slot.copy_(torch.tensor(slots, dtype=torch.int32), non_blocking=True)
        self.meta.local.seq_start.copy_(torch.tensor(positions, dtype=torch.int32), non_blocking=True)
        self.graph.replay()
        return self.logits, self.ws.taps[:self.k * self.m]


class Engine:
    def __init__(self, model_dir: str, max_ctx: int = 32768, spec: bool = True, spec_threshold: float = 0.8,
                 prefill_chunk: int = 2048, graphs: bool = True, dense_fp8: bool = False, kv_fp8: bool = False,
                 slots: int = 1, vision: bool = False):
        """slots: concurrent sequences (cache slots and batched decode graphs 1..slots). Speculation and prefix
        reuse apply to slot 0, which the scheduler gives to a request running alone."""
        self.slots = slots
        self.vision = None
        if vision:
            from .vision import VisionEncoder
            self.vision = VisionEncoder(model_dir)
        self.cfg = InklingConfig.from_pretrained(model_dir)
        self.w = load_weights(self.cfg, model_dir)
        ring = self.cfg.sliding_window + 64
        kv_dtype = torch.float8_e4m3fn if kv_fp8 else torch.bfloat16
        self.cache = InklingCache(self.cfg, CacheSpec(slots=slots, max_ctx=max_ctx + 16, ring=ring, kv_dtype=kv_dtype))
        self.model = FusedModel(self.cfg, self.w, dense_fp8=dense_fp8)
        self.model.compact()
        self.prefill_chunk = prefill_chunk
        self.pws = self.model.workspace(prefill_chunk, 1)
        self.max_ctx = max_ctx
        self.spec = spec
        self.spec_threshold = spec_threshold
        self.drafter = None
        if spec:
            from .dspark import Drafter
            self.model.write_taps = True
            self.cache.enable_spec(max(VERIFY_WIDTHS), slots)
            self.drafter = Drafter(model_dir, self.w, max_ctx, slots=slots)
        # every graph shares one memory pool: they never run concurrently and keep outputs in persistent buffers
        self.pool = torch.cuda.graph_pool_handle()
        self.decode_graphs = {bs: FusedDecodeGraph(self.model, self.cache, bs, capture=graphs, pool=self.pool)
                              for bs in range(1, slots + 1)}
        self.decode_graph = self.decode_graphs[1]
        self.verify = ({(k, m): VerifyGraph(self.model, self.cache, k, m, pool=self.pool) for k in range(1, slots + 1)
                        for m in VERIFY_WIDTHS} if spec and graphs else {})
        if spec and graphs:
            self.drafter.capture(self.pool)
        self.gamma = 2.0
        self.draft_cost, self.verify_costs, self.decode_costs = 1.0, {}, {}
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

        draft = t(lambda: self.drafter.graphs[(self.drafter.buckets[0], 0)].replay())
        convs = (self.cache.k_conv, self.cache.v_conv, self.cache.attn_conv, self.cache.mlp_conv)
        snap = [x.clone() for x in convs]
        costs = {m: draft + t(lambda g=self.verify[(1, m)]: g.graph.replay()) for m in VERIFY_WIDTHS}
        # for the scheduler's choice between batched speculation and batched plain decode
        self.draft_cost = draft
        self.verify_costs = {km: t(lambda g=g: g.graph.replay(), 10) for km, g in self.verify.items()}
        self.decode_costs = {k: t(lambda g=g: g.graph.replay(), 10) for k, g in self.decode_graphs.items()}
        for x, v in zip(convs, snap):
            x.copy_(v)
        self.drafter.reset()
        return costs

    def batched_spec_rate(self, accept_estimates: list[float]) -> tuple[float, float]:
        """(expected tokens/ms of a batched DSpark round, tokens/ms of a batched plain step) for requests
        with these expected accepted drafts per round."""
        k = len(accept_estimates)
        best = 0.0
        for m in VERIFY_WIDTHS:
            tokens = sum(1.0 + min(a, m - 1) for a in accept_estimates)
            best = max(best, tokens / (k * self.draft_cost + self.verify_costs[(k, m)]))
        return best, k / self.decode_costs[k]

    def _expected(self, conf: list[float], m: int, gamma: float | None = None) -> float:
        """Expected accepted drafts of a width-m verify: leading products of conf ** gamma."""
        g = self.gamma if gamma is None else gamma
        exp, run = 0.0, 1.0
        for c in conf[:m - 1]:
            run *= c ** g
            exp += run
        return exp

    def _pick_width(self, conf: list[float], gamma: float | None = None) -> int:
        """Width maximizing expected committed tokens per ms. The drafter's confidences overstate
        acceptance; gamma (confidence ** gamma) is tuned online by `_observe`."""
        best, best_rate = VERIFY_WIDTHS[0], -1.0
        for m in VERIFY_WIDTHS:
            rate = (1.0 + self._expected(conf, m, gamma)) / self.costs[m]
            if rate > best_rate:
                best, best_rate = m, rate
        return best

    def _observe(self, conf: list[float], m: int, accepted: int):
        err = accepted - self._expected(conf, m)
        self.gamma = min(8.0, max(0.5, self.gamma - 0.05 * err))

    # ------------------------------------------------------------------ prefill
    def prefill_slot(self, slot: int, ids: list[int], image_feats=None) -> torch.Tensor:
        """Fresh prefill of `ids` into a non-zero slot (no prefix reuse). Returns last logits."""
        assert slot != 0
        return _drain(self.prefill_steps(slot, ids, image_feats))[0]

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
            d._use(0)
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
            d._use(0)
            for l in (0, 2):
                d.k_ctx[l].copy_(kv[l][0])
                d.v_ctx[l].copy_(kv[l][1])
                d.ctx_pos[l].copy_(pos[l])
            d.f_rnn.copy_(f)
            d.length = length

    def prefill(self, ids: list[int], image_feats=None) -> tuple[torch.Tensor, int]:
        """Bring slot 0 to `ids`; see prefill_steps. Returns (last logits, reused tokens)."""
        return _drain(self.prefill_steps(0, ids, image_feats))

    @torch.inference_mode()
    def prefill_steps(self, slot: int, ids: list[int], image_feats=None):
        """Generator: prefill `ids` into `slot` one chunk per step (yields between chunks so a scheduler can
        interleave decode steps); returns (last logits, reused tokens). Slot 0 reuses (a) everything processed
        so far when `ids` extends it, or (b) the previous prompt's snapshot when `ids` extends that prompt (a
        chat turn whose template re-renders the earlier answer); other slots start fresh."""
        start = 0
        img_pos = [i for i, t in enumerate(ids) if t == IMAGE_TOKEN_ID]
        if img_pos and (image_feats is None or image_feats.shape[0] != len(img_pos)):
            raise ValueError(f"{len(img_pos)} image placeholder tokens but "
                             f"{0 if image_feats is None else image_feats.shape[0]} image features")
        if slot == 0 and not img_pos:  # placeholder ids do not identify the image: never reuse those
            prev = self.state_ids
            drafter_ok = lambda n: self.drafter is None or self.drafter.length == n  # noqa: E731
            if 0 < len(prev) < len(ids) and ids[:len(prev)] == prev and drafter_ok(len(prev)):
                start = len(prev)
            elif self.snap is not None and 0 < len(self.snap["ids"]) < len(ids) and ids[:len(self.snap["ids"])] == self.snap["ids"]:
                self._restore()
                start = len(self.snap["ids"])
        if start == 0:
            self.cache.reset_slot(slot)
            if self.drafter is not None:
                self.drafter.reset(slot)
        logits = None
        img = torch.tensor(img_pos, dtype=torch.long) if img_pos else None
        for s in range(start, len(ids), self.prefill_chunk):
            part = ids[s:s + self.prefill_chunk]
            b = Batch.build([SeqChunk(slot, s, len(part), "last")], [part])
            ef = None
            if img is not None:
                sel = ((img >= s) & (img < s + len(part))).nonzero()[:, 0]
                if len(sel):
                    ef = ((img[sel] - s).to(self.pws.blk.device), image_feats[sel.to(image_feats.device)])
            logits = self.model.forward(b, self.cache, self.pws, embed_feats=ef)
            if self.drafter is not None:
                self.drafter.commit(self.pws.taps[:len(part)], s, slot=slot)
            if s + self.prefill_chunk < len(ids):
                yield
        if slot == 0:
            self.state_ids = list(ids)
            self._snapshot(ids)
        return logits[-1].clone(), start

    # ----------------------------------------------------------------- generate
    _probs = staticmethod(probs)

    def _sample(self, logits, sp: Sampling, gen) -> int:
        return sample(logits, sp, gen)

    @torch.inference_mode()
    def stream(self, ids: list[int], max_new: int, stop_ids=(), sampling: "Sampling | None" = None,
               spec: bool | None = None, stats: "GenStats | None" = None, image_feats=None):
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
        logits, st.reused_tokens = self.prefill(ids, image_feats)
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
                new, commit = self.spec_round(out[-1], P, sp, gen, st)
                P += commit
                for t in new:
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

    def spec_round(self, anchor: int, P: int, sp: Sampling, gen, st: "GenStats | None" = None,
                   slot: int = 0) -> tuple[list[int], int]:
        """One DSpark round for the sequence in `slot` whose anchor (sampled, not yet processed) is at P.
        Returns (new tokens: accepted drafts + the next token, rows committed = positions advanced)."""
        drafts, conf = self.drafter.draft(anchor, P, max(sp.temperature, 0.0), slot=slot)
        info = torch.cat([conf, drafts.float()]).tolist()  # one sync
        confs, drafted = info[:7], [int(d) for d in info[7:]]
        seeded = sp.seed is not None
        m = self._pick_width(confs, SEEDED_GAMMA if seeded else None)
        vt = self.verify_tokens[m]
        vt[0] = anchor
        vt[1:].copy_(drafts[:m - 1])
        logits, taps = self.verify[(1, m)].run(vt, [slot], [P])
        if sp.temperature <= 0:
            target = logits.argmax(-1)
            acc_t = (drafts[:m - 1] == target[:m - 1]).int().cumprod(0).sum(0, keepdim=True)
            acc, *tgt = torch.cat([acc_t, target.int()]).tolist()  # one sync
            nxt = tgt[acc]
        else:
            acc, nxt = self._accept_sampled(drafts[:m - 1], logits, sp, gen)
        commit = acc + 1
        if not seeded:
            self._observe(confs, m, acc)
        self.cache.commit_spec(slot, commit, 0)
        self.drafter.commit(taps[:commit], P, slot=slot)
        if st is not None:
            st.rounds += 1
            st.accepted += acc
            st.widths[m] = st.widths.get(m, 0) + 1
        return drafted[:acc] + [nxt], commit

    def batched_spec_round(self, reqs: list) -> list[tuple[list[int], int]]:
        """One DSpark round for several sequences at once: each drafts in its slot, one verify graph checks
        them all (common width), and each accepts independently. reqs: objects with .slot, .P, .out (last =
        anchor), .sp, .gen, .stats. Returns per request (new tokens, rows committed)."""
        d = self.drafter
        drafts = []
        multi = d.draft_multi([r.out[-1] for r in reqs], [r.P for r in reqs],
                              [max(r.sp.temperature, 0.0) for r in reqs], [r.slot for r in reqs])
        for r, (tok, conf, cand, lg_c) in zip(reqs, multi):
            q = torch.softmax(lg_c / r.sp.temperature, -1) if r.sp.temperature > 0 else None
            drafts.append((tok, conf, cand if q is not None else None, q))
        info = torch.cat([torch.cat([c, t.float()]) for t, c, _, _ in drafts]).tolist()  # one sync
        k = len(reqs)
        confs = [info[i * 14:i * 14 + 7] for i in range(k)]
        drafted = [[int(x) for x in info[i * 14 + 7:i * 14 + 14]] for i in range(k)]
        m = max(self._pick_width(c, SEEDED_GAMMA if r.sp.seed is not None else None) for c, r in zip(confs, reqs))
        vt = torch.empty(k * m, dtype=torch.int64, device="cuda")
        for i, r in enumerate(reqs):
            vt[i * m] = r.out[-1]
            vt[i * m + 1:(i + 1) * m].copy_(drafts[i][0][:m - 1])
        logits, taps = self.verify[(k, m)].run(vt, [r.slot for r in reqs], [r.P for r in reqs])
        lg = logits.view(k, m, -1)
        greedy = [i for i, r in enumerate(reqs) if r.sp.temperature <= 0]
        res = {}
        if greedy:
            gi = torch.tensor(greedy, device=lg.device)
            target = lg[gi].argmax(-1)  # [g, m]
            dr = torch.stack([drafts[i][0][:m - 1] for i in greedy])
            acc = (dr == target[:, :m - 1]).int().cumprod(1).sum(1)
            flat = torch.cat([acc[:, None], target.int()], 1).tolist()  # one sync
            for j, i in enumerate(greedy):
                a = flat[j][0]
                res[i] = (a, flat[j][1 + a])
        for i, r in enumerate(reqs):
            if i not in res:
                _, _, cand, q = drafts[i]
                res[i] = accept_sampled(drafts[i][0][:m - 1], lg[i], cand[:m - 1], q[:m - 1], r.sp, r.gen)
        out = []
        for i, r in enumerate(reqs):
            a, nxt = res[i]
            commit = a + 1
            if r.sp.seed is None:
                self._observe(confs[i], m, a)
            self.cache.commit_spec(r.slot, commit, i)
            d.commit(taps[i * m:i * m + commit], r.P, slot=r.slot)
            r.stats.rounds += 1
            r.stats.accepted += a
            r.stats.widths[m] = r.stats.widths.get(m, 0) + 1
            out.append((drafted[i][:a] + [nxt], commit))
        return out

    def _accept_sampled(self, drafts, target_logits, sp, gen):
        cand, q = self.drafter.draft_distribution(drafts.shape[0], sp.temperature)
        return accept_sampled(drafts, target_logits, cand, q, sp, gen)

    def generate(self, ids: list[int], max_new: int, stop_ids=(), spec: bool | None = None,
                 sampling: "Sampling | None" = None) -> tuple[list[int], GenStats]:
        st = GenStats()
        out = list(self.stream(ids, max_new, stop_ids, sampling, spec, st))
        return out, st
