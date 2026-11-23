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


class GpuTimer:
    """Measured GPU time of graph replays without extra syncs: events around a replay are read once they have
    completed (after the round's own host sync), and `harvest` hands each (key, ms) to a callback."""

    def __init__(self):
        self.events, self.pending = {}, []

    def start(self, key):
        ev = self.events.get(key)
        if ev is None:
            ev = self.events[key] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        if any(k == key for k, _ in self.pending):  # still unread: skip this sample
            return None
        ev[0].record()
        return ev

    def stop(self, key, ev):
        if ev is not None:
            ev[1].record()
            self.pending.append((key, ev))

    def harvest(self, fn):
        keep = []
        for key, (e0, e1) in self.pending:
            if e1.query():
                fn(key, e0.elapsed_time(e1))
            else:
                keep.append((key, (e0, e1)))
        self.pending = keep


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


CHUNK_GRAPH_MAX = 32  # prefill chunks up to this many tokens replay a captured graph


class ChunkGraph:
    """Captured prefill of one n-token chunk (n <= CHUNK_GRAPH_MAX) of a sequence in any slot at any start,
    committing its state like the eager path. Eager prefill launches ~800 kernels, so short prompts were bound by
    launch latency (an 8-token prompt took 21 ms, the same work replays in ~8 ms). All chunk graphs share one
    workspace (they never run concurrently)."""

    def __init__(self, model: FusedModel, cache: InklingCache, n: int, ws, pool=None):
        self.model, self.cache, self.n, self.ws = model, cache, n, ws
        seqs = [SeqChunk(slot=0, start=0, n=n, want_logits="last")]
        self.batch = Batch(seqs=seqs, input_ids=torch.zeros(n, dtype=torch.int64, device="cuda"),
                           positions=torch.arange(n, dtype=torch.int64, device="cuda"), offsets=[0],
                           logit_rows=torch.full((1,), n - 1, dtype=torch.int64, device="cuda"))
        self.meta = build_step_meta(self.batch, 4, 16)
        self.base_pos = torch.arange(n, dtype=torch.int64, device="cuda")
        snap = [(x, x[:, 0].clone()) for x in (cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv)]
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
            x[:, 0].copy_(v)

    def _run(self):
        return self.model.forward(self.batch, self.cache, self.ws, self.meta)

    def run(self, tokens: list[int], slot: int, start: int):
        """Returns (last-row logits [1, V], taps [n, 4, H]) views into the shared workspace."""
        self.batch.input_ids.copy_(torch.tensor(tokens, dtype=torch.int64), non_blocking=True)
        torch.add(self.base_pos, start, out=self.batch.positions)
        self.meta.local.seq_slot.fill_(slot)
        self.meta.local.seq_start.fill_(start)
        self.graph.replay()
        return self.logits, self.ws.taps[:self.n]


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
                 slots: int = 1, vision: bool = False, fp4_prefill: bool | None = None):
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
        self.model = FusedModel(self.cfg, self.w, dense_fp8=dense_fp8, fp4_prefill=fp4_prefill)
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
        self.chunk_graphs = {}
        if graphs:
            self.cws = self.model.workspace(CHUNK_GRAPH_MAX, 1, 16)
            self.chunk_graphs = {n: ChunkGraph(self.model, self.cache, n, self.cws, pool=self.pool)
                                 for n in range(1, CHUNK_GRAPH_MAX + 1)}
        if spec and graphs:
            self.drafter.capture(self.pool)
        self.gamma = 2.0
        self.draft_cost, self.verify_costs, self.decode_costs, self.draft_multi_costs = 1.0, {}, {}, {}
        self.timer = GpuTimer()
        self.batch_rounds: dict = {}  # k -> batched rounds so far
        self.costs = self._calibrate() if spec and graphs else {m: 1.0 + 0.1 * m for m in VERIFY_WIDTHS}
        self.seeded_costs = dict(self.costs)  # fixed at calibration: a seeded request's widths never depend on traffic
        self.verify_tokens = {m: torch.zeros(m, dtype=torch.int64, device="cuda") for m in VERIFY_WIDTHS}
        self.slot_ids: list[list[int]] = [[] for _ in range(slots)]  # tokens whose positions each slot holds
        self.snaps: list = [None] * slots  # per slot: state at the end of its last prompt (see _snapshot)
        # compile and autotune the prefill kernels (first and later chunks) now, not in the first long request
        warm = min(2 * prefill_chunk + 100, max_ctx - 16)
        self.prefill(list(range(1000, 1000 + warm)))
        self.cache.reset_slot(0)
        self.slot_ids[0], self.snaps[0] = [], None
        if self.drafter is not None:
            self.drafter.reset()

    @property
    def state_ids(self) -> list[int]:  # slot 0 (single-stream API)
        return self.slot_ids[0]

    @state_ids.setter
    def state_ids(self, ids: list[int]):
        self.slot_ids[0] = ids

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
        # distinct tokens per row: identical rows route to the same experts and hide the cost of width and
        # batch size (the MoE reads every hit expert); `_observe_cost` then tracks the measured replays
        gen = torch.Generator(device="cuda").manual_seed(0)
        for g in list(self.verify.values()) + list(self.decode_graphs.values()):
            ids = g.batch.input_ids
            ids.copy_(torch.randint(1000, 150000, ids.shape, generator=gen, device="cuda"))
        costs = {m: draft + t(lambda g=self.verify[(1, m)]: g.graph.replay()) for m in VERIFY_WIDTHS}
        # for the scheduler's choice between batched speculation and batched plain decode
        self.draft_cost = draft
        self.verify_costs = {km: t(lambda g=g: g.graph.replay(), 10) for km, g in self.verify.items()}
        self.decode_costs = {k: t(lambda g=g: g.graph.replay(), 10) for k, g in self.decode_graphs.items()}
        self.draft_multi_costs = {k: draft * (1 + 0.5 * (k - 1)) for k in self.decode_graphs}  # then measured
        for x, v in zip(convs, snap):
            x.copy_(v)
        self.drafter.reset()
        return costs

    COST_EMA = 0.2  # weight of a new measurement

    def _observe_cost(self, key, ms: float):
        """Online cost tracking from measured replays (GpuTimer): ("verify", k, m), ("decode", k), ("draft", k)."""
        a = self.COST_EMA
        if key[0] == "verify":
            km = key[1:]
            if km in self.verify_costs:
                self.verify_costs[km] = (1 - a) * self.verify_costs[km] + a * ms
                if km[0] == 1:
                    self.costs[km[1]] = self.draft_cost + self.verify_costs[km]
        elif key[0] == "decode" and key[1] in self.decode_costs:
            self.decode_costs[key[1]] = (1 - a) * self.decode_costs[key[1]] + a * ms
        elif key[0] == "draft" and key[1] in self.draft_multi_costs:
            self.draft_multi_costs[key[1]] = (1 - a) * self.draft_multi_costs[key[1]] + a * ms

    def harvest_costs(self):
        if self.verify_costs:
            self.timer.harvest(self._observe_cost)

    def _expected(self, conf: list[float], m: int, gamma: float | None = None) -> float:
        """Expected accepted drafts of a width-m verify: leading products of conf ** gamma."""
        g = self.gamma if gamma is None else gamma
        exp, run = 0.0, 1.0
        for c in conf[:m - 1]:
            run *= c ** g
            exp += run
        return exp

    def _pick_width(self, conf: list[float], gamma: float | None = None, seeded: bool = False) -> int:
        """Width maximizing expected committed tokens per ms. The drafter's confidences overstate
        acceptance; gamma (confidence ** gamma) is tuned online by `_observe`. Seeded requests use the fixed
        calibration costs (reproducible); the others the online measured ones."""
        costs = self.seeded_costs if seeded else self.costs
        best, best_rate = VERIFY_WIDTHS[0], -1.0
        for m in VERIFY_WIDTHS:
            rate = (1.0 + self._expected(conf, m, gamma)) / costs[m]
            if rate > best_rate:
                best, best_rate = m, rate
        return best

    # common verify width of a batched round (bench/experiments/sched_widths.py, greedy, 4 requests): width 4
    # 532-535 tok/s vs 449 (2), 447-461 (8), 472 (3), 474 (6), 511-519 for the per-round confidence pick, 451-528
    # for a measured-rate bandit, 477-514 for ragged per-request widths. Width 8 pays only for a pair of requests
    # that both accept well (2 requests, rep 0: 472 vs 447 for 4); in a 4-request run's tail it lost (517-526).
    BATCH_WIDTH = 4
    WIDE_PAIR_ACCEPT = 1.0  # mean recent accepted drafts per round of a 2-request batch for width 8 (1.0: 482 vs
    # 472 tok/s for 1.5 on prompts 0-1, level on the others)

    predraft = True  # batched rounds launch the next round's multi-slot draft before returning
    predraft_key = None

    def _pick_width_batched(self, reqs) -> int:
        if len(reqs) == 2 and sum(getattr(r, "accept_ema", 0.0) for r in reqs) / 2 >= self.WIDE_PAIR_ACCEPT:
            return 8
        return self.BATCH_WIDTH

    def _observe(self, conf: list[float], m: int, accepted: int):
        err = accepted - self._expected(conf, m)
        self.gamma = min(8.0, max(0.5, self.gamma - 0.05 * err))

    # ------------------------------------------------------------------ prefill
    def prefill_slot(self, slot: int, ids: list[int], image_feats=None) -> torch.Tensor:
        """Fresh prefill of `ids` into a non-zero slot (no prefix reuse). Returns last logits."""
        assert slot != 0
        return _drain(self.prefill_steps(slot, ids, image_feats))[0]

    # ----------------------------------------------------------- prefix reuse
    def _snap_state(self, slot: int) -> list[torch.Tensor]:
        """The slot's state a snapshot keeps: conv states, sliding rings and the drafter's rings. Global KV is
        position-indexed, so positions < len(ids) stay valid without a copy."""
        c = self.cache
        state = [t[:, slot] for t in (c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv, c.local_k, c.local_v)]
        if self.drafter is not None:
            d = self.drafter
            d._use(slot)
            state += [d.k_ctx[l] for l in (0, 2)] + [d.v_ctx[l] for l in (0, 2)] + [d.ctx_pos[l] for l in (0, 2)]
            state.append(d.f_rnn)
        return state

    def _snapshot(self, ids: list[int], slot: int = 0):
        """Copy the slot's state at the end of a prompt (buffers allocated on the slot's first snapshot)."""
        state = self._snap_state(slot)
        old = self.snaps[slot]
        bufs = old["bufs"] if old is not None else [torch.empty_like(t) for t in state]
        for b, t in zip(bufs, state):
            b.copy_(t)
        self.snaps[slot] = {"ids": list(ids), "bufs": bufs,
                            "draft_len": self.drafter.lengths[slot] if self.drafter is not None else 0}

    def _restore(self, slot: int = 0):
        snap = self.snaps[slot]
        for t, b in zip(self._snap_state(slot), snap["bufs"]):
            t.copy_(b)
        if self.drafter is not None:
            self.drafter.lengths[slot] = snap["draft_len"]

    def reusable(self, slot: int, ids: list[int]) -> int:
        """Prompt tokens of `ids` that `slot` can reuse: everything it processed so far when `ids` extends that,
        else its last prompt (snapshot) when `ids` extends that prompt; 0 for prompts with images."""
        if IMAGE_TOKEN_ID in ids:  # placeholder ids do not identify the image
            return 0
        prev = self.slot_ids[slot]
        if 0 < len(prev) < len(ids) and ids[:len(prev)] == prev and (
                self.drafter is None or self.drafter.lengths[slot] == len(prev)):
            return len(prev)
        snap = self.snaps[slot]
        if snap is not None and 0 < len(snap["ids"]) < len(ids) and ids[:len(snap["ids"])] == snap["ids"]:
            return len(snap["ids"])
        return 0

    def prefill(self, ids: list[int], image_feats=None) -> tuple[torch.Tensor, int]:
        """Bring slot 0 to `ids`; see prefill_steps. Returns (last logits, reused tokens)."""
        return _drain(self.prefill_steps(0, ids, image_feats))

    @torch.inference_mode()
    def prefill_steps(self, slot: int, ids: list[int], image_feats=None):
        """Generator: prefill `ids` into `slot` one chunk per step (yields between chunks so a scheduler can
        interleave decode steps); returns (last logits, reused tokens). The slot reuses (a) everything it processed
        so far when `ids` extends it, or (b) its previous prompt's snapshot when `ids` extends that prompt (a chat
        turn whose template re-renders the earlier answer); see `reusable`."""
        start = 0
        img_pos = [i for i, t in enumerate(ids) if t == IMAGE_TOKEN_ID]
        if img_pos and (image_feats is None or image_feats.shape[0] != len(img_pos)):
            raise ValueError(f"{len(img_pos)} image placeholder tokens but "
                             f"{0 if image_feats is None else image_feats.shape[0]} image features")
        start = self.reusable(slot, ids)
        prev = self.slot_ids[slot]
        extends_prev = start > 0 and start == len(prev) and ids[:start] == prev
        if start and not extends_prev:  # resume from the snapshot of the slot's last prompt
            self._restore(slot)
        # the slot's cache changes from here on: nothing in it is reusable until this prompt completes
        self.slot_ids[slot] = []
        if start == 0 and self.snaps[slot] is not None:
            self.snaps[slot]["ids"] = []
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
            cg = self.chunk_graphs.get(len(part)) if ef is None else None
            if cg is not None:
                logits, taps = cg.run(part, slot, s)
            else:
                logits = self.model.forward(b, self.cache, self.pws, embed_feats=ef)
                taps = self.pws.taps[:len(part)]
            if self.drafter is not None:
                self.drafter.commit(taps, s, slot=slot)
            if s + self.prefill_chunk < len(ids):
                yield
        self.slot_ids[slot] = list(ids)
        self._snapshot(ids, slot)
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
        self.harvest_costs()
        drafts, conf = self.drafter.draft(anchor, P, self._draft_temp(sp), slot=slot)
        info = torch.cat([conf, drafts.float()]).tolist()  # one sync
        confs, drafted = info[:7], [int(d) for d in info[7:]]
        seeded = sp.seed is not None
        m = self._pick_width(confs, SEEDED_GAMMA if seeded else None, seeded)
        vt = self.verify_tokens[m]
        vt[0] = anchor
        vt[1:].copy_(drafts[:m - 1])
        ev = self.timer.start(("verify", 1, m)) if self.verify_costs else None
        logits, taps = self.verify[(1, m)].run(vt, [slot], [P])
        self.timer.stop(("verify", 1, m), ev)
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
        self.harvest_costs()
        k = len(reqs)
        temps = [self._draft_temp(r.sp) for r in reqs]
        key = (tuple(r.slot for r in reqs), tuple(r.P for r in reqs), tuple(r.out[-1] for r in reqs), tuple(temps))
        if self.predraft_key == key:  # launched at the end of the previous round, already running or done
            multi = [(d.m_tok[i], d.m_conf[i], d.m_cand[i], d.m_heads.logits[i]) for i in range(k)]
        else:
            ev = self.timer.start(("draft", k)) if self.verify_costs else None
            multi = d.draft_multi([r.out[-1] for r in reqs], [r.P for r in reqs], temps, [r.slot for r in reqs])
            self.timer.stop(("draft", k), ev)
        self.predraft_key = None
        for r, (tok, conf, cand, lg_c) in zip(reqs, multi):
            q = torch.softmax(lg_c / self._draft_temp(r.sp), -1) if r.sp.temperature > 0 else None
            drafts.append((tok, conf, cand if q is not None else None, q))
        m_tok = d.m_tok[:k]  # [k, 7] drafts of batch positions 0..k-1 (the views returned above)
        # the width depends only on k, so the verify is queued right behind the draft: no host round trip between
        # them; drafts, confidences and the greedy acceptance come back in one sync after the verify
        m = self._pick_width_batched(reqs)
        anchors = torch.tensor([r.out[-1] for r in reqs], dtype=torch.int64).to("cuda", non_blocking=True)
        vt = torch.cat([anchors[:, None], m_tok[:, :m - 1]], 1).reshape(-1)
        ev = self.timer.start(("verify", k, m))
        logits, taps = self.verify[(k, m)].run(vt, [r.slot for r in reqs], [r.P for r in reqs])
        self.timer.stop(("verify", k, m), ev)
        lg = logits.view(k, m, -1)
        greedy = [i for i, r in enumerate(reqs) if r.sp.temperature <= 0]
        parts = [torch.cat([d.m_conf[:k], m_tok.float()], 1).reshape(-1)]  # ids < 2^24: exact in fp32
        if greedy:
            if len(greedy) == k:  # no index tensor (a blocking host-to-device copy)
                target, dr = lg.argmax(-1), m_tok[:, :m - 1]
            else:
                gi = torch.tensor(greedy, dtype=torch.int64).to("cuda", non_blocking=True)
                target = lg[gi].argmax(-1)  # [g, m]
                dr = m_tok[gi, :m - 1]
            acc = (dr == target[:, :m - 1]).int().cumprod(1).sum(1)
            parts.append(torch.cat([acc[:, None], target], 1).float().reshape(-1))
        vals = torch.cat(parts).tolist()  # the round's one sync
        confs = [vals[i * 14:i * 14 + 7] for i in range(k)]
        drafted = [[int(x) for x in vals[i * 14 + 7:i * 14 + 14]] for i in range(k)]
        res = {}
        base = 14 * k
        for j, i in enumerate(greedy):
            row = vals[base + j * (m + 1):base + (j + 1) * (m + 1)]
            a = int(row[0])
            res[i] = (a, int(row[1 + a]))
        for i, r in enumerate(reqs):
            if i not in res:
                _, _, cand, q = drafts[i]
                res[i] = accept_sampled(drafts[i][0][:m - 1], lg[i], cand[:m - 1], q[:m - 1], r.sp, r.gen)
        out = []
        self.cache.commit_spec_multi([r.slot for r in reqs], [res[i][0] + 1 for i in range(k)])
        for i, r in enumerate(reqs):
            a, nxt = res[i]
            commit = a + 1
            if r.sp.seed is None:
                self._observe(confs[i], m, a)
            d.commit(taps[i * m:i * m + commit], r.P, slot=r.slot)
            r.stats.rounds += 1
            r.stats.accepted += a
            r.stats.widths[m] = r.stats.widths.get(m, 0) + 1
            out.append((drafted[i][:a] + [nxt], commit))
        if self.predraft:
            # the next round's drafts (same requests, their new anchors / positions) go to the GPU now, so it drafts
            # while the host hands out this round's tokens; the next round uses them if nothing changed
            nP = [r.P + c for r, (_, c) in zip(reqs, out)]
            nA = [toks[-1] for toks, _ in out]
            d.draft_multi(nA, nP, temps, [r.slot for r in reqs])
            self.predraft_key = (tuple(r.slot for r in reqs), tuple(nP), tuple(nA), tuple(temps))
        return out

    # sampled requests draft at temperature * scale; acceptance stays exact for any scale > 0. 0.8 measured best
    # at temperature 1 (bench/experiments/bench_draft_temp.py: +2-4% tok/s over 1.0; 0.7 and below lose)
    draft_temp_scale = 0.8

    def _draft_temp(self, sp: Sampling) -> float:
        return sp.temperature * self.draft_temp_scale if sp.temperature > 0 else 0.0

    def _accept_sampled(self, drafts, target_logits, sp, gen):
        cand, q = self.drafter.draft_distribution(drafts.shape[0], self._draft_temp(sp))
        return accept_sampled(drafts, target_logits, cand, q, sp, gen)

    def generate(self, ids: list[int], max_new: int, stop_ids=(), spec: bool | None = None,
                 sampling: "Sampling | None" = None) -> tuple[list[int], GenStats]:
        st = GenStats()
        out = list(self.stream(ids, max_new, stop_ids, sampling, spec, st))
        return out, st
