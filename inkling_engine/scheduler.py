"""Continuous batching over the engine's slots (one GPU-owning thread).

Requests are admitted into free slots between steps (prefill runs then). A request decoding alone runs
DSpark rounds (each slot has its own draft context and verify graphs); whenever two or more requests are
decoding, they share one batched CUDA-graph decode step, and every request's draft context keeps up, so it
resumes speculation as soon as it is alone again. Slot 0 is given out first and keeps prefix reuse.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from .engine import VERIFY_WIDTHS, Engine, GenStats
from .sampling import Sampling, sample


@dataclass
class Request:
    ids: list[int]
    max_new: int
    stop: set
    sp: Sampling
    on_token: callable  # called from the scheduler thread with each token id, then None at the end
    stats: GenStats = field(default_factory=GenStats)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    slot: int = -1
    P: int = 0
    out: list = field(default_factory=list)
    gen: torch.Generator | None = None
    spec_ok: bool = False
    t_decode: float = 0.0
    t_start: float = 0.0
    prefill_gen: object = None
    image_feats: object = None


class Scheduler:
    def __init__(self, engine: Engine):
        self.e = engine
        self.inbox: queue.Queue = queue.Queue()
        self.waiting: list[Request] = []
        self.prefilling: list[Request] = []  # slot assigned, prompt being processed chunk by chunk
        self.active: dict[int, Request] = {}
        self.thread = threading.Thread(target=self._loop, daemon=True, name="inkling-scheduler")
        self.thread.start()

    # ------------------------------------------------------------------ API
    def submit(self, ids, max_new, stop, sp: Sampling, on_token, image_feats=None) -> Request:
        req = Request(list(ids), max_new, set(stop), sp, on_token, image_feats=image_feats)
        self.inbox.put(req)
        return req

    def generate(self, ids, max_new, stop=(), sp: Sampling | None = None) -> tuple[list[int], GenStats]:
        """Blocking helper (tests/benchmarks)."""
        out = []
        req = self.submit(ids, max_new, stop, sp or Sampling(), lambda t: t is not None and out.append(t))
        req.done.wait()
        if req.error:
            raise req.error
        return out, req.stats

    # ----------------------------------------------------------------- loop
    def _loop(self):
        torch.cuda.set_device(0)
        with torch.inference_mode():
            while True:
                block = not self.active and not self.waiting and not self.prefilling
                try:
                    self.waiting.append(self.inbox.get(block=block))
                    while True:
                        self.waiting.append(self.inbox.get_nowait())
                except queue.Empty:
                    pass
                self._admit()
                if self.prefilling:  # one prompt chunk, then one decode step for the others
                    self._prefill_step(self.prefilling[0])
                if not self.active:
                    continue
                try:
                    lone = next(iter(self.active.values())) if len(self.active) == 1 else None
                    if lone is not None and lone.spec_ok:
                        self._spec_step(lone)
                    else:
                        self._batched_step()
                except Exception as exc:  # fail the running requests, keep serving
                    for req in list(self.active.values()):
                        req.error = exc
                        self._finish(req)

    def _busy_slots(self):
        return set(self.active) | {r.slot for r in self.prefilling}

    def _admit(self):
        while self.waiting:
            free = [s for s in range(self.e.slots) if s not in self._busy_slots()]
            if not free:
                return
            req = self.waiting.pop(0)
            try:
                self._begin(req, 0 if 0 in free else free[0])
            except Exception as exc:
                self._fail(req, exc)

    def _fail(self, req: Request, exc: Exception):
        req.error = exc
        if req in self.prefilling:
            self.prefilling.remove(req)
        self.active.pop(req.slot, None)
        req.slot = -1
        req.on_token(None)
        req.done.set()

    def _begin(self, req: Request, slot: int):
        e = self.e
        req.stats.prompt_tokens = len(req.ids)
        req.max_new = min(req.max_new, e.max_ctx - len(req.ids) - max(VERIFY_WIDTHS))
        if req.max_new <= 0:
            raise ValueError(f"prompt of {len(req.ids)} tokens does not fit max_ctx {e.max_ctx}")
        if req.sp.seed is not None:
            req.gen = torch.Generator(device="cuda")
            req.gen.manual_seed(req.sp.seed)
        req.slot = slot
        req.t_start = time.time()
        req.prefill_gen = e.prefill_steps(slot, req.ids, req.image_feats)
        self.prefilling.append(req)

    def _prefill_step(self, req: Request):
        if req.cancel.is_set():
            self.prefilling.remove(req)
            req.slot = -1
            req.on_token(None)
            req.done.set()
            return
        try:
            next(req.prefill_gen)
            return  # more chunks to go
        except StopIteration as stop:
            logits, req.stats.reused_tokens = stop.value
        except Exception as exc:
            self._fail(req, exc)
            return
        self.prefilling.remove(req)
        e = self.e
        if req.sp.seed is not None and e.drafter is not None:
            e.drafter.seed.fill_(req.sp.seed & 0x7FFFFFFF)
        tok = sample(logits, req.sp, req.gen)
        req.stats.prefill_s = time.time() - req.t_start
        req.P, req.out, req.spec_ok = len(req.ids), [tok], e.spec
        req.t_decode = time.time()
        self.active[req.slot] = req
        req.on_token(tok)
        self._check_done(req)

    def _check_done(self, req: Request) -> bool:
        if req.out[-1] in req.stop or len(req.out) >= req.max_new or req.cancel.is_set():
            self._finish(req)
            return True
        return False

    def _finish(self, req: Request):
        st = req.stats
        st.generated = len(req.out)
        st.decode_s = time.time() - req.t_decode if req.t_decode else 0.0
        if req.slot == 0:
            self.e.state_ids = (req.ids + req.out)[:req.P]
        self.active.pop(req.slot, None)
        req.on_token(None)
        req.done.set()

    # ---------------------------------------------------------------- steps
    def _spec_step(self, req: Request):
        new, commit = self.e.spec_round(req.out[-1], req.P, req.sp, req.gen, req.stats, slot=req.slot)
        req.P += commit
        for t in new:
            req.out.append(t)
            req.on_token(t)
            if self._check_done(req):
                return

    def _batched_step(self):
        e = self.e
        reqs = [self.active[s] for s in sorted(self.active)]
        g = e.decode_graphs[len(reqs)]
        lg = g.step([r.slot for r in reqs], [r.P for r in reqs], [r.out[-1] for r in reqs])
        if e.drafter is not None:  # keep every request's draft context current for when it runs alone
            for i, r in enumerate(reqs):
                if r.spec_ok:
                    e.drafter.commit(g.ws.taps[i:i + 1], r.P, slot=r.slot)
        greedy = [i for i, r in enumerate(reqs) if r.sp.temperature <= 0]
        picks = dict(zip(greedy, lg[greedy].argmax(-1).tolist())) if greedy else {}  # one sync
        for i, r in enumerate(reqs):
            r.P += 1
            t = picks[i] if i in picks else sample(lg[i], r.sp, r.gen)
            r.out.append(t)
            r.on_token(t)
            self._check_done(r)
