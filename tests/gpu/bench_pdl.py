"""In-process A/B/A of PDL launches: graph replay times with PDL on, off, on (graphs re-captured each time)."""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import VerifyGraph  # noqa: E402
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.fused_model import FusedDecodeGraph  # noqa: E402
from inkling_engine.ops import pdl  # noqa: E402

eng = Engine(sys.argv[1], max_ctx=8192, spec=True)
eng.prefill(list(range(1000, 1600)))


def t(fn, n=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


for flag in (True, False, True, False):
    pdl.ACTIVE[0] = flag
    dg = FusedDecodeGraph(eng.model, eng.cache, 1)
    vg = {m: VerifyGraph(eng.model, eng.cache, m) for m in (2, 4, 8)}
    eng.drafter.graphs.clear()
    eng.drafter.commit_graphs.clear()
    eng.drafter.capture()
    eng.drafter.anchor.fill_(1000)
    eng.drafter.P.fill_(600)
    dr = eng.drafter.graphs[(eng.drafter.buckets[0], 0)]
    r = [t(dg.graph.replay), t(dr.replay)] + [t(vg[m].graph.replay) for m in (2, 4, 8)]
    print(f"PDL {'on ' if flag else 'off'}: decode {r[0]:.3f}  draft {r[1]:.3f}  verify2 {r[2]:.3f}  "
          f"verify4 {r[3]:.3f}  verify8 {r[4]:.3f} ms")
