"""Where the time of a scheduler batched step goes (graph replay, sampling sync, draft-context commits)."""

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

eng = Engine(sys.argv[1], max_ctx=8192, spec=True, slots=4)
for s in range(4):
    if s == 0:
        eng.prefill(list(range(1000, 1100)))
    else:
        eng.prefill_slot(s, list(range(1000, 1100)))


def run(k, steps, commit, sync):
    g = eng.decode_graphs[k]
    P = [100] * k
    toks = [1000] * k
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        lg = g.step(list(range(k)), P, toks)
        if commit:
            for i in range(k):
                eng.drafter.commit(g.ws.taps[i:i + 1], P[i], slot=i)
        if sync:
            toks = lg.argmax(-1).tolist()
        P = [p + 1 for p in P]
    torch.cuda.synchronize()
    return (time.time() - t0) / steps * 1000


for k in (2, 4):
    print(f"bs={k}: replay only {run(k, 100, False, False):.2f} ms | + argmax sync {run(k, 100, False, True):.2f} ms"
          f" | + draft commits {run(k, 100, True, True):.2f} ms per step")
