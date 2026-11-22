"""Multi-slot draft vs per-slot drafts (same contexts): token agreement and timing."""

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

eng = Engine(sys.argv[1], max_ctx=8192, spec=True, slots=4)
d = eng.drafter
torch.manual_seed(0)
prompts = [torch.randint(1000, 60000, (300 + 50 * s,)).tolist() for s in range(4)]
for s in range(4):
    if s == 0:
        eng.prefill(prompts[0])
    else:
        eng.prefill_slot(s, prompts[s])
anchors, Ps = [1234, 2345, 3456, 4567], [len(p) for p in prompts]
single = []
for s in range(4):
    tok, conf = d.draft(anchors[s], Ps[s], 0.0, slot=s)
    single.append(tok.clone())
multi = d.draft_multi(anchors, Ps, [0.0] * 4, [0, 1, 2, 3])
agree = [next((j for j in range(7) if int(multi[s][0][j]) != int(single[s][j])), 7) for s in range(4)]
print("leading draft tokens equal (multi vs single), per slot:", agree)


def t(fn, n=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


for k in (2, 4):
    sl = list(range(k))
    print(f"k={k}: {k} single drafts {t(lambda: [d.draft(anchors[s], Ps[s], 0.0, slot=s) for s in sl]):.2f} ms | "
          f"one multi draft {t(lambda: d.draft_multi(anchors[:k], Ps[:k], [0.0] * k, sl)):.2f} ms")
