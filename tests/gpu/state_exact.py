"""fp64 check that chunked prefill reproduces one-shot prefill bit-exactly.

usage (Modal): modal run modal/engine_app.py::parity --args "state_exact.py N SPLIT [VISIT]"
Prints per-visit relative error of the second chunk; every row must be 0.
"""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import Batch, SeqChunk  # noqa: E402
from inkling_engine.cache import CacheSpec, InklingCache  # noqa: E402
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.model import InklingModel, RefBackend  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

path = sys.argv[1]
cfg = InklingConfig.from_pretrained(path)
w = load_weights(cfg, path)
cache = InklingCache(cfg, CacheSpec(slots=1, max_ctx=2048, ring=cfg.sliding_window + 64))
from inkling_engine.ops import ref  # noqa: E402
ref.COMPUTE = torch.float64
m = InklingModel(cfg, w, RefBackend(cfg, exact=True))
m.capture_visits = list(range(cfg.num_visits + 1))
torch.manual_seed(0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
split = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ids = torch.randint(1000, 50000, (N,)).tolist()

m.debug_visit = int(sys.argv[4]) if len(sys.argv) > 4 else 18
cache.reset_slot(0)
_, one = m.forward(Batch.build([SeqChunk(0, 0, N, "all")], [ids]), cache)
cache.reset_slot(0)
d_one = dict(m.debug)
_, a = m.forward(Batch.build([SeqChunk(0, 0, split, "all")], [ids[:split]]), cache)
_, b = m.forward(Batch.build([SeqChunk(0, split, N - split, "all")], [ids[split:]]), cache)
for k, v in m.debug.items():
    x = d_one[k][split:].float(); y = v.float()
    print(f"  visit {m.debug_visit} {k:6s} rel err {((x - y).norm() / x.norm().clamp_min(1e-30)).item():.2e}")
worst = 0.0
for vi in [6, 12, 18, 24, 30, 36, 42, 47, 48]:
    x = one[vi][split:].float()
    y = b[vi].float()
    pre = ((one[vi][:split].float() - a[vi].float()).norm() / a[vi].float().norm()).item()
    err = ((x - y).norm() / x.norm()).item()
    v = cfg.visits[vi] if vi < cfg.num_visits else None
    worst = max(worst, err, pre)
    print(f"before visit {vi:2d} {v}: chunk2 rel err {err:.2e} | chunk1 rel err {pre:.2e}")
print("STATE", "EXACT" if worst == 0.0 else f"MISMATCH {worst:.2e}")
sys.exit(0 if worst == 0.0 else 1)
