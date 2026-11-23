"""Is the small-M grouped MoE kernel bandwidth- or instruction-bound? Same routing replayed (weights L2-hot) vs a
new random routing per step (weights from DRAM), NVFP4 layer, T tokens. Modal: parity --args "grouped_hot_cold.py"
"""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.ops import moe_triton as mt  # noqa: E402
from inkling_engine.quant import ExpertWeight  # noqa: E402

E, H, I, TOPK, STEPS = 128, 2304, 1152, 8, 24
torch.manual_seed(0)


def nvfp4(e, n, k):
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    s = (torch.rand(e, n, k // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
    return ExpertWeight("nvfp4", w, s, torch.full((e,), 0.01, device="cuda"))


w13, w2 = nvfp4(E, 2 * I, H), nvfp4(E, H, I)


def timed(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 5 / STEPS * 1e3


for t in (4, 8, 16, 32):
    ids = [torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(t)]).int() for _ in range(STEPS)]
    wts = torch.rand(t, TOPK, device="cuda")
    x = (torch.randn(t, H, device="cuda") * 0.5).to(torch.bfloat16)
    sc = mt.GroupedScratch(t, TOPK, E, H, I)
    rows = torch.empty(t * TOPK, H, device="cuda")
    cold = timed(lambda: [mt.moe_grouped_small(x, i, wts, w13, w2, 4.0, 25.0, sc, rows) for i in ids])
    hot = timed(lambda: [mt.moe_grouped_small(x, ids[0], wts, w13, w2, 4.0, 25.0, sc, rows) for _ in ids])
    n_exp = sum(len(torch.unique(i)) for i in ids) / STEPS
    mb = n_exp * (2 * I * H + H * I) / 2 / 1e6 * (1 + 1 / 8)  # fp4 bytes + e4m3 scales per 16
    print(f"T={t:2d}: {n_exp:5.1f} experts ({mb:5.0f} MB) | cold {cold:6.1f} us ({mb / cold * 1e-3 * 1e3:5.2f} TB/s) | "
          f"hot {hot:6.1f} us")
