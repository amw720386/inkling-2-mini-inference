"""Small-M grouped MoE (verify / batched steps) configs with expert weights streamed from DRAM, vs the engine's
autotuned (L2-hot) choices; bitwise check against them.

Modal: modal run modal/engine_app.py::parity --args "bench_grouped_cold.py [T,T,..] [fmts]"
"""

import sys

import torch
import triton

sys.path.insert(0, "/root")
from inkling_engine.ops import moe_triton as mt  # noqa: E402
from inkling_engine.quant import ExpertWeight  # noqa: E402

E, H, I, TOPK, STEPS = 128, 2304, 1152, 8, 24
print(torch.cuda.get_device_name(0))
torch.manual_seed(0)
# engine choices (print_configs.py): (fmt, act, BM) -> (BN, BK, warps, stages)
SHIPPED = {("nvfp4", 1, 16): (64, 256, 8, 3), ("nvfp4", 0, 16): (64, 128, 4, 3),
           ("mxfp8", 1, 16): (64, 256, 4, 3), ("mxfp8", 0, 16): (64, 128, 4, 3),
           ("nvfp4", 1, 32): (128, 256, 8, 2), ("nvfp4", 0, 32): (128, 128, 4, 2),
           ("mxfp8", 1, 32): (64, 256, 4, 2), ("mxfp8", 0, 32): (32, 128, 8, 3)}
CFGS = [(bn, bk, nw, ns) for bn in (32, 64, 128) for bk in (64, 128, 256) for nw in (2, 4, 8) for ns in (2, 3, 4)]


def nvfp4(e, n, k):
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    s = (torch.rand(e, n, k // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
    return ExpertWeight("nvfp4", w, s, torch.full((e,), 0.01, device="cuda"))


def mxfp8(e, n, k):
    w = (torch.randn(e, n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    return ExpertWeight("mxfp8", w, torch.randint(118, 124, (e, n, k // 32), dtype=torch.uint8, device="cuda"))


def run(w, n, k, act, t):
    p = t * TOPK
    bm = 16 if t <= 16 else 32
    sc = mt.GroupedScratch(t, TOPK, E, H, I)
    plans = []
    for _ in range(STEPS):
        ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(t)]).int()
        st = torch.zeros(E, dtype=torch.int32, device="cuda")
        en = torch.zeros(E, dtype=torch.int32, device="cuda")
        rs = torch.zeros(p, dtype=torch.int32, device="cuda")
        mt._route_sort_kernel[(1,)](ids.reshape(-1), st, en, rs, p, E=E, BP=triton.next_power_of_2(p), num_warps=4)
        plans.append((st, en, rs))
    wts = torch.rand(p, device="cuda")
    x = (torch.randn(p, k, device="cuda") * 0.3).to(torch.bfloat16)
    out = torch.zeros(STEPS, p, n // 2 if act else n, dtype=torch.bfloat16 if act else torch.float32, device="cuda")
    f = w.expert_factor if w.fmt == "nvfp4" else wts
    scl = mt._scale_view(w)

    def launch(cfg):
        bn, bk, nw, ns = cfg
        for s, (st, en, rs) in enumerate(plans):
            src, div, orow = (rs, TOPK, rs) if act else (sc.ident, 1, rs)
            mt._grouped_kernel[(E, triton.cdiv(n, bn))](
                x, x.stride(0), src, div, w.weight, scl, f, sc.tile_expert, st, en, out[s], out.stride(1), wts, orow,
                n, k, mt._fmt_id(w), act, 4.0, 25.0, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)

    def timed(cfg):
        launch(cfg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch(cfg)
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 10 / STEPS * 1e3

    shipped = SHIPPED[(w.fmt, act, bm)]
    out.zero_()
    launch(shipped)
    torch.cuda.synchronize()
    ref = out.clone()
    res = []
    for cfg in [shipped] + [c for c in CFGS if c != shipped]:
        if k % cfg[1]:
            continue
        out.zero_()
        try:
            us = timed(cfg)
        except Exception:
            continue
        res.append((us, cfg, torch.equal(out, ref)))
    print(f"{w.fmt} {'w13' if act else 'w2 '} T={t} BM={bm}: shipped {shipped} {res[0][0]:.2f} us; "
          f"{sum(r[2] for r in res)}/{len(res)} configs bit-equal")
    for us, cfg, eq in sorted(res)[:4]:
        print(f"    {us:6.2f} us  BN={cfg[0]} BK={cfg[1]} warps={cfg[2]} stages={cfg[3]}  same bits: {eq}")


TS = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 16, 32]
FMTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["nvfp4", "mxfp8"]
for fm in FMTS:
    w13 = nvfp4(E, 2 * I, H) if fm == "nvfp4" else mxfp8(E, 2 * I, H)
    w2 = nvfp4(E, H, I) if fm == "nvfp4" else mxfp8(E, H, I)
    for t in TS:
        run(w13, 2 * I, H, 1, t)
        run(w2, H, I, 0, t)
