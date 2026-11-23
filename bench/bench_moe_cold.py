"""Decode MoE GEMV configs with expert weights streamed from DRAM (random experts per step over a 128-expert
layer) vs the autotuner's L2-hot choice; reports which configs give the same bits as the shipped choice.

Modal: modal run modal/engine_app.py::parity --args "bench_moe_cold.py"
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


def nvfp4(e, n, k):
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    s = (torch.rand(e, n, k // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
    return ExpertWeight("nvfp4", w, s, torch.full((e,), 0.01, device="cuda"))


def mxfp8(e, n, k):
    w = (torch.randn(e, n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 124, (e, n, k // 32), dtype=torch.uint8, device="cuda")
    return ExpertWeight("mxfp8", w, s)


# the engine's autotuned choices (print_configs.py), as (BN, BK, warps, stages)
SHIPPED = {("nvfp4", 1): (32, 256, 8, 2), ("nvfp4", 0): (32, 128, 4, 3),
           ("mxfp8", 1): (32, 128, 8, 2), ("mxfp8", 0): (32, 128, 8, 2)}
CFGS = [(bn, bk, nw, ns) for bn in (16, 32, 64) for bk in (128, 256, 512) for nw in (2, 4, 8) for ns in (2, 3, 4)]


def run(fmt_name, w, n, k, act, t):
    pairs = t * TOPK
    ids = [torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(t)]).reshape(-1).int()
           for _ in range(STEPS)]
    wts = torch.rand(pairs, device="cuda")
    x = (torch.randn(max(t, pairs), k, device="cuda") * 0.3).to(torch.bfloat16)
    xdiv = TOPK if act else 1
    out_dt = torch.bfloat16 if act else torch.float32
    out = torch.zeros(STEPS, pairs, (n // 2) if act else n, dtype=out_dt, device="cuda")
    f = w.expert_factor if w.fmt == "nvfp4" else wts
    sc = mt._scale_view(w)
    fid = mt._fmt_id(w)

    def launch(cfg=None):
        for s in range(STEPS):
            if cfg is None:
                mt._moe_gemv_kernel[lambda m: (pairs, triton.cdiv(n, m["BN"]))](
                    x, xdiv, x.stride(0), w.weight, sc, f, ids[s], out[s], out.stride(1), wts, n, k, fid, act, 4.0, 25.0)
            else:
                bn, bk, nw, ns = cfg
                mt._moe_gemv_kernel.fn[(pairs, triton.cdiv(n, bn))](
                    x, xdiv, x.stride(0), w.weight, sc, f, ids[s], out[s], out.stride(1), wts, n, k, fid, act, 4.0, 25.0,
                    BN=bn, BK=bk, num_warps=nw, num_stages=ns)

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

    shipped = SHIPPED[(w.fmt, act)]
    launch(shipped)
    torch.cuda.synchronize()
    ref = out.clone()
    res = []
    for cfg in [shipped] + [c for c in CFGS if c != shipped]:
        if k % cfg[1]:
            continue
        try:
            us = timed(cfg)
        except Exception:
            continue
        res.append((us, cfg, torch.equal(out, ref)))
    sh = res[0]
    print(f"{fmt_name} {'w13' if act else 'w2 '} T={t}: shipped {sh[1]} {sh[0]:.2f} us (bits ref)")
    for us, cfg, eq in sorted(res)[:8]:
        print(f"    {us:6.2f} us  BN={cfg[0]} BK={cfg[1]} warps={cfg[2]} stages={cfg[3]}  same bits: {eq}")


TS = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4]
FMTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["nvfp4", "mxfp8"]
for t in TS:
    if "nvfp4" in FMTS:
        run("nvfp4", nvfp4(E, 2 * I, H), 2 * I, H, 1, t)
        run("nvfp4", nvfp4(E, H, I), H, I, 0, t)
    if "mxfp8" in FMTS:
        run("mxfp8", mxfp8(E, 2 * I, H), 2 * I, H, 1, t)
        run("mxfp8", mxfp8(E, H, I), H, I, 0, t)
