"""Decode GEMV configs timed with weights rotated past L2 (the autotuner times L2-hot repeats), bitwise check.

Modal: modal run modal/engine_app.py::parity --args "bench_gemv_cold.py"
"""

import sys

import torch
import triton

sys.path.insert(0, "/root")
from inkling_engine.ops import gemv_triton as gt  # noqa: E402

print(torch.cuda.get_device_name(0))
CONFIGS = [(bn, bk, nw, ns) for bn in (16, 32, 64) for bk in (128, 256, 512, 1024) for nw in (2, 4, 8)
           for ns in (2, 3, 4, 5, 6)]


def timed(step, reps=10):
    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def sweep(name, n, k, m, launch, n_copies, fp8=False, ref_cfg=None):
    ws = [(torch.randn(n, k, device="cuda") * (100 if fp8 else 0.02)).to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
          for _ in range(n_copies)]
    x = torch.randn(3, m, k, device="cuda", dtype=torch.bfloat16)
    ref, res = None, []
    for bn, bk, nw, ns in ([ref_cfg] if ref_cfg else []) + [c for c in CONFIGS if c != ref_cfg]:
        if bn * bk * (1 if fp8 else 2) * ns > 160 * 1024:
            continue
        y = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
        try:
            ms = timed(lambda: [launch(x, w, y, bn, bk, nw, ns) for w in ws]) / n_copies
        except Exception as e:  # out of resources
            continue
        if ref is None:
            ref = y.clone()
        res.append((ms * 1e3, bn, bk, nw, ns, torch.equal(y, ref)))
    res.sort()
    gb = n * k * (1 if fp8 else 2) / 1e9
    print(f"{name} N={n} K={k} M={m}: {len(res)} configs, all bit-equal: {all(r[5] for r in res)}")
    for us, bn, bk, nw, ns, eq in res[:6]:
        print(f"   {us:6.1f} us {gb / us * 1e6:6.0f} GB/s  BN={bn} BK={bk} warps={nw} stages={ns}  same bits: {eq}")
    if ref_cfg:
        eqs = sorted((r[0], r[1:5]) for r in res if r[5])
        print(f"   bit-equal to {ref_cfg}: {len(eqs)} configs; fastest {eqs[:3]}")
    # what the shipped autotuner picks (L2-hot)
    y = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    return res


def seg_launch(x, w, y, bn, bk, nw, ns):
    n, k = w.shape
    b = (2304, 3328, 4352, 6528)  # multiples of 64 inside N
    gt._seg_gemv_kernel.fn[(triton.cdiv(n, bn),)](x, x.stride(0), x.stride(1), w, y, y, x.shape[1], n, k, w.stride(0),
                                                  y.stride(0), *b, BN=bn, BK=bk, MP=16 if x.shape[1] <= 16 else 32,
                                                  FP8=False, num_warps=nw, num_stages=ns)


def dense_launch(x, w, y, bn, bk, nw, ns):
    n, k = w.shape
    x0 = x[0]
    gt._dense_gemv_kernel.fn[(triton.cdiv(n, bn),)](x0, w, y, x0.shape[0], n, k, x0.stride(0), w.stride(0),
                                                    y.stride(0), BN=bn, BK=bk, MP=16 if x0.shape[0] <= 16 else 32,
                                                    num_warps=nw, num_stages=ns)


def fp8_launch(x, w, y, bn, bk, nw, ns):
    n, k = w.shape
    x0 = x[0]
    gt._fp8_gemv_kernel.fn[(triton.cdiv(n, bn),)](x0, w, SC[n], y, x0.shape[0], n, k, x0.stride(0), w.stride(0),
                                                  y.stride(0), BN=bn, BK=bk, MP=16 if x0.shape[0] <= 16 else 32,
                                                  num_warps=nw, num_stages=ns)


def seg8_launch(x, w, y, bn, bk, nw, ns):
    n, k = w.shape
    b = (2304, 3328, 4352, 6528)
    gt._seg_gemv_kernel.fn[(triton.cdiv(n, bn),)](x, x.stride(0), x.stride(1), w, SC[n], y, x.shape[1], n, k,
                                                  w.stride(0), y.stride(0), *b, BN=bn, BK=bk,
                                                  MP=16 if x.shape[1] <= 16 else 32, FP8=True, num_warps=nw,
                                                  num_stages=ns)


SC = {}
if len(sys.argv) > 2 and sys.argv[2] == "fp8":
    # drafter / dense_fp8 shapes; engine picks (print_configs.py) for reference
    SHIP = {5120: (16, 512, 4, 3), 2304: (32, 512, 4, 4), 18432: (64, 256, 8, 3), 32827: (32, 128, 4, 4)}
    for n, k in ((5120, 2304), (18432, 2304), (2304, 9216), (32827, 2304), (2304, 3072)):
        SC[n] = torch.rand(n, device="cuda") + 0.5
        res = sweep(f"fp8 (engine {SHIP.get(n)})", n, k, 1, fp8_launch, max(3, 400 * 2**20 // (n * k)), fp8=True,
                    ref_cfg=SHIP.get(n))
        SHIPPED_US = [r for r in res if (r[1], r[2], r[3], r[4]) == SHIP.get(n)]
        if SHIPPED_US:
            print(f"   engine config: {SHIPPED_US[0][0]:.1f} us")
    SC[8768] = torch.rand(8768, device="cuda") + 0.5
    sweep("seg fp8", 8768, 2304, 1, seg8_launch, 20, fp8=True)
else:
    for m in (1, 8, 32):
        sweep("seg", 8768, 2304, m, seg_launch, 10)
        sweep("wo ", 2304, 3072, m, dense_launch, 24)
