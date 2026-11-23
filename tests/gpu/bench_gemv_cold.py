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


def sweep(name, n, k, m, launch, n_copies):
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(n_copies)]
    x = torch.randn(3, m, k, device="cuda", dtype=torch.bfloat16)
    ref, res = None, []
    for bn, bk, nw, ns in CONFIGS:
        if bn * bk * 2 * ns > 160 * 1024:
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
    gb = n * k * 2 / 1e9
    print(f"{name} N={n} K={k} M={m}: {len(res)} configs, all bit-equal: {all(r[5] for r in res)}")
    for us, bn, bk, nw, ns, eq in res[:6]:
        print(f"   {us:6.1f} us {gb / us * 1e6:6.0f} GB/s  BN={bn} BK={bk} warps={nw} stages={ns}")
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


for m in (1, 8, 32):
    sweep("seg", 8768, 2304, m, seg_launch, 10)
    sweep("wo ", 2304, 3072, m, dense_launch, 24)
