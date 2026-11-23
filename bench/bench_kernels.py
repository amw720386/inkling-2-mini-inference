"""Bandwidth microbenchmarks for the decode-critical kernels (weights rotated past L2).

Modal: modal run modal/engine_app.py::parity --args "bench_kernels.py"
"""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.ops import ref  # noqa: E402
from inkling_engine.ops.gemv_triton import gemv  # noqa: E402
from inkling_engine.ops.moe_triton import moe_decode  # noqa: E402
from inkling_engine.quant import ExpertWeight  # noqa: E402


def timeit(fn, n=50):
    for _ in range(3):
        fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1e3  # us


print(torch.cuda.get_device_name(0))
# peak copy bandwidth
a = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
b = torch.empty_like(a)
us = timeit(lambda i: b.copy_(a), n=20)
print(f"copy 512MiB: {us:.0f} us -> {2 * a.numel() / us / 1e3:.0f} GB/s (read+write)")
del a, b

ROT = 12  # distinct weight copies, > L2
for name, n, k in [("w_all", 8600, 2304), ("w_qgrs", 6552, 2304), ("wk", 1024, 2304), ("wo", 2304, 3072),
                   ("dyn", 32, 2304)]:
    rot = 1 if name == "unembed" else ROT
    ws = [torch.randn(n, k, device="cuda").to(torch.bfloat16) for _ in range(rot)]
    x = torch.randn(1, k, device="cuda").to(torch.bfloat16)
    out = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
    nbytes = n * k * 2
    t_tr = timeit(lambda i: gemv(x, ws[i % rot], out))
    t_cb = timeit(lambda i: torch.mm(x, ws[i % rot].t(), out=out))
    print(f"{name:8s} [{n}x{k}] {nbytes / 2**20:7.1f} MiB  triton {t_tr:7.1f} us ({nbytes / t_tr / 1e3:5.0f} GB/s)"
          f"  cublas {t_cb:7.1f} us ({nbytes / t_cb / 1e3:5.0f} GB/s)")
    del ws

# MoE decode, real-sized random weights
for fmt in ("nvfp4", "mxfp8"):
    e = 128
    if fmt == "nvfp4":
        w13 = ExpertWeight(fmt, torch.randint(0, 255, (e, 2304, 1152), dtype=torch.uint8, device="cuda"),
                           (torch.rand(e, 2304, 144, device="cuda") * 0.01).to(torch.float8_e4m3fn),
                           torch.full((e,), 0.5, device="cuda"))
        w2 = ExpertWeight(fmt, torch.randint(0, 255, (e, 2304, 576), dtype=torch.uint8, device="cuda"),
                          (torch.rand(e, 2304, 72, device="cuda") * 0.01).to(torch.float8_e4m3fn),
                          torch.full((e,), 0.5, device="cuda"))
    else:
        w13 = ExpertWeight(fmt, (torch.randn(e, 2304, 2304, device="cuda") * 0.1).to(torch.float8_e4m3fn),
                           torch.full((e, 2304, 72), 120, dtype=torch.uint8, device="cuda"))
        w2 = ExpertWeight(fmt, (torch.randn(e, 2304, 1152, device="cuda") * 0.1).to(torch.float8_e4m3fn),
                          torch.full((e, 2304, 36), 120, dtype=torch.uint8, device="cuda"))
    x = torch.randn(1, 2304, device="cuda").to(torch.bfloat16)
    perm = torch.randperm(128, device="cuda").view(16, 8)
    wts = torch.full((1, 8), 1.0, device="cuda")
    act = torch.empty(8, 1152, dtype=torch.bfloat16, device="cuda")
    rows = torch.empty(8, 2304, dtype=torch.float32, device="cuda")
    per = sum(t[0].numel() * t.element_size() for t in (w13.weight, w2.weight, w13.scale, w2.scale))
    for hw in ((True, False) if fmt == "nvfp4" else (True,)):
        us = timeit(lambda i: moe_decode(x, perm[i % 16].view(1, 8), wts, w13, w2, 4.0, 25.0, act, rows, hw_fp4=hw))
        print(f"moe {fmt} hw_fp4={hw} bs1 (w13+w2, 8 experts): {us:.1f} us -> {8 * per / us / 1e3:.0f} GB/s")
    if fmt == "nvfp4":
        r1 = moe_decode(x, perm[0].view(1, 8), wts, w13, w2, 4.0, 25.0, act, rows, hw_fp4=True).clone()
        r2 = moe_decode(x, perm[0].view(1, 8), wts, w13, w2, 4.0, 25.0, act, rows, hw_fp4=False).clone()
        print("hw vs sw decode max|d| =", (r1 - r2).abs().max().item(), "max|r| =", r2.abs().max().item())
