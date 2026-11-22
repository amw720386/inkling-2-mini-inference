"""Pure-read bandwidth ceiling vs the engine's GEMVs (weights rotated past L2)."""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root")
from inkling_engine.ops.gemv_triton import gemv, seg_gemv  # noqa: E402


@triton.jit
def _read_kernel(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in tl.range(pid * BLOCK, N, tl.num_programs(0) * BLOCK):
        acc += tl.load(X + start + tl.arange(0, BLOCK), mask=start + tl.arange(0, BLOCK) < N, other=0.0,
                       eviction_policy="evict_first").to(tl.float32)
    tl.store(Out + pid, tl.sum(acc, 0))


def t(fn, n=20):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1e3


x = torch.randn(512 * 2**20, device="cuda", dtype=torch.bfloat16)  # 1 GiB
out = torch.empty(4096, device="cuda")
sms = torch.cuda.get_device_properties(0).multi_processor_count
for progs in (sms, 2 * sms, 4 * sms, 8 * sms):
    for blk in (1024, 4096):
        us = t(lambda: _read_kernel[(progs,)](x, out, x.numel(), BLOCK=blk, num_warps=8))
        print(f"read 1 GiB: programs {progs:5d} block {blk}: {x.numel() * 2 / us / 1e3:.0f} GB/s")
us = t(lambda: x.sum())
print(f"torch.sum 1 GiB: {x.numel() * 2 / us / 1e3:.0f} GB/s")
del x
ws = [torch.randn(8768, 2304, device="cuda").to(torch.bfloat16) for _ in range(12)]
xin = torch.randn(3, 1, 2304, device="cuda").to(torch.bfloat16)
o = torch.empty(1, 8768, device="cuda", dtype=torch.bfloat16)
i = [0]


def run():
    i[0] += 1
    seg_gemv(xin, ws[i[0] % 12], o, (6592, 7616, 8640, 8704))


us = t(run, 60)
print(f"seg_gemv 8768x2304 bf16: {us:.1f} us -> {8768 * 2304 * 2 / us / 1e3:.0f} GB/s")

from inkling_engine.ops import gemv_triton as G  # noqa: E402

x1 = torch.randn(1, 2304, device="cuda").to(torch.bfloat16)


def run_plain():
    i[0] += 1
    gemv(x1, ws[i[0] % 12], o)


us = t(run_plain, 60)
print(f"plain gemv 8768x2304 bf16: {us:.1f} us -> {8768 * 2304 * 2 / us / 1e3:.0f} GB/s")
print("seg best config:", G._seg_gemv_kernel.best_config, "| plain best config:", G._dense_gemv_kernel.best_config)
