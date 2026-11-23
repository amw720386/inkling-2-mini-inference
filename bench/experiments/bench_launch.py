"""Per-kernel floor inside a CUDA graph: chains of tiny dependent kernels (Modal: parity --args "bench_launch.py")."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tiny(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) + 1.0)


@triton.jit
def _wide(X, Y, N: tl.constexpr):  # many CTAs, each a tiny slice (like a GEMV launch)
    i = tl.program_id(0) * N + tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) + 1.0)


def chain(kind, n, grid):
    x = torch.zeros(grid * 1024, device="cuda")
    y = torch.zeros_like(x)
    def run():
        a, b = x, y
        for _ in range(n):
            (_tiny if kind == "tiny" else _wide)[(grid,)](a, b, 1024, num_warps=4)
            a, b = b, a
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1000


for kind, grid in (("tiny", 1), ("wide", 188), ("wide", 576)):
    print(f"{kind} grid {grid}: {chain(kind, 600, grid):.2f} us per kernel")
