"""Programmatic dependent launch from Triton inside a CUDA graph: correctness + launch-gap savings."""

import torch
import triton
import triton.language as tl


@triton.jit
def _step(X, Y, N: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) + 1.0)


def chain(x, y, n_launch, pdl):
    a, b = x, y
    for _ in range(n_launch):
        if pdl:
            _step[(1,)](a, b, 1024, PDL=True, launch_pdl=True)
        else:
            _step[(1,)](a, b, 1024, PDL=False)
        a, b = b, a


def test_pdl_chain():
    print(triton.__version__)
    x = torch.zeros(1024, device="cuda")
    y = torch.zeros(1024, device="cuda")
    times = {}
    for pdl in (False, True):
        x.zero_()
        y.zero_()
        chain(x, y, 2, pdl)  # compile
        x.zero_()
        y.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            chain(x, y, 200, pdl)
        g.replay()
        torch.cuda.synchronize()
        # 200 increments alternating buffers: x holds 199 + offset pattern; check ordering held
        assert x[0].item() == 200.0 or y[0].item() == 200.0, (x[0].item(), y[0].item())
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        times[pdl] = e0.elapsed_time(e1) / 10 / 200 * 1e3
    print(f"per-launch time in graph: plain {times[False]:.2f} us, PDL {times[True]:.2f} us")
