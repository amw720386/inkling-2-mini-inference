"""Decode message-gate kernels (down / up): launch-config sweep, bitwise check against the shipped config.

Modal: modal run modal/engine_app.py::parity --args "bench_msg.py"
"""

import sys

import torch
import triton

sys.path.insert(0, "/root")
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.ops import fused_triton as fk  # noqa: E402

cfg = InklingConfig.from_pretrained(sys.argv[1])
H, R = cfg.hidden_size, cfg.gate_rank
NG, L = 2, 24  # gates per call, distinct weight copies rotated like layers
print(torch.cuda.get_device_name(0), f"H={H} rank={R}")
dev = "cuda"
downs = [torch.randn(NG * R, 2 * H, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(L)]
gains = [torch.randn(NG * H, R, device=dev, dtype=torch.bfloat16) * 0.1 for _ in range(L)]
forgets = [torch.randn(NG * H, R, device=dev, dtype=torch.bfloat16) * 0.1 for _ in range(L)]


def timed(step):
    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 20 / L * 1e3


for T in (1, 8, 32):
    hs = torch.randn(NG, T, H, device=dev)
    ms = torch.randn(NG, T, H, device=dev)
    # ---- down
    ref = None
    row = []
    for br, nw in ((4, 4), (1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (8, 4), (8, 8)):
        gb = torch.zeros(T, NG, R, device=dev)

        def step():
            for d in downs:
                fk._msg_down_kernel[(T, NG, R // br)](hs, hs.stride(0), hs.stride(1), ms, ms.stride(0), ms.stride(1),
                                                      d, gb, cfg.rms_eps, GATE0=0, NGATE=NG, H=H,
                                                      BH=triton.next_power_of_2(H), RANK=R, BR=br, BK=1024,
                                                      num_warps=nw)
        us = timed(step)
        if ref is None:
            ref = gb.clone()
        row.append(f"br{br}w{nw}={us:.2f}{'' if torch.equal(gb, ref) else '!'}")
    print(f"T={T:2d} down us: " + " ".join(row))
    g_in = ref
    # ---- up (no norm)
    ref = None
    row = []
    for bc, nw in ((128, 4), (128, 1), (128, 2), (128, 8)):
        nblk = triton.cdiv(H, bc)
        dst = torch.zeros(NG, T, H, device=dev)

        def step():
            for gn, fg in zip(gains, forgets):
                fk._msg_up_kernel[(T, NG, nblk)](hs, hs.stride(0), hs.stride(1), ms, ms.stride(0), ms.stride(1), g_in,
                                                 gn, fg, dst, dst.stride(0), dst.stride(1), g_in, 1,
                                                 GATE0=0, NGATE=NG, NORM=False, H=H, RANK=R, BC=bc,
                                                 Counter=g_in, Wn=g_in, Out=dst, BH=triton.next_power_of_2(H),
                                                 num_warps=nw)
        us = timed(step)
        if ref is None:
            ref = dst.clone()
        row.append(f"bc{bc}w{nw}={us:.2f}{'' if torch.equal(dst, ref) else '!'}")
    print(f"T={T:2d} up   us: " + " ".join(row))
