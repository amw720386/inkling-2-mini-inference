"""Router (logits + top-k) decode kernel sweep: experts per program (BE) x num_warps.

Modal: modal run modal/engine_app.py::parity --args "bench_router.py"
"""

import sys

import torch
import triton

sys.path.insert(0, "/root")
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.ops import fused_triton as fk  # noqa: E402

cfg = InklingConfig.from_pretrained(sys.argv[1])
H, E, K = cfg.hidden_size, cfg.n_experts, cfg.top_k
L = 40  # distinct weight copies, rotated like the layers of a decode step
print(torch.cuda.get_device_name(0), f"H={H} E={E} top_k={K}")
ws = [torch.randn(E, H, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
bias = torch.randn(E, device="cuda") * 0.01


def run(T, be, nw):
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    lg = torch.empty(T, E, device="cuda")
    cnt = torch.zeros(T, dtype=torch.int32, device="cuda")
    ids = torch.empty(T, K, dtype=torch.int32, device="cuda")
    wts = torch.empty(T, K, device="cuda")

    def step():
        for w in ws:
            fk._router_topk_kernel[(T, E // be)](x, x.stride(0), w, bias, lg, cnt, ids, wts, 1.0, H=H,
                                                 BH=triton.next_power_of_2(H), E=E, BE=be, TOPK=K, num_warps=nw)
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
    us = e0.elapsed_time(e1) / 20 / L * 1e3
    ref = torch.topk(torch.sigmoid(x.float() @ ws[-1].float().T) + bias, K).indices.sort().values
    ok = bool((ids.sort().values == ref.int()).float().mean() > 0.95)
    return us, ok


for T in (1, 4, 8, 16, 32):
    row = []
    for be in (1, 8):
        for nw in (1, 4):
            us, ok = run(T, be, nw)
            row.append(f"be{be}w{nw}={us:.2f}{'' if ok else '!'}")
    print(f"T={T:2d} us/launch: " + " ".join(row))
