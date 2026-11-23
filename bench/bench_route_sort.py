"""Expert sort for grouped MoE steps: num_warps sweep (integer results, checked equal).

Modal: modal run modal/engine_app.py::parity --args "bench_route_sort.py"
"""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root")
from inkling_engine.ops import moe_triton as mt  # noqa: E402



@triton.jit
def _old_sort(Ids, Starts, Ends, RowSrc, P, E: tl.constexpr, BP: tl.constexpr):
    """The O(P^2) rank version (reference)."""
    i = tl.arange(0, BP)
    ok = i < P
    e = tl.load(Ids + i, mask=ok, other=E)
    ea = tl.arange(0, E)
    counts = tl.sum((e[None, :] == ea[:, None]).to(tl.int32), 1)
    ends = tl.cumsum(counts, 0)
    starts = ends - counts
    tl.store(Starts + ea, starts)
    tl.store(Ends + ea, ends)
    same_before = (e[:, None] == e[None, :]) & (i[None, :] < i[:, None])
    rank = tl.sum(same_before.to(tl.int32), 1)
    start_e = tl.sum(tl.where(ea[None, :] == e[:, None], starts[None, :], 0), 1)
    tl.store(RowSrc + start_e + rank, i, mask=ok)


E, K, L = 128, 8, 48
print(torch.cuda.get_device_name(0))
for T in (2, 4, 8, 16, 32):
    p = T * K
    ids = [torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)]).int() for _ in range(L)]
    ids[0] = torch.randint(0, 4, (T, K), device="cuda", dtype=torch.int32)  # heavy collisions
    ref = []
    for l in range(L):
        st = torch.zeros(E, dtype=torch.int32, device="cuda")
        en = torch.zeros(E, dtype=torch.int32, device="cuda")
        rs = torch.zeros(p, dtype=torch.int32, device="cuda")
        _old_sort[(1,)](ids[l].reshape(-1), st, en, rs, p, E=E, BP=triton.next_power_of_2(p), num_warps=8)
        ref.append((st, en, rs))
    outs, row = {}, []
    for nw in (8, 4, 2, 1):
        st = torch.zeros(L, E, dtype=torch.int32, device="cuda")
        en = torch.zeros(L, E, dtype=torch.int32, device="cuda")
        rs = torch.zeros(L, p, dtype=torch.int32, device="cuda")

        def step():
            for l in range(L):
                mt._route_sort_kernel[(1,)](ids[l].reshape(-1), st[l], en[l], rs[l], p, E=E,
                                            BP=triton.next_power_of_2(p), num_warps=nw)
        step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        outs[nw] = (st.clone(), en.clone(), rs.clone())
        same = all(torch.equal(outs[nw][j][l], ref[l][j]) for l in range(L) for j in range(3))
        row.append(f"w{nw}={e0.elapsed_time(e1) / 20 / L * 1e3:.2f}{'' if same else '!'}")
    print(f"T={T:2d} P={p:3d} us: " + " ".join(row))
