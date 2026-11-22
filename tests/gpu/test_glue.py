"""Prefill glue kernels vs eager torch: expert tile planning, router GEMM, message gates (synthetic weights)."""

from types import SimpleNamespace

import torch

from inkling_engine.ops import fused_triton as fk
from inkling_engine.ops.moe_triton import plan_tiles


def test_plan_tiles():
    torch.manual_seed(0)
    for t, bm in ((300, 64), (2048, 128), (2048, 64), (5, 128)):
        ids = torch.randint(0, 128, (t, 8), device="cuda", dtype=torch.int32)
        ids[: t // 3] = torch.arange(8, device="cuda", dtype=torch.int32)  # skewed load
        row_src, te, ts, tend, n = plan_tiles(ids, 128, bm)
        flat = ids.reshape(-1).long()
        assert sorted(row_src.tolist()) == list(range(t * 8))  # a permutation
        counts = torch.bincount(flat, minlength=128)
        starts = torch.cumsum(counts, 0) - counts
        covered = torch.zeros(t * 8, dtype=torch.int32)
        for i in range(n):
            s, e, ex = int(ts[i]), int(tend[i]), int(te[i])
            if s >= e:
                continue
            assert e - s <= bm and s >= int(starts[ex]) and e <= int(starts[ex] + counts[ex])
            assert (flat[row_src[s:e].long()] == ex).all()
            covered[s:e] += 1
        assert (covered == 1).all()


def test_router_gemm():
    torch.manual_seed(1)
    x = torch.randn(1000, 2304, device="cuda").to(torch.bfloat16)
    w = torch.randn(128, 2304, device="cuda") * 0.02
    bias = torch.randn(128, device="cuda") * 0.01
    ids = torch.empty(1000, 8, dtype=torch.int32, device="cuda")
    wts = torch.empty(1000, 8, dtype=torch.float32, device="cuda")
    fk.route(x, w.to(torch.bfloat16), bias, None, ids, wts, 8, 8.0, w_f32=w)
    ref = (x.double() @ w.double().t())
    fp32 = x.float() @ w.t()
    ids_ref = torch.empty_like(ids)
    wts_ref = torch.empty_like(wts)
    fk._topk_kernel[(1000,)](fp32, fp32.stride(0), bias, ids_ref, wts_ref, E=128, TOPK=8, scale=8.0, num_warps=4)
    logits = torch.empty(1000, 128, device="cuda")
    for prec in ("tf32x3", "ieee"):
        fk._router_gemm_kernel[(16,)](x, x.stride(0), w, logits, 1000, K=2304, E=128, BM=64, BK=32, num_warps=8,
                                      PREC=prec)
        print(prec, "err", (logits.double() - ref).abs().max().item())
    fk._router_gemm_kernel[(16,)](x, x.stride(0), w, logits, 1000, K=2304, E=128, BM=64, BK=32, num_warps=8)
    err = (logits.double() - ref).abs().max().item()
    err32 = (fp32.double() - ref).abs().max().item()
    print("router max err", err, "cuBLAS fp32", err32)
    assert err < 4 * err32 + 1e-6
    agree = (torch.sort(ids, 1).values == torch.sort(ids_ref, 1).values).all(1).float().mean().item()
    print("top-8 set agreement", agree)
    assert agree > 0.995


def _msg_eager(hs, ms, lw, gate0, n_gate, cfg, norm_w):
    eps = cfg.rms_eps
    outs = []
    for j in range(n_gate):
        h, m = hs[j], ms[j]
        xn = (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        mn = (m * torch.rsqrt(m.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        g = (torch.cat([xn, mn], -1) @ lw.ctl_down[gate0 + j].t()).float()
        f = torch.tanh(g @ lw.ctl_forget[gate0 + j].float().t())
        ga = torch.tanh(g @ lw.ctl_gain[gate0 + j].float().t())
        o = (1.0 - f) * h + ga * m
        if norm_w is not None:
            o = (o * torch.rsqrt(o.square().mean(-1, keepdim=True) + eps) * norm_w[j].float()).to(torch.bfloat16)
        outs.append(o)
    return torch.stack(outs)


def test_msg_gates_prefill():
    torch.manual_seed(2)
    h, t, r = 2304, 300, 64
    cfg = SimpleNamespace(rms_eps=1e-6, gate_rank=r)
    lw = SimpleNamespace(ctl_down=(torch.randn(4, r, 2 * h, device="cuda") * 0.02).to(torch.bfloat16),
                         ctl_forget=(torch.randn(4, h, r, device="cuda") * 0.1).to(torch.bfloat16),
                         ctl_gain=(torch.randn(4, h, r, device="cuda") * 0.1).to(torch.bfloat16))
    hs = torch.randn(1, t, h, device="cuda").expand(3, -1, -1)  # broadcast attention input, as local layers do
    ms = torch.randn(3, t, h, device="cuda")
    norm_w = (1 + 0.1 * torch.randn(3, h, device="cuda")).to(torch.bfloat16)
    out = torch.empty(3, t, h, dtype=torch.bfloat16, device="cuda")
    tmp = torch.empty(3, t, h, dtype=torch.float32, device="cuda")
    fk.msg_gates(hs, ms, lw, 0, 3, None, out, cfg, norm_w=norm_w, tmp=tmp)
    ref = _msg_eager(hs, ms, lw, 0, 3, cfg, norm_w)
    e1 = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    out32 = torch.empty(2, t, h, dtype=torch.float32, device="cuda")
    fk.msg_gates(hs[:2], ms[:2], lw, 1, 2, None, out32, cfg)
    ref32 = _msg_eager(hs[:2], ms[:2], lw, 1, 2, cfg, None)
    e2 = ((out32 - ref32).norm() / ref32.norm()).item()
    print("msg gates rel err (normed bf16, fp32)", e1, e2)
    assert e1 < 4e-3 and e2 < 5e-4  # bf16 rounding flips of the normed inputs feed g


def test_fp8_mm():
    from inkling_engine.ops.gemv_triton import Fp8Weight, fp8_mm

    torch.manual_seed(3)
    x = torch.randn(1000, 2304, device="cuda").to(torch.bfloat16)
    w = (torch.randn(6592, 2304, device="cuda") * 0.02).to(torch.bfloat16)
    w8 = Fp8Weight(w)
    out = torch.empty(1000, 6592, dtype=torch.bfloat16, device="cuda")
    fp8_mm(x, w8, out)
    deq = (w8.w8.float() * w8.scale[:, None]).to(torch.bfloat16)
    ref = (x.float() @ deq.float().t())
    err = ((out.float() - ref).norm() / ref.norm()).item()
    seg = torch.empty(1000, 512, dtype=torch.bfloat16, device="cuda")
    fp8_mm(x, w8.rows(1024, 1536), seg)
    err_seg = ((seg.float() - ref[:, 1024:1536]).norm() / ref[:, 1024:1536].norm()).item()
    print("fp8_mm rel err vs dequantized-weight GEMM", err, "segment", err_seg)
    assert err < 0.04 and err_seg < 0.04
