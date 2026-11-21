"""Triton MoE kernels vs the torch reference on real expert weights (layers 0 and 20)."""

import os

import pytest
import torch
from safetensors import safe_open

from inkling_engine.ops import ref
from inkling_engine.ops.moe_triton import moe_decode, moe_prefill
from inkling_engine.quant import expert_weight_from_checkpoint

W = os.environ.get("WEIGHTS", "/weights/Inkling-2-Mini")


def load_layer(layer):
    pre = f"model.llm.layers.{layer}.mlp."
    t = {}
    with safe_open(f"{W}/model-00001-of-00001.safetensors", framework="pt", device="cuda") as f:
        for n in f.keys():
            if n.startswith(pre):
                t[n] = f.get_tensor(n)
    w13 = expert_weight_from_checkpoint(t, pre + "experts.w13")
    w2 = expert_weight_from_checkpoint(t, pre + "experts.w2")
    return w13, w2, t[pre + "gate.weight"].float(), t[pre + "gate.bias"].float()


@pytest.fixture(scope="module", params=[0, 20])
def layer(request):
    return load_layer(request.param)


def ref_rows(x, ids, wts, w13, w2):
    out = []
    for tok in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[tok, j])
            a = ref.softcapped_swiglu(torch.nn.functional.linear(x[tok:tok + 1], w13.dequant(slice(e, e + 1))[0]), 4.0, 25.0)
            y = torch.nn.functional.linear(a, w2.dequant(slice(e, e + 1))[0]).float() * wts[tok, j]
            out.append(y)
    return torch.cat(out)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("t", [1, 4])
def test_decode_matches_reference(layer, t):
    w13, w2, rw, rb = layer
    torch.manual_seed(t)
    x = (torch.randn(t, 2304, device="cuda") * 0.5).to(torch.bfloat16)
    wts, ids = ref.route(x, rw, rb, 8, 8.0)
    rows = moe_decode(x, ids, wts, w13, w2, 4.0, 25.0)
    exp = ref_rows(x, ids, wts, w13, w2)
    err = rel_err(rows, exp)
    print(w13.fmt, t, "decode rel err", err)
    assert err < 1e-2


@pytest.mark.parametrize("t", [33, 300])
def test_prefill_matches_reference(layer, t):
    w13, w2, rw, rb = layer
    torch.manual_seed(t)
    x = (torch.randn(t, 2304, device="cuda") * 0.5).to(torch.bfloat16)
    wts, ids = ref.route(x, rw, rb, 8, 8.0)
    out = moe_prefill(x, ids, wts, w13, w2, 4.0, 25.0)
    exp = ref.moe(x, wts, ids, w13, w2, 4.0, 25.0)
    err = rel_err(out, exp)
    print(w13.fmt, t, "prefill rel err", err)
    assert err < 1e-2


def test_decode_speed(layer):
    w13, w2, rw, rb = layer
    x = torch.randn(1, 2304, device="cuda").to(torch.bfloat16)
    wts, ids = ref.route(x, rw, rb, 8, 8.0)
    act = torch.empty(8, 1152, dtype=torch.bfloat16, device="cuda")
    rows = torch.empty(8, 2304, dtype=torch.float32, device="cuda")
    for _ in range(3):
        moe_decode(x, ids, wts, w13, w2, 4.0, 25.0, act, rows)
    # 16 disjoint expert sets so weights are mostly not L2-resident across calls
    perm = torch.randperm(128, device="cuda").view(16, 8)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(20):
            moe_decode(x, perm[i % 16].view(1, 8), wts, w13, w2, 4.0, 25.0, act, rows)
    g.replay()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(10):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()
    us = ev0.elapsed_time(ev1) * 1000 / 200
    per_expert = w13.weight[0].numel() * w13.weight.element_size() + w2.weight[0].numel() * w2.weight.element_size()
    per_expert += w13.scale[0].numel() + w2.scale[0].numel()
    gbs = 8 * per_expert / (us * 1e-6) / 1e9
    print(f"{w13.fmt} decode bs1: {us:.1f} us/call, {gbs:.0f} GB/s effective (rotating 16 expert sets)")


@pytest.mark.parametrize("t", [2, 8, 16])
def test_grouped_small_matches_reference(layer, t):
    from inkling_engine.ops.moe_triton import GroupedScratch, moe_grouped_small

    w13, w2, rw, rb = layer
    torch.manual_seed(100 + t)
    x = (torch.randn(t, 2304, device="cuda") * 0.5).to(torch.bfloat16)
    wts, ids = ref.route(x, rw, rb, 8, 8.0)
    sc = GroupedScratch(16, 8, 128, 2304, 1152)
    rows = torch.empty(16 * 8, 2304, dtype=torch.float32, device="cuda")
    out = moe_grouped_small(x, ids.int(), wts, w13, w2, 4.0, 25.0, sc, rows)
    exp = ref_rows(x, ids, wts, w13, w2)
    err = rel_err(out, exp)
    print(w13.fmt, t, "grouped rel err", err)
    assert err < 1e-2
