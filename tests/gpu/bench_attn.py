"""Prefill attention microbenchmark with ablations (Modal: parity --args "bench_attn.py").

Times the last 2048-token chunk of an 8k prompt for a local (window 1024, ring) and a global layer, with the
prefill tiles (32 tokens x 4 head slots). Ablations rebuild the kernel
from its source with one piece removed, to see where the time goes (results are wrong, timings are not).
"""

import importlib.util
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import SeqChunk  # noqa: E402
from inkling_engine.ops import attn_triton  # noqa: E402

H, KVH, D, R, W = 24, 8, 128, 1024, 1024
T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048  # new tokens (1 = decode)
START = int(sys.argv[3]) if len(sys.argv) > 3 else 6144  # cached positions before them

BIAS = "        bias = tl.load(Rel + t_glob[:, None] * rel_stride_t + head[:, None] * R + dist, mask=in_rel, other=0.0)\n"
GATHER = """        d0 = start + q0 - (kmin + b * BN)
        jj = tl.arange(0, 2 * BN)
        dcol = d0 - (BN - 1) + jj
        pc = tl.load(Proj + tl.arange(0, 16)[:, None] * R + dcol[None, :], mask=((dcol >= 0) & (dcol < R))[None, :],
                     other=0.0)
        mfull = tl.dot(rr, pc).to(tl.bfloat16).to(tl.float32)
        gidx = (row // 4)[:, None] - tl.arange(0, BN)[None, :] + (BN - 1)
        bias = tl.where(in_rel, tl.gather(mfull, gidx, 1), 0.0)
"""


def _gather(src):
    """Bias from the 16-dim relative states and the layer's [16, R] projection inside the kernel (one small MMA per
    key block, then a per-row shifted gather), instead of a [T, H, R] table."""
    reps = [
        ("    Rel, rel_stride_t,  # [T, H, R]\n", "    Rel, rel_stride_t, Proj,\n"),
        ("rel, rel.stride(0), sinks", "rel, rel.stride(0), proj, sinks"),
        ("meta: AttnMeta, *,", "meta: AttnMeta, *, proj=None,"),
        ("    r = rel.shape[-1]\n", "    r = proj.shape[-1]\n"),
        ("    m_i = tl.full((ROWS,), float(\"-inf\"), tl.float32)\n",
         "    m_i = tl.full((ROWS,), float(\"-inf\"), tl.float32)\n"
         "    rr = tl.load(Rel + t_glob[:, None] * rel_stride_t + head[:, None] * 16 + tl.arange(0, 16)[None, :],\n"
         "                 mask=row_ok[:, None], other=0.0)\n"),
        (BIAS, GATHER),
    ]
    for old, new in reps:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


ABLATIONS = {
    "full": [],
    "gather": _gather,
    "no_bias": [(BIAS.strip(), "bias = tl.zeros((ROWS, BN), tl.bfloat16)")],
}


def variant(name):
    src = open(attn_triton.__file__, encoding="utf-8").read()
    abl = ABLATIONS[name]
    if callable(abl):
        src = abl(src)
    for old, new in ([] if callable(abl) else abl):
        assert old in src, (name, old)
        src = src.replace(old, new)
    src = src.replace("from .pdl import", "from inkling_engine.ops.pdl import")
    path = os.path.join(tempfile.mkdtemp(), f"attn_{name}.py")
    open(path, "w", encoding="utf-8").write(src)
    spec = importlib.util.spec_from_file_location(f"attn_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def timed(fn, reps=20):
    """Per-call time of fn replayed in a CUDA graph (no Python launch overhead, as in the engine)."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


q = torch.randn(T, H, D, device="cuda").to(torch.bfloat16)
kn = torch.randn(T, KVH, D, device="cuda").to(torch.bfloat16)
vn = torch.randn(T, KVH, D, device="cuda").to(torch.bfloat16)
rstate = (torch.randn(T, H, 16, device="cuda") * 0.5).to(torch.bfloat16)
proj = (torch.randn(16, R, device="cuda") * 0.5).to(torch.bfloat16)
rel = (rstate.float() @ proj.float()).to(torch.bfloat16)
qgrs = torch.randn(T, H * D + H, device="cuda").to(torch.bfloat16)
gate, sinks = qgrs[:, :H * D], qgrs[:, H * D:]
seqs = [SeqChunk(0, START, T)]
for kind, cap, ring, window in (("local", W + 64, W + 64, W), ("global", START + T, 0, 0)):
    kc = torch.randn(1, cap, KVH, D, device="cuda").to(torch.bfloat16)
    vc = torch.randn(1, cap, KVH, D, device="cuda").to(torch.bfloat16)
    keys = min(window, START + T) if window else START + T / 2
    flops = 4 * T * H * D * keys
    for name in ABLATIONS:
        mod = variant(name)
        for tok in (32,):
            from inkling_engine.fast import pick_splits  # noqa: E402

            tok = 4 if T <= 4 else (8 if T <= 8 else (16 if T <= 256 else 32))  # as build_step_meta
            nl, ng = pick_splits(START + T, (T + tok - 1) // tok, KVH, torch.cuda.get_device_properties(0).multi_processor_count, W)
            meta = mod.AttnMeta.build(seqs, [0], tok_per_tile=tok, n_split=nl if window else ng)
            if name == "gather":
                fn = lambda: mod.inkling_attention(q, kn, vn, kc, vc, rstate, sinks, gate, meta, window=window,
                                                   ring=ring, scale=D ** -0.5, proj=proj)
            else:
                fn = lambda: mod.inkling_attention(q, kn, vn, kc, vc, rel, sinks, gate, meta, window=window,
                                                   ring=ring, scale=D ** -0.5)
            out = fn().float()
            if name == "full":
                ref_out = out
            elif name == "gather":
                print("  gather vs table rel err", ((out - ref_out).norm() / ref_out.norm()).item())
            ms = timed(fn)
            kv_bytes = 2 * min(keys, START + T) * KVH * D * 2
            print(f"{kind:6s} {name:15s} tok {tok:3d} splits {meta.n_split:2d}: {ms * 1000:7.1f} us  "
                  f"{flops / ms / 1e9:6.1f} TFLOP/s  {kv_bytes / ms / 1e6:6.0f} GB/s K/V")
