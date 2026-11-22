"""FP4 tensor-core prefill experts vs the bf16 grouped path (Modal: parity --args "fp4_check.py").

Checks the NVFP4 activation quantizer, that the dot_scaled GEMM matches the bf16 path fed the dequantized
activations (isolates the GEMM from the quantization error), reports the end-to-end error that activation
quantization adds, whether the compiled kernel uses the native block-scaled MMA, and timings.
"""

import glob
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/root")
from inkling_engine.ops import ref  # noqa: E402
from inkling_engine.ops.moe_fp4 import moe_prefill_fp4, nvfp4_dequantize, nvfp4_quantize  # noqa: E402
from inkling_engine.ops.moe_triton import moe_prefill  # noqa: E402
from inkling_engine.quant import expert_weight_from_checkpoint  # noqa: E402

W = sys.argv[1]


def load_layer(layer):
    pre = f"model.llm.layers.{layer}.mlp."
    t = {}
    with safe_open(f"{W}/model-00001-of-00001.safetensors", framework="pt", device="cuda") as f:
        for n in f.keys():
            if n.startswith(pre):
                t[n] = f.get_tensor(n)
    return (expert_weight_from_checkpoint(t, pre + "experts.w13"), expert_weight_from_checkpoint(t, pre + "experts.w2"),
            t[pre + "gate.weight"].float(), t[pre + "gate.bias"].float())


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


ok = True
torch.manual_seed(0)
x = (torch.randn(4096, 2304, device="cuda") * 0.5).to(torch.bfloat16)
x[:, :7] *= 20  # outlier channels, as real hidden states have
q, s, o = nvfp4_quantize(x)
e = rel(nvfp4_dequantize(q, s, o), x)
print(f"quantizer rel err {e:.4f}")
ok &= e < 0.12

w13, w2, rw, rb = load_layer(3)
for t in (300, 2048):
    xt = x[:t].contiguous()
    wts, ids = ref.route(xt, rw, rb, 8, 8.0)
    fp4 = moe_prefill_fp4(xt, ids, wts, w13, w2, 4.0, 25.0).view(t, 8, -1).sum(1)
    base = moe_prefill(xt, ids, wts, w13, w2, 4.0, 25.0).view(t, 8, -1).sum(1)
    # GEMM-only check for w13: bf16 path on dequantized activations (w2 input differs, so compare end to end)
    xd = nvfp4_dequantize(*nvfp4_quantize(xt)).to(torch.bfloat16)
    deq = moe_prefill(xd, ids, wts, w13, w2, 4.0, 25.0).view(t, 8, -1).sum(1)
    e_q, e_g = rel(fp4, base), rel(fp4, deq)
    print(f"T={t}: fp4 vs bf16 path {e_q:.4f}; fp4 vs bf16-on-dequantized-x {e_g:.4f}")
    ok &= e_q < 0.15 and e_g < e_q

ptx = [p for p in glob.glob(os.environ.get("TRITON_CACHE_DIR", "/root/.triton/cache") + "/**/_grouped_fp4_kernel*.ptx",
                            recursive=True)]
native = any("block_scale" in open(p).read() for p in ptx)
print(f"ptx files {len(ptx)}, native block-scaled mma: {native}")

for t in (2048, 8192):
    xt = torch.randn(t, 2304, device="cuda").to(torch.bfloat16)
    wts, ids = ref.route(xt, rw, rb, 8, 8.0)
    a = timed(lambda: moe_prefill(xt, ids, wts, w13, w2, 4.0, 25.0))
    b = timed(lambda: moe_prefill_fp4(xt, ids, wts, w13, w2, 4.0, 25.0))
    bq = timed(lambda: nvfp4_quantize(xt))
    print(f"T={t}: bf16 grouped {a:.2f} ms, fp4 {b:.2f} ms (x quant {bq:.3f} ms), speedup {a / b:.2f}x")
print("FP4", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
