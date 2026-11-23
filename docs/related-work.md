# Related work

Other Inkling-2-Mini engines, and what this engine took from them. The numbers here are the authors' own claims on
their own hardware; they have not been reproduced. Same-GPU measurements of SGLang and vLLM are in
[benchmarks.md](benchmarks.md).

| Engine | Hardware | Decode, 1 request (plain / DSpark) | Prefill | Source |
|---|---|---:|---:|---|
| [ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (C++/CUDA, FP8 dense weights, FP8 global KV) | RTX 5090 | 306 / 486-568 tok/s | 10.9k tok/s at 20k tokens | README and commit log |
| vLLM main + PR #69168 | RTX 5090 | - / 323 → 348 tok/s | - | vllm-project/vllm#69168 |
| vLLM official image | RTX 5090 | - / 417 tok/s (code prompt) | - | social-media post |
| SGLang | RTX PRO 6000 | 160 / 231 tok/s (prefill included) | - | HF discussion #9 |
| [maya-ik2](https://github.com/alesha-pro/maya-inkling-mini) | RTX 3090 | 125 / 131-212 tok/s | 2.7k tok/s at 4k | README |
| [inkling-ingine-3090](https://github.com/skyne98/inkling-ingine-3090) (ingine port) | RTX 3090 | 131 / 191-247 tok/s | - | README |
| [Xolotl](https://github.com/Indras-Mirror/Xolotl) | RTX 4090 | 113 / 136 tok/s | 4.2k tok/s at 32k | README |
| llama.cpp (Unsloth fork, UD-Q4_K_XL) | RTX 5090 | - / ~80 tok/s | - | llama.cpp#32707 |

## What came from where

- **SGLang** (sgl-project/sglang#51926): the model structure and the RTX 5090 serving recipe. The low-M GEMV
  design follows `sm12x_bf16_gemv`.
- **transformers** (`modeling_inkling2_loop.py`): the numerical reference and the NVFP4 / MXFP8 dequantization.
- **ingine**:
  - `dspark.py` is a port of its PyTorch DSpark reference.
  - Its kernel contracts (`native/KERNELS.md`) shaped the fused decode design, the FP8 options and the
    verify-width selection.
  - The drafter's shortlist head and loading weights before the PDL wait follow its design. The shortlist ranking
    here is rebuilt from public text by `tools/build_shortlist.py`.
- **vLLM** (`vllm/models/inkling2/nvidia/ops`): the FP4 prefill path follows its sm_120 expert GEMM. The NVFP4
  activation quantizer in `inkling_engine/ops/moe_fp4.py` is a trimmed port of its `nvfp4_quantize`.

All of these are Apache-2.0.
