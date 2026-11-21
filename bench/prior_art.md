# Prior art: Inkling 2 Mini inference optimization (as of 2026-11-21)

Inkling 2 Mini came out yesterday, but there is already a lot of work on it. **One project is almost exactly what you are building:** a standalone C++/CUDA engine for Inkling-2-Mini on one RTX 5090 (sm_120a). It is Apache-2.0 and its kernel contracts are written up in detail. Read it first.

How I checked: READMEs, PR bodies, file trees and selected source files read through `gh`. The numbers below are the authors' own claims; I did not reproduce any of them. Rows from the aliteq.com and dev.to roundups were summarized by a fetch tool, so they are lower confidence.

---

## Ranked finds

### 1. ingine: custom C++/CUDA engine, one RTX 5090 (best match by far)
- **URL:** https://github.com/satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0). Ready-to-run weights: https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine. It serves the original model when `INGINE_ABLATION` is empty.
- **What it is:** One C++20/CUDA 13.4 binary (`ingine-serve`) with no Python or PyTorch at runtime. It is built only for `sm_120a` and runs Inkling-2-Mini with an OpenAI-compatible API. It began as a PyTorch engine with CUDA extensions, which was ported to C++ and then deleted (the old code is in the history up to commit 44bc891). The infrastructure comes from `satellitedown/fngine`.
- **Performance (RTX 5090, one request), from the README and commit messages:**
  - **Plain decode:** 306 / 301 tok/s (chat / code).
  - **DSpark decode:** 486-568 tok/s (greedy and T=1). Replays of agent traffic: 502-563 tok/s.
  - **Prefill:** 10,900 tok/s at 20K tokens, 9,600 at 64K, 8,100 at 131K, 7,000 at 200K. TTFT is 0.09 s on a short prompt.
  - **Long context:** decoding after a 131K prompt runs at 354 tok/s, after a 254K prompt at 258 tok/s.
  - **Memory:** 23.8 GiB of VRAM at 262K context.
- **Optimizations that matter for sm_120:**
  - **Decode step:**
    - Each decode step is a fixed sequence of custom kernels captured as **one CUDA graph per call width (M = 1, 2, 4, 8)**.
    - Kernels are chained with **PDL**. Weight loads are issued before `griddepcontrol.wait`, which needs a `pdl_wait_fenced` trick so ptxas does not sink the loads.
    - L2 hints: weights read once per step are loaded **evict-first** (+1-2%). Evict-last on small weights measured slower.
    - Plain decode is **pipelined**: step N+1 is launched before the host reads token N.
  - **Experts:**
    - Routed experts use W4A16: NVFP4 codes are decoded in registers with Blackwell's FP4 conversion instructions, against BF16 activations. MXFP8 experts are handled the same way.
    - Gate and up rows are interleaved in `w13`.
    - The `moe_down` kernel also does the weighted sum over the 8 experts and the post-MLP short conv. The target is at least 80% of DRAM bandwidth.
  - **Dense layers:** dense GEMVs use **FP8 E4M3 weights with per-row scales** (`--dense-fp8`). There is a multi-segment `proj` GEMV with 3 inputs per token, for the separate q/k/v inputs in later loop passes.
  - **Fused glue kernels:**
    - `mix` does residual update + attention residual + RMSNorm, and writes the side outputs (DSpark taps, rollback copies) in the same launch so the PDL chain is not broken.
    - `message_gates` handles the loop gates.
    - The router is a fused sigmoid + top-8 + renormalize kernel.
    - The head fuses the softcap.
  - **Attention:**
    - One fused decode kernel does q/rel clipping, log-scaling tau, static or dynamic K/V conv, K clip, V RMSNorm, the cache write, flash-decoding split-K with relative bias + sink, and the output gate.
    - **KV cache:** FP8 E4M3 with no scale on the 4 full-attention layers (8 visits, 16 KiB per token). Sliding layers use a BF16 1024-slot ring with a `kpos` array.
    - Verify-width full attention reads E4M3 K fragments directly into mma.
  - **DSpark:**
    - The verify width is chosen **on the device** from draft confidences, through CUDA-graph switch nodes for verify, accept/rollback and commit+next-draft.
    - The host synchronizes once per round.
    - The drafter uses a 32,768-token shortlist head (`draft_shortlist.bin`), tensor-core block attention and an E4M3 global ring.
  - **Prefill:** chunked (2048), with its own flash-attention kernel (relative bias in-kernel) and grouped MoE GEMMs.
  - **Serving:** prefix reuse with checkpoints, plus "parking" of idle conversations in spare VRAM.
  - **Testing:** FP64 CPU oracles per kernel, bit-lock hashes, and golden logits from the PyTorch engine.
- **Files to read:**
  - `native/KERNELS.md`: kernel contracts. This is the most useful single document; it specifies the exact model math, including softcaps, clip thresholds, conv order and tau(p).
  - `native/DESIGN.md`
  - `native/src/kernels/moe/moe.cu`, `native/src/kernels/moe/dequant.cu`
  - `native/src/kernels/attn/attn.cu` (68 KB)
  - `native/src/kernels/dense/{gemv.cu,mix.cu,gates.cu,router.cu,cluster.cuh,dsmem.cuh}`
  - `native/src/kernels/common/{pdl.cuh,l2.cuh}`
  - `native/src/kernels/head/head.cu`
  - `native/src/kernels/prefill/{attention.cu,wgemm.cu,moe.cu,rows.cu}`
  - `native/src/kernels/draft/{block.cu,attention.cu,markov.cu,accept.cu}`
  - `native/src/model/{decoder.cpp,runtime.cpp,spec.cpp,drafter.cpp,generator.cpp,weights.cpp,model_data.h}`
  - `native/apps/bench/main.cpp`, `native/apps/validate/main.cpp`, `tools/inkling_ref.py`
- **Related:** https://github.com/skyne98/inkling-ingine-3090 is an sm_86 port of ingine. It reports 131 tok/s plain and 191-247 with DSpark on a 3090, and adds expert offload to RAM. See `PORT-3090.md`, which lists which Blackwell-only features (FP4/FP8 conversion instructions, clusters, PDL) ingine depends on.

### 2. vLLM upstream Inkling 2 / Inkling 2 Mini (reference numerics, sm_120 kernels)
- **URLs:**
  - Model PR (merged 2026-11-20): https://github.com/vllm-project/vllm/pull/69028
  - Blog: https://vllm.ai/blog/2026-11-20-inkling2-mini (source: `vllm-project.github.io/_posts/2026-11-20-inkling2-mini.md`)
  - Recipe: https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini (file `recipes/models/thinkingmachines/Inkling-2-Mini.yaml`)
  - Docker image: `vllm/vllm-openai:inkling-2-mini`
- **Model files:** `vllm/models/inkling2/nvidia/{model.py,loop.py,attention.py,moe.py,sconv.py,dspark.py}` and `vllm/models/inkling2/configs.py`.
- **Ops:** `vllm/models/inkling2/nvidia/ops/`:
  - **Consumer-Blackwell MoE:** `moe_sm120.py`, Triton `tl.dot_scaled` "e2m1" x "e2m1". This is native NVFP4 MMA, so **activations are quantized to NVFP4 (W4A4)** with the reference's "four-or-six" quantization and outer row scales. MXFP8 is dequantized.
  - **Low-M GEMV:** `gemv.py`, Triton tensor-core GEMV for M ≤ 16 on SM12x.
  - **Routing:** `moe.py` does fused route, quantize and scatter.
  - **Other ops:** `attention.py` (FA4 relative attention, gate/sink, log-scaling tau, split planner), `attn_res.py`, `conv_mix.py`, `sconv.py`, `loop.py`, `nvfp4.py`, `mxfp8.py`, `block_scales.py`, `dspark.py`, `trtllm_bmm.py`/`.cu`.
- **CUDA sources:** `csrc/inkling2/{conv_mix.cu,allreduce_conv_mix.cu,seq_parallel.cu}`. The fused all-reduce + short conv + residual + attention-residual mix + RMSNorm kernel is built only for Hopper and SM10x.
- **Numerics:** vLLM claims bitwise MoE parity with the reference given the same inputs. It uses the same NVFP4/MXFP8 activation quantization as the reference, and the FP32 message merges run without FMA contraction. **ingine (find 1) instead uses BF16 activations into the experts.** Decide which you match.
- **sm_120 PRs:**
  - **#69168** (open, by gau-nernst), "[Perf][Inkling2-Mini] Improve small M GEMM on 5090 (sm120)": adds `ops/ll_gemm_sm120.py` (`sm12x_bf16_ll_gemm`).
    - Deterministic split-K for narrow N: short-conv projection N=32, up to 7.1x.
    - Batched GEMM for the loop blend: up to 5.1x.
    - Fused q/k/v from three inputs: 1.19x, 25.8 µs = 1.54 TB/s at 8608x2304.
    - **End to end on an RTX 5090 with DSpark (7 tokens):** c1 mean ITL 10.70 → 9.89 ms, **322.7 → 347.6 tok/s**, acceptance 3.53. c2: 424-459 tok/s.
  - **#69021** (merged): keeps the TMA `k_group_partials` path on SM12x with 64-wide N tiles and 2 stages, so it fits in 101,376 B of shared memory. 1.25-1.5x faster on GB10.
  - **#69096** (open): replaces `reshape_and_cache_flash` in the DSpark draft KV write. The prebuilt image had no sm_120 SASS for it (issue #69081).
  - **#69165** (open): Ampere/Ada. Shows which features are SM90+ only (PDL, TMA, e4m3 in Triton).
  - **#69104** (open): drops the tile-8 TRT-LLM-gen expert GEMM, which hit an illegal memory access on B200.
  - **#69141** (open): compressed-tensors W4A16/W4A8/W8A8 experts.
- **Workarounds for sm_120 bring-up:**
  - https://github.com/Steven0236/inkling2mini-sm120-speculator-fix: the shipped image had only sm_75 binaries for the drafter KV write. The fix swaps in the vLLM 0.34.0 `_C_stable_libtorch.abi3.so` and replaces FlashInfer's precompiled RMSNorm with torch fp32 math. The same diagnosis is in HF discussion #9.
  - https://github.com/urbanspr1nter/inkling-2-mini-rtxpro6000-vllm: builds PR #69028 for SM120. Its patch `0001-inkling2-sm120-runtime-max-ctas.patch` stops the MoE kernels from recompiling for every new prompt length.
- **Other numbers:**
  - GB200 TP1 forward pass: decode step at batch 1 is 3.85 ms with 3 passes, 2.67 ms with 2, 1.51 ms with 1. Verify for 1 request at 51K context takes 5.87 ms; an 8K prefill takes 116 ms.
  - RTX 5090: 416.6 tok/s on a code prompt with the official image and DSpark (X post by @niklaslenz_ai, via the aliteq.com roundup).

### 3. SGLang Inkling-2-Mini, plus the DGX Spark speed-overlay repo
- **URLs:**
  - Model PR: https://github.com/sgl-project/sglang/pull/51926
  - Cookbook: `docs/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini.mdx`. Its recipe config `docs/src/snippets/configs/thinkingmachines/inkling-2-mini.jsx` includes an **`rtx5090` recipe**.
  - Docker image: `lmsysorg/sglang:inkling-2-mini`
- **sm_120 kernels:**
  - `python/sglang/kernels/jit/csrc/inkling2/routed_quant_sm120.cuh`
  - `python/sglang/kernels/ops/gemm/inkling2_grouped_sm120.py`: native NVFP4/MXFP8 grouped GEMMs
  - `python/sglang/kernels/ops/moe/inkling2_experts_sm120.py`
  - `python/sglang/kernels/ops/gemm/sm12x_bf16_gemv.py`: small-M decode GEMV, also used for the FP32 LM head
  - `python/sglang/kernels/ops/quantization/inkling2_mxfp8_scales_sm120.py`
  - `python/sglang/kernels/ops/elementwise/inkling2_loop.py`: fused loop message controllers
  - Older: PR #45420 fits small-M grouped GEMM into SM120 shared memory.
- **Other Inkling2 kernels:**
  - JIT CUDA: `kernels/jit/csrc/inkling2/{prologue,gate_sink,route,routed_rows,sconv,exchange}.cuh`
  - Ops: `kernels/ops/attention/inkling2_attn*.py`, `layernorm/inkling2_qk_l2_clip.py`, `mamba/inkling2_sconv.py`
- **RTX 5090 recipe:** TP1, FA4 attention, context capped at 65,536 (Balanced) or 32,768 (DSpark), 8 draft tokens, `--max-running-requests 4`. It turns on `SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1`, `SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1`, `..._PREPARE_INPUTS_ONCE=1` and `..._SKIP_PREFILLED_INPUTS=1`. GSM8K 93.6-94.2%. **No tok/s is published for the 5090.**
- **RTX PRO 6000 (sm_120), from HF discussion #9 by adcostanza:** 231.3 tok/s with DSpark and 159.5 tok/s plain at c1, prefill included, accept length 2.71. FP8 KV with FA4 failed during graph capture; BF16 KV works. Details: https://github.com/adcostanza/benchmarks/tree/main/inkling2-mini-vs-qwen38-27b
- **https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark** (GB10 / sm_121, the closest relative of sm_120):
  - Ten import-time overlays on SGLang, all with unchanged outputs.
  - **Decode:** speedbench c1 40.6 → 61.7 tok/s.
  - **Cold prefill:** 8K tokens 3,399 → 5,293 tok/s, 128K tokens 2,422 → 3,726 tok/s.
  - The README tabulates the ideas, with switches and measured effects. The ones worth stealing:
    - A calibrated **verify-width policy**: verify only the first L draft rows, where the row's survival probability pays for its marginal expert reads. Dead rows reuse the anchor row's experts.
    - A **hot-vocabulary draft head** (65,536 tokens, 98.8% coverage).
    - FP8/MXFP8 copies of the draft head and draft layers.
    - A **certified greedy LM head**: a 6-bit screen plus an error bound, then an exact GEMV only on candidate tiles. Head time 2.90 → 1.67 ms.
    - "pico12": an exactly coded 12-bit BF16 weight GEMV (vendored from https://github.com/blaesus/dsgb10, MIT).
    - A NVFP4 tile table for decode: `tile_k` 256 on GB10, vs 128 in SGLang's RTX 5090 table.
    - **Prefill attention with the relative bias computed in-kernel** (FA2-style mma.sync).
    - Prefill fusion of loop-message mix, attention-residual mix + RMSNorm, and short conv + residual.
    - Stacked qkvr computed per source column block.
    - Quantize the MoE input to NVFP4 once, then gather.
  - Their profile puts c1 decode at ~13.4 GB per step: dense BF16 2.8 GB, head 0.93 GB, draft 1.65 GB, experts ~6.5 GB.
  - **Files:** `overlay/*/README.md`, `docs/profile/PROFILE.md`, `docs/prefill/PREFILL.md`, `docs/experiments.md` (includes rejected ideas).

### 4. maya-ik2 (Project Maya / Strata family): standalone CUDA engine, RTX 3090
- **URL:** https://github.com/alesha-pro/maya-inkling-mini (MIT). Upstream: https://github.com/mw00/project-maya PR #1095.
- **Performance (RTX 3090 at 300 W, Unsloth UD-Q4_K_XL, FP8 KV):**
  - Plain decode: 124-126 tok/s.
  - With DSpark: 131-212 tok/s.
  - Prefill: 2,742 tok/s at 4K, 2,079 at 128K.
  - llama.cpp on the same card: 57 tok/s.
- **Ideas:**
  - llama.cpp needs about 7,350 kernel launches per token; this engine uses **757 launches replayed as one CUDA graph**.
  - The NVFP4 experts are decoded in registers and fed to the tensor cores (745 GB/s). Attention reads the KV cache at up to 846 GB/s.
  - Verify-window experts are **grouped on the GPU** so each expert is read once for all tokens that picked it: 5 tokens cost 15.6 ms vs 7.8 ms for 1.
  - The engine drafts only when drafting pays.
- **Files:** `docs/INKLING2-MINI.md` (with "how it works" and "what didn't work" sections), `ref/inkling2.py` (float64 reference).

### 5. TokenSpeed (LightSeek Foundation)
- **What it is:** https://github.com/lightseekorg/tokenspeed (~2.9K stars). An inference engine for agentic workloads: a C++ finite-state-machine scheduler plus a Python execution plane, with a pluggable kernel registry. It is used mostly on GB200/GB300/B300 and MI355X at TP4+. Blog: https://lightseek.org/blog/tokenspeed-inkling-2.html (Inkling 2 Small, day 0). There is no Inkling-2-Mini blog post yet.
- **Inkling 2 code:**
  - Models: `python/tokenspeed/runtime/models/{inkling2.py,inkling2_loop.py,inkling2_attention.py,inkling2_monokernel.py,inkling2_mtp.py}`
  - Drafter: `runtime/execution/drafter/inkling2.py`
  - Layers: `runtime/layers/moe/inkling2.py`, `runtime/layers/quantization/inkling2.py`, `runtime/layers/attention/backends/specific/inkling.py`, `runtime/layers/attention/kv_cache/hybrid_inkling.py`
  - Kernels: `tokenspeed-kernel/.../ops/moe/_triton/inkling_topk.py`, `ops/conv/triton.py`
- **The Inkling2 "decode monokernel" is AMD gfx950 only, at TP4/EP4.** One persistent launch runs a whole layer, with stages handing values through tagged mailboxes in a scratch buffer. It is bit-identical to the regular path, and a looped core's later passes split into an attention part and an MLP part. Files: `tokenspeed-kernel-amd/.../ops/gfx950/monokernel/inkling2/{layer,plan,attention,experts,mix,conv,route,head,quant}.py` and the design doc `docs/design/monokernel.md`. **There is no NVIDIA or sm_120 monokernel for Inkling2.** The NVIDIA path uses FA4, FlashInfer/CUTLASS and Triton.
- **Inkling 2 Mini recipe** (`docs/recipes/models.md`, "Inkling 2 Mini"): only a 4-GPU AMD command is given. Options: `--inkling2-loop-num-passes`, `--inkling2-loop-share-cache`, and an MXFP8 KV cache (E4M3 with one power-of-two scale per 32 values; 0.70x the bytes of BF16).
- **Usefulness for a single 5090:** design ideas only (persistent per-layer kernels, mailbox/tag scheme, bit-identical discipline). There is no sm_120 code to borrow.

### 6. Xolotl's `xolotl-inkling` engine (RTX 4090)
- **URL:** https://github.com/Indras-Mirror/Xolotl
- **Performance:** decode 113 tok/s, 136 with DSpark. Prefill went 1.2K → 4.2K tok/s at 32K once it got tensor-core prompt attention with the relative bias in-kernel ("ported from Basalt"), plus 2048-row chunks.
- **Files:** `src/program/inkling_fused.cu` (88 KB of fused decode kernels: conv, SiTU, stream-stack), `src/program/inkling_main.cpp`, `src/program/inkling_dspark.cpp`, `docs/INKLING.md`, `docs/BASALT_PORTS.md`, `tools/inkling/parity.py`. The forward graph is ported from llama.cpp PR #32694.

### 7. llama.cpp / ggml and Unsloth GGUFs
- **Merged:**
  - **#32694** "Inkling 2 Mini" (pcuenca): `src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`, `conversion/inkling2.py`
  - #32217: Inkling 2 text + vision
  - #32429: attention-mask fixes
  - #32657 / #32669: `.scale` tensors, i.e. per-expert NVFP4 scales
  - #32704: re-share loop slots
- **Open:**
  - **#32740** "compact bias on FLASH_ATTN_EXT for CUDA/ROCm/Vulkan": without it, the relative bias forces the non-FA path and every intermediate is materialized. Files: `ggml/src/ggml-cuda/fattn-{common,tile,mma-f16,vec}.cuh`, `tests/test-flash-attn-indexed-bias.cpp`.
  - **#32218**: DSpark (`src/models/dspark.cpp`, `common/speculative.cpp`)
  - #32695 (danielhanchen): alternate looped layout
  - #32752: build mask in chunks
- **5090 report:** issue **#32707**. IQ4_NL had very slow prefill and crashed on a 5090, at about 30 tok/s. With the Unsloth fork, UD-Q4_K_XL and DSpark, one comment reports about 80 tok/s.
- **Unsloth:**
  - Weights: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF. UD-Q4_K_XL is 17.63 GB, KLD 0.0107, and keeps the original NVFP4 experts with MXFP8 → Q8_0. There is a `dspark-Inkling-2-Mini-Q8_0.gguf` drafter. UD-Q8_K_XL is the 22.87 GB reference.
  - Docs: https://unsloth.ai/docs/models/inkling-2-mini (no tok/s numbers)
  - Required fork build: `unslothai/llama.cpp` release `b12588-mix-c0c3de8`. Flags: `--spec-type draft-dspark --spec-draft-n-max 3`.
- **Best llama.cpp fork for CUDA ideas:** https://github.com/nahcmon/inkling-fast-llama.cpp, branch `inkling-fast2`.
  - On an RTX 3080 Ti: decode 12 → 77 tok/s, prefill 199 → 1,632 tok/s at 48K.
  - **Banded flash attention** with the relative bias computed in-kernel, working on quantized KV: `ggml/src/ggml-cuda/fattn-banded.cu`
  - **Fused depth mix** (block attention residual): `ggml/src/ggml-cuda/depthmix.cu`, op `GGML_OP_DEPTH_MIX`
  - **Fused SiTU / short conv + state + residual / dynamic K/V conv / row L2 clip**: `ggml/src/ggml-cuda/inkling2.cu`, `inkling.cu`
  - Folded projection weights (`inkling2.fast_fold`)
  - A gate-free NVFP4 MMVQ variant in `mmvq.cu`, which was 50x faster at multiple tokens per expert
  - Docs: `inkling-fast/README.md`
- **Smaller forks:** https://github.com/vene-xif/llama.cpp-inkling2 (long-context memory fix), https://github.com/Trecto34/llama-inkling2-vulkan.

### 8. Ampere ports (ideas only, not sm_120)
- **https://github.com/rafaelkupper/inkling-2-mini-ninfer-3090:** a patched NInfer engine (https://github.com/Neroued/ninfer) running UD-IQ4_XS. 125.9 tok/s on a 3090 with DSpark.
- **https://github.com/agent2x0r/inkling2-sm86-kernels:** vLLM on 2x RTX 3090 at TP2. 114.7 tok/s plain, 166-210 with DSpark, prefill 4.9-5.7K tok/s.
- **https://github.com/snailbrainx/inkling-2-mini-sm80:** sm_80 (CMP 170HX). Decode 105 → 144 tok/s, prefill up to 7.3K tok/s.
- **Software FP8 KV conversion on Ampere:** https://github.com/MKRWW/inkling2-ampere and Hiutaky/vllm branch `ampere-port` (HF discussion #20).

### 9. MLX / Apple (reference math only)
- **mlx-lm PR #2086** "Add Inkling 2 Mini model" (open, stacked on #2061): `mlx_lm/models/inkling2/{layers.py,kernels.py,model.py,checkpoint.py}`. On an M3 Ultra: 28.8 → 32.1 tok/s decode with custom Metal kernels, no DSpark.
- **Other Apple projects:** https://github.com/internet-projects/inkling2-mlx (has DSpark), https://github.com/lucasgelfond/fast-inkling-2-mini (26.8-28.8 tok/s on an M4), https://github.com/zuiho-kai/inkling2-mlx-fastpath, https://github.com/razorback16/inkling-infer.
- **transformers:** `src/transformers/models/inkling2/` is merged. Several projects also cite an "Inkling2Loop" transformers PR as their reference. Open PR #51062 adds native NVFP4/MXFP8 experts on MPS.

### 10. Official Thinking Machines code
- **GitHub org:** https://github.com/thinking-machines-lab. The `github.com/thinkingmachines` org is an **unrelated** Philippine data company.
- **Inference code:** there is **no official standalone inference repo**. The org has `inkling-verify`, an eval harness (DeepSWE, MMMU-Pro, HLE, NIAH, MMAU) for checking implementations; vLLM used it. It also has `batch_invariant_ops`, `tinker` and `tinker-cookbook`.
- **Reference implementations:** the HF transformers modeling code plus the vLLM, SGLang and TokenSpeed integrations, which Thinking Machines co-authored.
- **Announcement:** https://thinkingmachines.ai/news/inkling-2-mini/ lists SGLang, vLLM, Ollama, Unsloth, NVIDIA Dynamo, TokenSpeed and Llama App.

### 11. First-gen Inkling (Inkling / Inkling Small)
- **https://github.com/RightNow-AI/inkling-turbo:** a tile-level **sheared relative-bias** attention kernel that replaces the `score_mod` callback. On H100 decode it is 2.66-2.75x faster than vLLM's served path at 64K, but 1.27x slower on sliding-window prefill. Its notes (`journal/regression-sm120-varlen-illegal-address.md`, `journal/regression-pack-gqa-shear-granularity.md`) cover the shear layout and an sm_120 varlen crash.
- **vLLM:** #51560 serves Inkling on SM12x via a Triton attention fallback; #61413 adds a shared Triton implementation (`inkling/common/`). The Inkling2 attention imports `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`.
- **TokenSpeed:** `runtime/models/inkling.py`, `inkling_nextn.py`.
- None of these are single-GPU consumer work.

---

## 5090 numbers in one place (single request unless noted)

| Engine | Mode | Decode tok/s | Prefill tok/s | Source |
|---|---|---:|---:|---|
| ingine (C++/CUDA, FP8 dense weights, FP8 global KV) | plain | 306 / 301 | 6.2K-10.9K | satellitedown README + commit log |
| ingine | DSpark | 486-568 (agent replay 502-563) | same | same |
| vLLM main → PR #69168 | DSpark (7 tokens) | 322.7 → 347.6 | n/a | vLLM PR #69168 |
| vLLM official image | DSpark, code prompt | 416.6 | n/a | X post via aliteq.com |
| SGLang `rtx5090` recipe | Balanced / DSpark | not published | n/a | SGLang cookbook |
| llama.cpp, ggml-org IQ4_NL | plain | ~30 (prefill very slow) | low | llama.cpp issue #32707 |
| llama.cpp Unsloth fork, UD-Q4_K_XL | DSpark | ~80 | n/a | llama.cpp issue #32707 comment |

For comparison, other sm_120-class hardware:
- RTX PRO 6000 on SGLang: 159.5 tok/s plain, 231.3 with DSpark.
- RTX PRO 5000 on vLLM: 175-215 tok/s (X post, unverified).

## Takeaways for your engine
1. **Start from ingine's kernel contracts** (`native/KERNELS.md`). They give the full decode-time math for every Inkling-2 component on sm_120a, and its numbers (about 300 tok/s plain, 500+ with DSpark) are the bar to beat.
2. **Decide the expert numerics up front:**
   - The reference, vLLM and SGLang quantize activations to NVFP4, so they can use native `mma` e2m1×e2m1 (`tl.dot_scaled`).
   - ingine and maya use W4A16, decoding FP4 in registers.
   - At M=1 both approaches are bandwidth-bound. W4A4 matters for prefill and wide verify.
3. **Decode:**
   - Use one CUDA graph per verify width, with PDL chaining and device-side verify-width selection.
   - Use FP8 for the dense weights (2.8 GB BF16 per step otherwise) and for the full-attention KV cache.
   - Use a shortlist or hot-vocab draft head, and consider a certified or screened LM head (0.93 GB BF16 per step).
4. **Prefill:** compute the relative bias inside the flash-attention kernel instead of materializing it (nahcmon, knapcio and Xolotl each measured large gains). Fuse the loop-message mixes, attention-residual mix + RMSNorm, and short conv + residual.
5. **Speculative decoding:** use a calibrated verify width, group verify rows by expert so each expert is read once, and dedupe the expert union across rows.
