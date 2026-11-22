# inkling-2-mini-inference

A single-GPU inference engine for [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(26.4B-total / 3.4B-active looped MoE) tuned for the RTX 5090 (sm_120, 32 GB). The model math follows SGLang's
Inkling 2 implementation (PR sgl-project/sglang#51926) and the transformers `inkling2_loop` reference; the decode
path is a small set of fused Triton kernels captured in CUDA graphs, with DSpark speculative decoding.

## Status

| | |
|---|---|
| Correctness | state handling (KV cache, sliding ring, conv states, loop passes) bit-exact in an fp64 check; logits within bf16 noise of transformers; greedy speculative output equal to plain greedy when verify and decode run the same kernels |
| Decode | fused per-token kernels, one CUDA graph per step; NVFP4 experts decoded with the hardware `cvt.e2m1x2`; grouped expert GEMMs for multi-token steps |
| Speculation | DSpark drafter (3 block layers + Markov head), verify widths 2/4/8 picked from measured costs and calibrated draft confidences; greedy (longest matching prefix) and sampled requests (speculative rejection sampling, exact for the target distribution: 220 vs 160 tok/s at temperature 1) |
| Serving | OpenAI-compatible server (chat + completions, streaming, reasoning split, tool calling, image input, sampling, stop sequences, logprobs, presence / frequency / repetition penalties, structured output (`response_format` JSON object / JSON schema via xgrammar), prefix reuse across chat turns in every slot (the scheduler routes a conversation back to the slot holding its history), cancel on disconnect, Prometheus `/metrics`); continuous batching over `--slots` (default 4) concurrent requests with interleaved chunked prefill; speculation for a lone request in any slot and, when it pays, for batched steps (one drafter pass for all slots, slot-independent verify graphs, adaptive choice from calibrated costs) |
| Options | `dense_fp8`: row-scaled FP8 copies of the attention projections and LM head, FP8 GEMMs for their prefill (faster, not bit-faithful to the bf16 checkpoint; GSM8K unchanged); `fp4_prefill`: NVFP4 activations and FP4 tensor cores (`tl.dot_scaled`) for the prefill expert GEMMs, as SGLang / vLLM do on this model (+24% 8k prefill on the proxy, GSM8K unchanged, DSpark acceptance slightly lower) |
| Images | transformers' vision tower + processor; features spliced at the image placeholders. Greedy output on an image prompt equals transformers' for 40/40 tokens (`tests/gpu/vision_check.py`) |
| Long context | `kv_fp8`: E4M3 cache for the full-attention layers; 256k context fits in 32 GiB (25.3 GiB after start-up), 123 tok/s DSpark decode at 131k |

## Performance

**These numbers are from an RTX PRO 6000 Blackwell (same GB202 / sm_120 chip as the RTX 5090, 188 vs 170 SMs,
similar memory bandwidth) with the process capped to 32 GiB. They are a proxy, not RTX 5090 measurements.**
Batch 1, greedy, 256 new tokens, median of short chat/raw prompts unless noted.

| | plain decode | DSpark decode | 8k-prompt decode (plain / DSpark) | prefill (8k) |
|---|---:|---:|---:|---:|
| this engine, bf16 | 159 tok/s | 295 tok/s | 154 / 279 tok/s | ~23k tok/s |
| this engine, `dense_fp8` | 191 tok/s | 311 tok/s | 184 / 271 tok/s | ~25k tok/s |
| this engine, bf16 + `fp4_prefill` | same | ~5% lower** | same / ~5% lower** | ~28k tok/s |
| SGLang (same GPU, cookbook RTX 5090 recipes) | 138 tok/s | 147 tok/s* | 133 / 243 tok/s* | ~18k tok/s |

\*\* DSpark accepts slightly fewer drafts after an fp4 prefill (the drafter reads prompt hidden states computed with
NVFP4 activations), see `bench/engine_results.md`.

\* SGLang's DSpark numbers come from its own benchmark prompts (its 8k figure is a document-summary prompt with high
acceptance), so they are not the same prompts as the rows above.

Concurrent requests through the scheduler (`tests/gpu/bench_batched_spec.py`): 2 requests 301 tok/s, 4 requests
390 tok/s aggregate (SGLang Balanced at concurrency 4: 269). Temperature 1 (the model's default), single stream:
224 tok/s with DSpark. The default server profile (4 slots x 32k context, DSpark, vision) holds 26.4 GiB after
start-up; a 256k-context profile fits too, which SGLang's 32 GB recipe cannot (see `bench/baseline_sglang.md`).
Full numbers: `bench/engine_results.md`.

Prefill caveat: by default the expert GEMMs dequantize to bf16 and run on bf16 tensor cores with fp32 accumulation,
which the RTX PRO 6000 runs at full rate and GeForce cards (RTX 5090) at roughly half rate, so default prefill on a real
5090 is likely noticeably slower than the proxy figure. `--fp4-prefill` moves the NVFP4 expert layers onto the FP4
tensor cores and avoids that. Decode is memory-bound and should carry over.

## Layout

```
inkling_engine/
  config.py        visit plan (prelude, 3 loop passes, coda) from config.json
  weights.py       checkpoint loading (NVFP4 / MXFP8 experts kept quantized)
  cache.py         static per-slot KV (global contiguous, sliding ring with slack) and conv states
  model.py         reference forward (pure torch ops), used to test everything else
  fused_model.py   fast forward: ~12 fused launches per layer visit, CUDA-graph decode
  dspark.py        DSpark drafter
  engine.py        prefill, decode, speculative rounds (single and batched), prefix reuse
  scheduler.py     continuous batching over the slots, interleaved prefill, cancellation
  sampling.py      temperature / top-k / top-p and speculative rejection sampling
  vision.py        image input (vision tower + processor)
  server.py        OpenAI-compatible HTTP server (streaming, tools, images, reasoning split)
  ops/             Triton kernels: attention, MoE (GEMV / grouped / FP4 tensor-core prefill), GEMV (bf16 / FP8), convs, fused glue
tests/             CPU unit tests; tests/gpu/ runs on Modal (parity, kernels, benchmarks, GSM8K, server)
modal/             Modal entry points (weights volume, SGLang baseline, engine tests)
bench/             SGLang baseline results and a survey of other Inkling 2 engines
```

## Running

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install triton==3.6.0   # newer than torch's pin; needed only for --fp4-prefill
pip install safetensors tokenizers transformers fastapi uvicorn pillow xgrammar   # xgrammar: response_format only
python -m inkling_engine.server --model /path/to/Inkling-2-Mini --port 8000            # bf16, DSpark, 4 slots x 32k
python -m inkling_engine.server --model /path/to/Inkling-2-Mini --kv-fp8 --max-ctx 262144 --slots 1 --no-vision
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json'   -d '{"messages": [{"role": "user", "content": "Hello!"}], "reasoning_effort": "low"}'
```

In chat, `response_format` constrains the answer section: the model reasons freely, and from its
`<|content_text|>` marker every token is masked to the JSON grammar or schema until the answer closes.

Requests that ask for `logprobs`, set a penalty or a `response_format` decode without speculation (the logprobs come from the plain decode step's logits:
the model's distribution before temperature / top-p).

Requests without a `temperature` use the checkpoint's `generation_config.json` (sampling at temperature 1.0, the
model's default, as SGLang does); `--default-temperature 0` makes greedy the default instead.

Options: `--dense-fp8` (FP8 attention projections and LM head, faster, not bit-faithful), `--fp4-prefill` (NVFP4
activations on the FP4 tensor cores for prefill expert GEMMs, Triton >= 3.6), `--kv-fp8` (FP8 cache for
the full-attention layers), `--no-spec`, `--slots N`, `--max-ctx N`, `--no-vision`. The first start compiles and
autotunes the Triton kernels (a few minutes); results are cached in Triton's cache directory.

Profiles (numbers from the RTX PRO 6000 proxy, batch 1):

| goal | flags | decode (plain / DSpark) | 8k prefill* | fidelity |
|---|---|---:|---:|---|
| checkpoint math | (none) | 159 / 295 tok/s | 533 ms | bf16 dense layers, as transformers / SGLang |
| speed | `--dense-fp8` | 191 / 311 tok/s | 506 ms | FP8 attention projections and LM head (GSM8K unchanged) |
| long prompts on GeForce | `--dense-fp8 --fp4-prefill` | 189 / 297 tok/s | 368 ms | adds NVFP4 activations for prefill experts (DSpark acceptance a little lower) |
| long context | `--kv-fp8 --max-ctx 262144 --slots 1` | | | FP8 cache for the full-attention layers |

\* 8192 random tokens, 2048-token chunks, all four modes on one machine (`bench_prefill.py 8192 2048 sweep`;
`--fp4-prefill` alone: 395 ms). Natural-text prompts route to fewer experts and run faster (23-28k tok/s).

On a GeForce card (RTX 5090) the bf16 prefill GEMMs run at half the PRO card's rate, so `--dense-fp8` and
`--fp4-prefill` should matter more for prefill there than these proxy numbers show.

GPU tests and benchmarks (Modal, `RTX-PRO-6000`):

```bash
modal run modal/download_weights.py                                   # weights into a Modal volume
modal run modal/engine_app.py::pytest_gpu                             # kernel tests
modal run modal/engine_app.py::parity                                 # engine vs transformers
modal run modal/engine_app.py::parity --args "state_exact.py 1300 1100"
modal run modal/engine_app.py::parity --args "bench_spec.py 256"      # plain vs DSpark
modal run modal/engine_app.py::parity --args "server_e2e.py"
```

## Credits

- SGLang (Apache-2.0): model structure and sm_120 recipe; the low-M GEMV design follows `sm12x_bf16_gemv`.
- vLLM (Apache-2.0): the FP4 prefill path follows its sm_120 Inkling 2 expert GEMM; the NVFP4 activation quantizer
  in `ops/moe_fp4.py` is a trimmed port of its `nvfp4_quantize`.
- transformers (Apache-2.0): `modeling_inkling2_loop.py`, the numerical reference and NVFP4/MXFP8 dequant.
- ingine by satellitedown (Apache-2.0, github.com/satellitedown/inkling-2-mini-abliterated-ingine): `dspark.py` is
  a port of its PyTorch DSpark reference; its kernel contracts (`native/KERNELS.md`) informed the fused decode
  design, FP8 options and verify-width selection.
