# SGLang baseline: Inkling-2-Mini

**Every number here comes from an RTX PRO 6000 (sm_120) proxy, memory-capped to 32 GB. None of them are RTX 5090 results.**

The RTX PRO 6000 Blackwell uses the same GB202 / sm_120 die as the RTX 5090, but it has 96 GB of memory and different clocks, power limits and memory bandwidth (1.8 TB/s on both cards, but the SM counts differ: 188 vs 170). Absolute numbers will not carry over 1:1. Use them as a reference point for our own engine, measured under the same proxy setup.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, compute capability 12.0, driver 610.57.04 (Modal `gpu="RTX-PRO-6000"`) |
| Image | `lmsysorg/sglang:inkling-2-mini`, index digest `sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de` (in-image `SGLANG_IMAGE_TAG=lmsysorg/sglang:nightly-dev-20261120-03fde9cc`, `SGLANG_BUILD_COMMIT=03fde9cc392a5f84a54a365635ffed0061cdf1b8`) |
| Software | SGLang `0.0.0.dev1+g03fde9cc3.d20261120`, torch 2.14.1+cu130, CUDA 13.0, Python 3.12.16 |
| Model | `thinkingmachines/Inkling-2-Mini`, HF revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`, 20.19 GB on disk (mixed NVFP4 / MXFP8, `quant_method: inkling2`), in Modal Volume `inkling2-mini-weights` at `/Inkling-2-Mini` |
| Weight memory (from the SGLang log) | target 18.47 GB, DSpark MTP draft +1.27 GB |
| Scripts | `modal/download_weights.py`, `modal/sglang_baseline.py` (`modal/inspect_image.py` is a CPU-only probe) |

### Server commands

The settings are the cookbook RTX 5090 flags with `--model-path` pointing at the volume. I added `--mem-fraction-static` to apply the 32 GB cap.

Balanced:
```
sglang serve --model-path /weights/Inkling-2-Mini --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 \
  --page-size 128 --grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling \
  --tool-call-parser inkling2 --host 0.0.0.0 --port 30000 --context-length 65536 --mem-fraction-static 0.22
```

DSpark:
```
SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 \
SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 \
sglang serve --model-path /weights/Inkling-2-Mini --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 \
  --page-size 128 --grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling \
  --tool-call-parser inkling2 --host 0.0.0.0 --port 30000 --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 8 --context-length 32768 --max-running-requests 4 --mem-fraction-static 0.232
```

To reproduce, run these from the repo root:
```
modal run modal/download_weights.py
modal run modal/sglang_baseline.py --mode full  --config balanced --mem-frac 0.22   # reference outputs, decode, prefill
modal run modal/sglang_baseline.py --mode extra --config balanced --mem-frac 0.22   # bench_serving + gsm8k
modal run modal/sglang_baseline.py --mode full  --config dspark   --mem-frac 0.232
modal run modal/sglang_baseline.py --mode prefill31k --config dspark --mem-frac 0.232
```
Raw JSON for every run is in `bench/raw/` and also in the volume under `/results/`, with full server logs next to it.

## Memory cap (32 GB emulation)

`--mem-fraction-static` only caps weights plus the KV/state pools. CUDA graphs and activations come on top of that. The largest extra is the prefill CUDA graph, which takes about 7.3 to 7.5 GB with this image. I lowered the fraction until nvidia-smi showed 31 GiB or less. Memory was sampled with `nvidia-smi --query-gpu=memory.used -lms 200` for the whole container lifetime.

| Config | mem-fraction-static | nvidia-smi peak | Steady state while benchmarking | KV pool (SGLang log) |
|---|---|---|---|---|
| Balanced | 0.22 | **30905 MiB** (full run), 31152 MiB (extra run); the peak happens during startup and CUDA-graph capture | 29469 to 29716 MiB | 1.93 GB: full-attention 42112 tokens, SWA 4096 tokens, `max_running_requests` capped to 16 by the mamba/state cache |
| DSpark | 0.232 | **31701 MiB** (30.96 GiB), during startup | 30461 to 30467 MiB | 1.46 + 0.15 GB (draft): full-attention 31744 tokens, SWA 3072 tokens, max_running 4 |
| Balanced, uncapped probe | 0.26 | 35009 MiB | n/a | 5.38 GB: 117120 / 11648 tokens |

These capacity limits come from the cap and would also hold on a real 32 GB card:
- **Balanced** cannot hold a single 64k request: the pool has 42112 tokens even though `--context-length` is 65536.
- **Concurrency is limited by the SWA pool.** With 4096 SWA tokens and a 1023-token window per request, only about 4 to 6 requests decode at once. That is why c16 throughput matches c4 below.
- **DSpark** admits at most **31737 input tokens**. A 32000-token prompt is rejected with "Input length (32000 tokens) exceeds the maximum allowed length (31737 tokens)". For DSpark, "32k" prefill was therefore measured at 31000 tokens.
- **DSpark at 0.22 hangs** (first attempt; log in the volume at `results/server_full_dspark_20261121-211018.log`). The pool was only 13568 full / **1280 SWA** tokens, with max_running_requests=1. An 8k-token prompt then stalled in chunked prefill: it scheduled one 1024-token chunk and one 128-token chunk, then sat at 0% GPU utilisation for more than 5 minutes. I killed the container and reran at 0.232.

## Results (RTX PRO 6000 (sm_120) proxy, memory-capped to 32 GB)

### (a) Single-stream decode, batch 1, greedy, 512 output tokens (`ignore_eos`)

The prompts use the chat template with `reasoning_effort="none"`, which renders as "Thinking effort level: 0". Each cell is the median of 3 runs after a warmup. Decode tok/s is (N-1) / (t_last - t_first_token), measured client-side over streaming `/generate`.

| Prompt | Balanced decode tok/s | DSpark decode tok/s | DSpark tokens per verify step |
|---|---|---|---|
| short (74 prompt tokens, "write a long essay") | **137.8** (135.6 to 137.8) | **147.4** (146.5 to 148.8) | 1.92 |
| long (8026 prompt tokens, "summarize the document above") | **132.5** | **242.9** | 3.24 |

The 8k decode runs reuse the radix-cached prompt (7936 cached tokens), so they measure decode only. DSpark's speedup depends heavily on content. Free-form essay writing gains only about 7%. Summarizing a document that is in context gains about 83%, because the drafts copy from the prompt.

### (b) TTFT and prefill throughput, batch 1

The prompts are raw token ids taken from SGLang's `test/long_prompt.txt` tiled to length. Each run starts with a unique prefix so the radix cache cannot hit. I requested `max_new_tokens=1`, so TTFT here is client-side time to the first streamed token. Each cell is the median of 3 runs after 1 warmup.

| Prompt | Balanced TTFT | Balanced prefill tok/s | DSpark TTFT | DSpark prefill tok/s |
|---|---|---|---|---|
| 8192 tokens | **445 ms** | **18,405** | **463 ms** | **17,678** |
| 32000 tokens | **1,878 ms** | **17,039** | rejected (limit 31737) | n/a |
| 31000 tokens (DSpark only) | n/a | n/a | **1,928 ms** | **16,075** |

For the short-prompt decode runs, TTFT was 30 to 33 ms for Balanced and 31 ms for DSpark (74 tokens).

### (c) Serving throughput (`python3 -m sglang.benchmark.serving`)

Settings: `--dataset-name random --random-input-len 1024 --random-output-len 512 --random-range-ratio 1.0 --seed 1`, with `--num-prompts 32` at concurrency 4 and `64` at concurrency 16. Output is fixed length (`ignore_eos`). In this image the module is `sglang.benchmark.serving`; `sglang.bench_serving` does not exist.

| Config | Concurrency | Output tok/s | Total tok/s | Median TTFT | Median TPOT | Median ITL | Accept length |
|---|---|---|---|---|---|---|---|
| Balanced | 4 | **269.4** | 808.1 | 1,496 ms | 10.60 ms | 10.30 ms | n/a |
| Balanced | 16 | **268.1** | 804.4 | 16,262 ms | 15.10 ms | 10.23 ms | n/a |
| DSpark | 4 | **555.5** | 1,666.6 | 1,525 ms | 3.03 ms | 2.32 ms | 4.29 |
| DSpark | 16 | **540.9** | 1,622.6 | 9,704 ms | 3.07 ms | 2.27 ms | 4.76 |

Under the cap neither config scales past roughly 4 concurrent decodes. Balanced is limited by the SWA pool and DSpark by `--max-running-requests 4`, so the extra c16 requests just wait in the queue, which shows up in TTFT. DSpark's accept length on random-token prompts with forced 512-token outputs is probably optimistic: degenerate or repetitive continuations are easy to draft. Treat the DSpark throughput numbers as an upper-end figure.

### gsm8k sanity check (`python3 -m sglang.test.few_shot_gsm8k --num-questions 200`)

Settings: 5-shot, raw completion (no chat template), greedy, max 512 new tokens.

| Config | Accuracy | Invalid | Latency | Output tok/s |
|---|---|---|---|---|
| Balanced | **0.850** | 0.000 | 67.3 s | 526.9 |
| DSpark | **0.830** | 0.000 | 45.2 s | 784.1 |

## Reference outputs for correctness tests

File: `tests/reference/sglang_ref.json` (410 KB). A copy is in the volume under `/results/full_balanced_*.json`, in the `reference` key.

- They come from the **Balanced** server, not speculative decoding. Sampling: `temperature=0`, `max_new_tokens=64`, `ignore_eos=true`, `return_logprob=true`, `top_logprobs_num=20`, `logprob_start_len=0`.
- There are 8 prompts. Four are chat prompts rendered with the HF chat template and `reasoning_effort="none"`; `rendered_prompt` holds the exact string. Four are raw completions, encoded as `tokenizer.encode(text, add_special_tokens=False)`. Every request was sent to `/generate` as `input_ids`.
- Each item stores `input_ids`, `output_ids` (64), `output_logprobs`, `output_top_logprobs` (64 × 20 `[token_id, logprob]`), `input_token_logprobs` and `input_top_logprobs` (top-20 for every prompt position), plus `output_text`.
- Re-running the same 8 requests gave identical token ids (`rerun_identical`: all true).
- Even at effort 0, the chat outputs start with a short plain-text reasoning preamble ("The user asks ..."). This is the model's real greedy behaviour, not an error.
- DSpark greedy outputs do **not** match the non-speculative greedy outputs token for token. Three of the 8 prompts match all 64 tokens. The others diverge at positions 4, 10, 19, 57 and 62, which fits small numeric differences between the batched verify kernel and single-token decode. Use the Balanced file as the reference, and compare with a top-k/logprob tolerance rather than exact ids after the first near-tie.

## Problems hit

1. In this transformers version, `apply_chat_template(tokenize=True)` returns a `BatchEncoding` (a `UserDict`). The first probe passed its keys as token ids and got HTTP 400. Fixed in the script.
2. `sglang.bench_serving` is not in this image; the module is `sglang.benchmark.serving`. `few_shot_gsm8k --host` takes a bare host, and `http://127.0.0.1` fails DNS resolution. Both are fixed; that part was rerun as the `extra` mode.
3. DSpark at mem-fraction 0.22 hangs in chunked prefill with a 1280-token SWA pool (see above).
4. Image startup takes about 4 to 5.5 minutes per container, mostly prefill CUDA-graph capture (120 to 172 s).
