# Benchmarks

**Hardware.** Every number here comes from an NVIDIA RTX PRO 6000 Blackwell on Modal, with the engine held to a
32 GB budget. The RTX PRO 6000 uses the RTX 5090's GB202 / sm_120 chip with 188 SMs (the 5090 has 170) and
similar GDDR7 bandwidth (about 1.5 TB/s measured). These are **proxy numbers, not RTX 5090 measurements.**

One difference matters for prefill: GeForce cards run bf16 tensor-core math with fp32 accumulation at half the PRO
card's rate. Prefill in bf16 should therefore be slower on a real 5090, and the FP8 / FP4 options should gain more.
Decode is memory-bound and should carry over.

## Same GPU, same client: this engine vs vLLM vs SGLang

![comparison](img/comparison.png)

|  | decode, greedy | decode, T=1 | TTFT 1k | TTFT 4k | TTFT 8k | TTFT 30k | 2 / 4 concurrent | GSM8K | peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| this engine | 280 | 262 | 66 ms | 232 ms | 476 ms | 2015 ms | 473 / 454 | 0.825 | 28.7 GiB |
| this engine, `--dense-fp8` | 363 | 325 | 60 ms | 216 ms | 446 ms | 1903 ms | 502 / 474 | 0.830 | 28.5 GiB |
| this engine, `--fp4-prefill` | 265 | 261 | 53 ms | 183 ms | 378 ms | 1660 ms | 470 / 444 | 0.835 | 28.6 GiB |
| vLLM | 242 | 206 | 58 ms | 183 ms | - | - | 161 / 243 | 0.840 | 32.2 GiB |
| SGLang | 241 | 201 | 94 ms | 222 ms | 432 ms | 1670 ms | 338 / 545 | 0.845 | 31.4 GiB |

Each server ran on its own Modal container with the same GPU and a 32 GB budget: our process cap, SGLang's
`--mem-fraction-static 0.232` (its RTX 5090 recipe value) and vLLM's `--gpu-memory-utilization 0.33`. The client
(`bench/serve_bench.py`) sends token-id prompts to `/v1/completions`, so tokenization and chat templates cannot
differ. Measured 2026-11-22: SGLang `lmsysorg/sglang:inkling-2-mini`, vLLM 0.34.0rc1 nightly, this engine with
torch 2.9.1 and Triton 3.6. Raw files: `results/serve_*.json`.

How to read the columns:

- **decode**: one request at a time, the 6 reference prompts, 256 tokens each with `ignore_eos`, DSpark on for
  every engine. The rate is measured from the first to the last streamed token. Continuing past the natural end
  of a prompt lowers acceptance for every engine alike, so these are below the in-process numbers further down.
- **TTFT**: random-token prompts, a fresh prompt each time, so no prefix cache helps.
- **concurrent**: aggregate output tokens per second for 2 or 4 simultaneous 256-token requests.
  Every engine is configured for at most 4 running requests at this memory budget.
- **peak GPU**: the highest `nvidia-smi` reading during the run, including the CUDA context.

**Where this engine loses:**

- **SGLang serves 4 concurrent requests faster** (545 vs 454-474 tok/s), and our throughput drops slightly from 2
  to 4 requests. This is being worked on.
- **Prefill in the default mode** is slower than SGLang beyond 4k tokens (476 vs 432 ms at 8k). SGLang and vLLM
  quantize the expert inputs to FP4. Our default keeps them bf16 (the checkpoint's math); with `--fp4-prefill`
  ours is faster (378 ms at 8k) and level at 30k.

**vLLM** could not hold a 32k context in this budget. It reported 7.3 GiB of KV cache left after weights, which is
about 6.9k tokens, so it ran with `--max-model-len 6144`. Its 8k and 30k TTFT are therefore missing, and its
concurrency numbers look starved by the small cache. On a GPU with more memory it would do better.

## In-process numbers

These scripts drive the engine directly, without HTTP, and requests stop at end of sequence. That is why they
differ from the server numbers above. All runs are on 2026-11-22.

| Measure | bf16 | `--dense-fp8` | Script |
|---|---:|---:|---|
| decode, 1 request, plain | 172 tok/s | 214 tok/s | `bench/bench_spec.py 256` |
| decode, 1 request, DSpark | 324 tok/s | 353 tok/s | same |
| decode after an 8k prompt, plain / DSpark | 164 / 316 tok/s | 197 / 328 tok/s | same |
| decode at temperature 1, DSpark | 262 tok/s | 262 tok/s | same |
| 2 / 4 concurrent requests, adaptive batched DSpark | 382 / 415 tok/s | - | `bench/bench_batched_spec.py` |
| 8k prefill, random tokens | 469 ms | 439 ms | `bench/bench_prefill.py 8192 2048 sweep` |
| 8k prefill with `--fp4-prefill` | 373 ms | 343 ms | same |
| 8-token prompt, time to first token | 10.9 ms | - | `bench/bench_ttft.py` |

**Long context** (`--kv-fp8 --max-ctx 262144 --slots 1`, `tests/gpu/longctx.py`):

| Measure | Value |
|---|---:|
| memory after start / peak | 25.3 / 28.6 GiB |
| prefill of a 131k-token prompt | 26.3 s |
| decode at 131k context, plain / DSpark | 112 / 123 tok/s |

**Memory** (`tests/gpu/memcheck.py`): the default server profile (4 slots × 32k, DSpark, vision) holds 26.4 GiB
after start-up and peaks at 28.8 GiB.

## Quality

GSM8K, 200 test questions, 5-shot, greedy (`tests/gpu/gsm8k_eval.py` in process, `bench/serve_bench.py` through
the server). The standard error at n = 200 is about 0.026, so differences below about 0.05 are noise.

| Configuration | Accuracy |
|---|---:|
| this engine, bf16 (runs on 2026-11-22) | 0.825 - 0.860 |
| this engine, `--dense-fp8` | 0.855 |
| this engine, `--fp4-prefill` | 0.850 |

**Agreement with transformers** (`tests/gpu/parity.py`): KL 1.4e-2 on prefill logits, inside the bf16 noise floor
of this model. Two bf16 evaluations of the same tokens with different batch shapes differ by KL ~5e-2.

**State handling** (KV, rings, conv states, loop passes) is bit-exact against the reference model in fp64
(`tests/gpu/state_exact.py`).

**Numerics of the options** (`tests/gpu/fp4_e2e.py`, 3072 tokens of natural text):

| Option | KL vs bf16 | Next-token NLL |
|---|---:|---:|
| bf16 noise floor | 3e-3 | 0.732 (bf16) |
| `--dense-fp8` | 6e-3 | 0.737 |
| `--fp4-prefill` | 9e-3 | 0.737 |

## Reproduce

```bash
modal run modal/download_weights.py                       # weights into a Modal volume
modal run modal/serve_bench.py --engine ours              # also: ours-fp8, sglang, vllm
modal run modal/serve_bench.py --engine ours --extra "--fp4-prefill" --tag fp4
python tools/results_table.py                             # the table above, from results/
python tools/plot_comparison.py                           # docs/img/comparison.png
modal run modal/engine_app.py::parity --args "bench_spec.py 256"
```

SGLang runs the `lmsysorg/sglang:inkling-2-mini` image with its RTX 5090 DSpark recipe. vLLM runs the
`vllm/vllm-openai:nightly` image with its Inkling-2-Mini recipe. The exact flags are in `modal/serve_bench.py`.
Raw results are in `results/`; `results/sglang/` holds the earlier SGLang cookbook runs.
