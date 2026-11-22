# Engine results (RTX PRO 6000 Blackwell proxy, memory capped to 32 GiB)

These are **not RTX 5090 measurements**. The RTX PRO 6000 uses the same GB202 / sm_120 chip with 188 SMs (the 5090
has 170) and similar GDDR7 bandwidth; every run caps the process at 32 GiB with
`torch.cuda.set_per_process_memory_fraction`. Measured 2026-11-21 on Modal (`RTX-PRO-6000`, driver 610/620,
torch 2.9.1+cu128; Triton 3.5.1, Triton 3.6.0 from 2026-11-22 on).

## Decode (batch 1, greedy, 256 new tokens; `tests/gpu/bench_spec.py`)

| | plain | DSpark | 8k prompt plain | 8k prompt DSpark |
|---|---:|---:|---:|---:|
| bf16 | 161.6 tok/s | 283.4 tok/s | 149.7 tok/s | 250.3 tok/s |
| `dense_fp8` | 183.1 tok/s | 288.8 tok/s | 174.2 tok/s | 265.0 tok/s |

(2026-11-22, Triton 3.6.0; Triton 3.5.1 gave the same within run-to-run variation: bf16 160.0 / 284.6, `dense_fp8`
188.5 / 281.0. Separate Modal machines differ by a few percent in decode.)

DSpark acceptance length (accepted drafts + 1 per round) ranges 1.9-5.5 by prompt. Fused Triton Markov head (draft 1.91 -> 1.50 ms). Graph replay costs
(`tests/gpu/bench_round.py`, bf16): decode 6.30 ms, draft 1.86 ms, verify m=2/4/8 7.82/9.40/10.22 ms.

Sampled decoding at temperature 1.0 (the model's default; same 6 prompts, seeded): plain 159.3 tok/s, DSpark with
speculative rejection sampling 219.7 tok/s (acceptance length 2.34).

Scheduler (`tests/gpu/bench_concurrency.py`, 4 slots, 16k context, DSpark on): 4 concurrent greedy requests of up to 256
tokens, ~300 tok/s aggregate including prefill (SGLang Balanced c4: 269); 2 concurrent: 230 tok/s. Each slot has its own
draft context, so whichever request is left decoding alone speculates.

Batched speculation (`tests/gpu/bench_batched_spec.py`, requests stopping at EOS, up to 384 tokens): with 2
concurrent requests, plain batched 210 tok/s, batched DSpark 248, adaptive 244; with 4, plain 290, batched
DSpark 320, adaptive 321. The drafts of all active requests run as one pass (shared GEMVs; 4 drafts 6.4 -> 3.1
ms), with graphs for every slot set captured at start-up. The scheduler speculates in a batched step only when the
calibrated costs and each request's recent acceptance predict more tokens/ms than a plain batched step.

Batch 4 decode (4 streams, plain): 9.88 ms/step, 405 tok/s aggregate (`tests/gpu/bench_fused.py`).

## Long context (`tests/gpu/longctx.py`, `kv_fp8`)

`kv_fp8` stores the full-attention layers' K/V as E4M3 (16 KiB per token over the 8 global visits; sliding rings
stay bf16), and every read uses the rounded value, so prefill, decode and verify agree. At 4k context the greedy
output equals the bf16 cache's for all 64 tokens checked.

| max_ctx 262,144, `kv_fp8`, DSpark on | |
|---|---:|
| memory after start-up / peak (incl. a 131k prefill) | 25.3 / 28.6 GiB |
| prefill of a 131,072-token prompt | 26.3 s (4,975 tok/s) |
| decode at 131k context, plain / DSpark | 112 / 123 tok/s |

SGLang's RTX 5090 recipe caps the context at 65,536 (Balanced) and 31,737 (DSpark) tokens under 32 GiB.

Bandwidth context: this RTX PRO 6000 reads at ~1.53 TB/s (`tests/gpu/bench_readbw.py`); the decode GEMVs run at
~1.30-1.34 TB/s (87%). A bf16 decode step moves ~5.1 GB (attention projections, experts, LM head), a 3.4 ms floor
against 6.2 ms measured; the rest is the serial chain of small per-token kernels.

Tried and dropped: an exact screened LM head for greedy steps (FP8 scores, error bound ||x||*||w - w8||, exact
recompute of the top-64 rows). On 1,951 real hidden states the rigorous bound certified only 19% of rows, so it
would fall back to the full head most of the time (experiment in git history, commit 0c0e4d5^).

## Memory and robustness

`tests/gpu/memcheck.py` (32 GiB cap): the default server profile (4 slots x 32k context, DSpark, vision) holds
26.3 GiB after start-up and peaks at 28.8 GiB through a 30k-token prefill; the long-context profile (1 slot,
256k, `kv_fp8`, vision) 25.2 / 28.4 GiB (31.6 GiB reserved by the allocator, so run it with `--no-vision` on a
5090 that also drives a display). All CUDA graphs share one memory pool.

`tests/gpu/soak.py`: 60 mixed requests from 6 concurrent clients (chat / completion / tools / 6k-token prompts,
greedy and sampled, streaming with 20% dropped mid-stream): 0 errors, scheduler drains, allocated memory identical
across waves.

## Prefill

`fp4_prefill` (`--fp4-prefill`, needs Triton >= 3.6): multi-token steps quantize the expert inputs of the NVFP4
layers (0-19) to NVFP4 (per-row outer scale, e4m3 scales per 16 values, 4-or-6 block-scale choice, as vLLM's sm_120
path does) and run both expert GEMMs as `tl.dot_scaled` e2m1 x e2m1 on the FP4 tensor cores (the compiled kernel uses
the native block-scaled MMA). Per layer (`tests/gpu/fp4_check.py`, layer 3, random inputs): 2048 tokens 1.88 -> 1.04 ms,
8192 tokens 7.95 -> 2.07 ms. Engine, 8k prompt, same day back to back (`bench_spec.py`): 16.4k -> 21.7k tok/s
before the glue work below, 22.9k -> 28.3k after it. Decode step costs are unchanged, but DSpark accepts slightly
fewer drafts after an fp4 prefill (temperature 1: acceptance 2.03-2.05 vs 2.14-2.32 over three runs each, 186-192 vs
196-219 tok/s; greedy about 5% lower), presumably because the drafter reads prompt hidden states computed with
NVFP4 activations. On the proxy the prefill saving and the decode loss roughly cancel for an 8k prompt with 256 new
tokens; fp4_prefill pays for long prompts with short answers, and more on a GeForce card.

Prefill glue (both paths; 8k random tokens with fp4, `bench_prefill.py ... ops`, 549 -> 416 ms): expert routing
uses a counting sort (histogram, tile plan and scatter kernels, no host sync) instead of `argsort`; the prefill
message gates run as a norm/concat kernel, one cuBLAS GEMM and tiled up-projection/norm kernels instead of ~15
eager fp32 ops per gate; the prefill router GEMM runs on the TF32 tensor cores with 3xTF32 products (max error
1.6e-6 vs 1.2e-6 for cuBLAS fp32 on unit-scale logits, identical top-8 sets in `tests/gpu/test_glue.py`).
GSM8K after these changes (bf16): 0.830.

Prefill attention (`tests/gpu/bench_attn.py`, last 2048-token chunk of an 8k prompt): the per-token relative-bias
table lookup (one bf16 gather per query/key pair) cost about half of the kernel. Issuing that load before the QK
MMA lets its latency overlap the dot: local layer 417 -> 338 us, global 2199 -> 1769 us (without the bias at all:
215 / 1398 us). Head-major tiles (no padded GQA rows) and branching on cache-only / chunk-only key blocks were
tried and dropped (no gain; the branches hurt software pipelining). Computing the bias inside the kernel from
the 16-dim relative states (a small MMA against the layer's [16, R] projection per key block, then a per-row shifted
`tl.gather`) is exact to 3e-5 but slower than the table (local 362 vs 333 us, global 2447 vs 1723 us), so the table
stays. 8k engine prefill, bf16: 23.7k tok/s. The bf16 path pays the GeForce half-rate bf16 MMA; the FP4 path does not, so its
gain on an RTX 5090 should be larger than on this proxy.

Numerics (`tests/gpu/fp4_e2e.py`, 3072 tokens of natural text, all logits): bf16 with 1024- vs 512-token chunks
(the bf16 noise floor) top-1 0.984, KL 2.7e-3; fp4 vs bf16 top-1 0.972, KL 8.2e-3; next-token NLL 0.730 (bf16) vs
0.738 (fp4). GSM8K with fp4 prefill: 0.850 (below). Off by default because it departs from the bf16 checkpoint math.

22.9k tok/s on an 8k prompt (2048-token chunks, `bench_spec.py`; 19.4k before the glue work above),
28.3k with `fp4_prefill`. Random tokens spread over more experts and run slower (`bench_prefill.py`). The expert GEMMs run on bf16 tensor cores with fp32
accumulation: full rate on the RTX PRO 6000, about half rate on GeForce (RTX 5090), so expect lower prefill on a 5090.

## Quality

GSM8K, 200 test questions, 5-shot, greedy, DSpark on (`tests/gpu/gsm8k_eval.py`):

| | accuracy |
|---|---:|
| engine bf16 | 0.835 (0.840 re-run after prefill/scheduler/vision/PDL changes) |
| engine `dense_fp8` | 0.825 |
| engine bf16 + `fp4_prefill` | 0.850 |
| SGLang Balanced (baseline) | 0.850 |
| SGLang DSpark (baseline) | 0.830 |

Standard error at n=200 is about 0.026, so these are indistinguishable.

Engine vs transformers (`tests/gpu/parity.py`): KL 1.5e-2 on prefill logits, inside the bf16 noise floor
(two bf16 evaluations of the same tokens with different batch shapes differ by KL ~5e-2 on this model). State
handling is bit-exact in fp64 (`tests/gpu/state_exact.py`).

## Comparison with SGLang (same GPU, cookbook RTX 5090 recipes; `bench/baseline_sglang.md`)

| | SGLang Balanced | SGLang DSpark | engine bf16 | engine `dense_fp8` |
|---|---:|---:|---:|---:|
| decode, short prompt | 137.8 | 147.4 | 149.9 / 235.3 (DSpark) | 172.9 / 251.7 (DSpark) |
| decode, 8k prompt | 132.5 | 242.9* | 144.3 / 227.6 (DSpark) | 164.1 / 239.0 (DSpark) |
| prefill, 8k | 18.4k | 17.7k | ~13k | ~13k |
| 4 streams aggregate | 269 | 556* | 405 (plain) | - |

\* different prompts: SGLang's 8k DSpark run summarizes an in-context document, and its c4 run uses random tokens
with forced-length outputs (high acceptance).
