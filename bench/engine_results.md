# Engine results (RTX PRO 6000 Blackwell proxy, memory capped to 32 GiB)

These are **not RTX 5090 measurements**. The RTX PRO 6000 uses the same GB202 / sm_120 chip with 188 SMs (the 5090
has 170) and similar GDDR7 bandwidth; every run caps the process at 32 GiB with
`torch.cuda.set_per_process_memory_fraction`. Measured 2026-11-21 on Modal (`RTX-PRO-6000`, driver 610/620,
torch 2.9.1+cu128, Triton 3.5.1).

## Decode (batch 1, greedy, 256 new tokens; `tests/gpu/bench_spec.py`)

| | plain | DSpark | 8k prompt plain | 8k prompt DSpark |
|---|---:|---:|---:|---:|
| bf16 | 160.0 tok/s | 284.6 tok/s | 150.3 tok/s | 249.6 tok/s |
| `dense_fp8` | 188.5 tok/s | 281.0 tok/s | 173.8 tok/s | 272.5 tok/s |

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

19.4k tok/s on an 8k prompt (2048-token chunks, `bench_spec.py`), 12.9k tok/s on 8k random tokens
(`bench_prefill.py`; random tokens spread over more experts). The expert GEMMs run on bf16 tensor cores with fp32
accumulation: full rate on the RTX PRO 6000, about half rate on GeForce (RTX 5090), so expect lower prefill on a 5090.

## Quality

GSM8K, 200 test questions, 5-shot, greedy, DSpark on (`tests/gpu/gsm8k_eval.py`):

| | accuracy |
|---|---:|
| engine bf16 | 0.835 (0.840 re-run after prefill/scheduler/vision/PDL changes) |
| engine `dense_fp8` | 0.825 |
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
