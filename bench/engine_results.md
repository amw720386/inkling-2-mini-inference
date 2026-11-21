# Engine results (RTX PRO 6000 Blackwell proxy, memory capped to 32 GiB)

These are **not RTX 5090 measurements**. The RTX PRO 6000 uses the same GB202 / sm_120 chip with 188 SMs (the 5090
has 170) and similar GDDR7 bandwidth; every run caps the process at 32 GiB with
`torch.cuda.set_per_process_memory_fraction`. Measured 2026-11-21 on Modal (`RTX-PRO-6000`, driver 610/620,
torch 2.9.1+cu128, Triton 3.5.1).

## Decode (batch 1, greedy, 256 new tokens; `tests/gpu/bench_spec.py`)

| | plain | DSpark | 8k prompt plain | 8k prompt DSpark |
|---|---:|---:|---:|---:|
| bf16 | 153.3 tok/s | 255.6 tok/s | 145.5 tok/s | 240.9 tok/s |
| `dense_fp8` | 172.9 tok/s | 251.7 tok/s | 164.1 tok/s | 239.0 tok/s |

DSpark acceptance length (accepted drafts + 1 per round) ranges 1.9-5.5 by prompt. Fused Triton Markov head (draft 1.91 -> 1.50 ms). Graph replay costs
(`tests/gpu/bench_round.py`, bf16): decode 6.30 ms, draft 1.86 ms, verify m=2/4/8 7.82/9.40/10.22 ms.

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

## Prefill

~13k tok/s on an 8k prompt (2048-token chunks), ~6k tok/s on a 512-token prompt.

## Quality

GSM8K, 200 test questions, 5-shot, greedy, DSpark on (`tests/gpu/gsm8k_eval.py`):

| | accuracy |
|---|---:|
| engine bf16 | 0.835 |
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
