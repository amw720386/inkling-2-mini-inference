# Optimization log

What changed the numbers, and what was tried and dropped. Timings are from the RTX PRO 6000 proxy; the scripts
are in `bench/` and `bench/experiments/`. Each section lists changes in the order they landed.

## Kept

### Decode

| Change | Effect |
|---|---|
| Fused per-token kernels and one CUDA graph per decode step | The engine's base design ([architecture.md](architecture.md)) |
| PDL: each kernel triggers its dependents right after its own wait; GEMVs, router, message gates and attention prefetch weights or cached K/V into L2 before the wait | Decode step 6.08 → 5.66 ms, verify(8) 8.68 → 8.27 ms (`bench/bench_pdl.py`) |
| One-tile depth attention for decode-sized steps | Depth attention 4.4 → 3.7 µs |

| Router top-k picks as one int64-key max instead of max + argmin + gather | Router 5.88 → 4.51 µs per launch, same picks (`bench/bench_router.py`) |
| Verify-sized MoE row sums split over channel blocks | verify(8) 8.31 → 8.07 ms, bit-identical |
| NVFP4 decode GEMV launch configs picked from DRAM-cold timings; the autotuner times L2-hot repeats | w13 at T=1 27.8 → 14.8 µs, decode step 5.61 → 5.47 ms (`bench/bench_moe_cold.py`) |
| Full-attention key blocks far from the queries skip the relative-bias gather | 30k prefill 1998 → 1892 ms (bf16), 131k prefill 26.3 → 15.0 s, DSpark at 131k 123 → 189 tok/s |

An earlier PDL attempt triggered at the start of each kernel: launches cascaded and spun on the SMs (decode
6.40 → 6.81 ms). Triggering only at completion overlapped nothing. Enabling the trigger after the wait also exposed
a kernel registered for PDL without a wait, which raced its producer. Every PDL kernel now waits first.

### Drafter

| Change | Effect |
|---|---|
| Fused drafter glue: residual + RMS norm, q/k norm + rotary, SiLU product | Draft 1.48 → 0.98 ms, with the next item (`bench/prof_draft.py`) |
| Per-KV-head block attention instead of SDPA `enable_gqa` | No 0.8 GB K/V copy at the 64k bucket |
| Shortlist head: candidates from 32k frequent tokens instead of all 200k (ranking built by `tools/build_shortlist.py`; ingine's idea) | Draft 0.97 → 0.63 ms, DSpark +9% |
| Unsorted candidate top-k, bf16 GEMM for the Markov head's fixed projection | Draft 0.63 → 0.58 ms |
| Fused drafter commit: one mix launch and one K-norm / rotary / ring-scatter launch per layer, instead of ~240 kernels | Commit 0.25 → 0.05 ms (`bench/prof_commit.py`) |
| Multi-slot drafts bucketed by context length | 4 slots 5.05 → 2.79 ms |
| Draft at 0.8x the request temperature (exact for any draft distribution) | +2-4% at temperature 1 (`bench/experiments/bench_draft_temp.py`) |

### Concurrent requests

The old 4-request path was slow for two reasons: the verify costs had been calibrated on identical rows (10.8 ms
against 18-19 ms with real ones), so the scheduler picked bad widths, and each round synced with the host several
times.

| Change | Effect |
|---|---|
| Sort-based expert ordering for grouped steps | 256 pairs 17.4 → 3.9 µs per launch (`bench/bench_route_sort.py`) |
| Verify costs measured in place, fixed width per batch size | 4 requests 449 (width 2) / 447-461 (width 8) → 525-533 tok/s at width 4 (`bench/experiments/sched_widths.py`) |
| One host sync per batched round, one Markov launch for all slots, commits in one gather/scatter | GPU idle 8.5 → 4.0% (`bench/experiments/sched_idle.py`) |
| The next round's draft launched before tokens are handed out | 516-519 → 532-538 tok/s at 4 requests, through the server |
| Tokens reach the event loop once per scheduler step | 507-523 → 523-532 tok/s (`bench/experiments/serve_gap.py`) |

### Prefill

| Change | Effect |
|---|---|
| Counting-sort expert tiles (no host sync), fused message gates, 3xTF32 router GEMM | 8k prefill (FP4 mode) 549 → 416 ms |
| Relative-bias load issued before the QK MMA | Attention: local 417 → 338 µs, global 2199 → 1769 µs (`bench/bench_attn.py`) |
| CUDA graphs for chunks of up to 32 tokens | 8-token prompt 20.9 → 10.9 ms (`bench/bench_ttft.py`) |
| bf16 MMAs for the message-gate up projections (both operands bf16: exact products) | 8k prefill 499 → 481 ms |
| Per-source depth attention for prefill-sized steps; the decode tile had spilled registers at prefill sizes | 481 → 469 ms |
| `--fp4-prefill`: NVFP4 activations and `tl.dot_scaled` e2m1 × e2m1 for the NVFP4 experts (needs Triton 3.6) | Per layer, 8k tokens 7.95 → 2.07 ms |
| `--dense-fp8` prefill: FP8 tensor-core GEMMs for the projections | Projection GEMMs ~110 → 44 ms |

## Dropped

| Idea | Result |
|---|---|
| Screened LM head (FP8 scores plus an error bound, exact top-64) | The bound certified only 19% of rows |
| Persistent fused decode MoE (w13, grid barrier, w2 in one launch) | 104-230 µs vs 28.7 µs; too few loads in flight (`bench/experiments/bench_moe_fused.py`) |
| Attention split-combine folded into the attention kernel | Attention 9.7 → 12.3 µs, more than the 1.4 µs combine it removed |
| MoE row sum folded into `post_mlp` for verify | No change |
| One host sync per speculative round (accept, commit and next draft on the device) | Idle 1.07 → 0.82 ms per round, but as much extra GPU work (`bench/experiments/bench_gap.py`) |
| PDL and prefetch in the Markov head | Draft 0.58 → 0.62 ms (each step depends on the last) |
| Pinned staging for graph inputs | No measurable change |
| Head-major attention tiles, cache-only / chunk-only key branches | No gain; the branches hurt software pipelining |
| Relative bias computed in-kernel (MMA + `tl.gather`) instead of a table | Local 362 vs 333 µs, global 2447 vs 1723 µs |
| Router GEMM as 3 × bf16 parts | 1.6x faster, but 30x the fp32 error |
| NVFP4 copies of the dense weights | Simulated KL 1.45e-2, 3x FP8's (`bench/experiments/fp4_dense_sim.py`) |
| Larger PDL prefetch (2 / 4 KiB per row) | Smaller verify gains than 1 KiB |
| FP4 × FP4 tensor-core MoE for verify-sized steps | Bandwidth-bound, no faster at 4-32 rows |
| Per-request verify widths inside one batched graph (MoE rows masked) | 477-514 vs 508-518 tok/s for a fixed width 4 |
| One prefill forward for several fresh short prompts | 491-496 vs 501-518 tok/s |
| Shorter GIL switch interval in the server | 0.5 ms: 480 vs 496-505 tok/s; 0.2 ms: no consistent change |
| Router with 1 warp, message gate with 2 warps | Faster, but another reduction order flips near-tied experts and changes the outputs compared against SGLang's |
| Verify widths 3 and 6 | Within noise of 2, 4, 8 |
| Drafter shortlist of 16k / 64k tokens | 16k loses acceptance; 64k is within noise of 32k |
