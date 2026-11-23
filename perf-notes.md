# perf/continued notes (RTX PRO 6000 proxy, 32 GiB cap, bf16 unless noted)

Same-day baseline (`bench_spec.py 256`, `fec93b5`): plain 173.9 tok/s, DSpark 325.1, round costs 2/4/8 =
7.62 / 7.90 / 8.69 ms, 8k prefill 26.8k tok/s.

## Kept
- Router top-k: one int64-key max per pick (score bits << 32 | E-1-expert) instead of max + argmin + gather,
  weights re-read from the logit row. Logits unchanged (same 4-warp reduction), selections bitwise identical
  (plain==sglang 59/9/4/35/64/64, spec==plain unchanged). Router launch (`bench_router.py`, 24-layer rotation):
  T=1 5.88 -> 4.51 us, T=32 14.3 -> 12.8 us. Round costs 7.56 / 7.86 / 8.66 ms.

- moe_sum for verify-sized steps (T <= 64): grid (T, H/256) instead of one program per token (elementwise
  sum over the 8 rows, so bit-identical). bench_round verify m=2/4/8: 6.91/7.38/8.31 -> 6.72/7.19/8.07 ms;
  bench_spec DSpark 340.4 tok/s (same-day 325.1), round costs 7.43/7.70/8.47 ms, patterns unchanged.
- Same-day check with the moe_sum change: dense_fp8 plain 209.1 / DSpark 351.3 tok/s (round costs 6.50 / 6.75 /
  7.43 ms); bench_concurrency 2 requests 387 tok/s, 4 requests 424 (main: 382 / 415).
- Grouped-MoE expert sort (verify / batched steps): tl.sort of keys expert * BP + pair instead of the O(P^2)
  rank matrix; integer results equal to the old kernel (bench_route_sort.py, incl. heavy collisions).
  Per launch: P=64 3.31 -> 1.78 us, P=128 8.74 -> 2.16, P=256 (4 slots x width 8) 17.4 -> 3.86.
  bench_concurrency 4 requests 424 -> 432 tok/s; single-stream verify change is below container noise.
- NVFP4 decode MoE GEMV configs pinned from DRAM-cold timings. The autotuner times L2-hot repeats (a layer's 8
  experts fit in L2) and picked 8 warps for w13; cold (bench_moe_cold.py, random experts per step over a
  128-expert layer): w13 T=1 27.8 -> 14.8 us, T=2 33.9 -> 25.4, T=3 46.6 -> 35.0; w2 T=2 21.1 -> 15.7. Same BK
  as the engine's picks (print_configs.py), same bits. bench_round decode 5.61 -> 5.47 ms, verify(2) 6.72 ->
  6.40 ms; bench_spec plain 177.9 tok/s (same-day 173.9), DSpark 341.0, 8k DSpark 317.3; patterns unchanged.
  MXFP8 (layers 20-23) faster configs change bits: left on the autotuner.
- MMA-based GEMVs (dense / seg): every config gives the same bits (99/99 in bench_gemv_cold.py). Pinning
  cold-best configs moved seg_gemv nothing in the engine (28.1 us either way) and wo ~0.4 us; not adopted.


## Concurrency (4 requests)
- Not host-bound: cProfile of the scheduler thread puts nearly all step time in GPU waits (a blocking
  torch.tensor(..., device=) / list-indexing copy and .tolist()); a CUDA trace of a real 4-request run shows the
  GPU busy 91.5% (adaptive) / 94.5% (plain) of the wall time (bench/experiments/sched_idle.py). The earlier
  "6.2 ms GPU per step" came from bench_sched_overhead.py feeding one token to every slot (one expert set);
  with 4 real sequences the bs=4 decode graph is 8.75 ms (prof_decode_batch.py), MoE 47%.
- The width policy's costs were calibrated on identical rows: verify(4,8) 10.8 ms calibrated vs 18.4-19.3 ms
  measured in place. Now: distinct random tokens at calibration, then online EMAs of measured replays
  (CUDA events read after the round's own sync, GpuTimer) for verify / decode / multi-draft.
- Batched rounds use a fixed width per batch size (sched_widths.py, greedy): 4 requests width 4 525-533 tok/s
  vs 449 (2), 447-461 (8), 472 (3), 474 (6), 511-519 confidence pick, 451-528 measured-rate bandit;
  3 requests 4; 2 requests 8. Speculate-vs-plain compares measured tokens per wall ms.
- Verify input built on the device in two ops; no blocking index tensor when every request is greedy.
- Server (serve_bench, concurrency only, standard 2 reps): 4 requests 453.9 (main) -> 496-505 tok/s;
  2 requests 459-473 (main 472.6). SGLang: 545. Over 6 prompt rotations 4 requests range 483-657 tok/s
  (acceptance depends on the prompts), so single runs are noisy.
- Dropped: FP4 x FP4 tensor-core MoE for verify-sized steps (bench_verify_moe_fp4.py: no faster at T=4-32,
  bandwidth-bound), sys.setswitchinterval(0.5 ms) in the server (479.9, worse).
- Batched acceptance equals solo acceptance (batched_accept.py); the 4-request average is pulled down by the
  tail once the fastest requests finish.

## GSM8K check
Greedy and deterministic: head gives 0.825 on 200 again. 600 questions: head 502/600 = 0.837, 0267d27
(before the per-source prefill depth attention) 512/600 = 0.853; on questions 201-600 they differ by 3.
Summation-order noise, not a regression.

## Dropped
- Verify widths (2, 3, 4, 6, 8): round costs 3 = 8.36 ms (> width 4), 6 = 8.41 ms (vs 8.70 for 8); DSpark
  328.8 vs 325.1 tok/s same day, within noise. Width 3 never chosen.
- Router num_warps=1 (and BE 2/8 for T > 8): 5.9 -> 3.9 us per launch, DSpark 333 tok/s, but the different fp32
  reduction order of the logits flips near-tied experts (plain==sglang 19/64 and 8/64 instead of 9 and 35).
- qkv_post with BT tokens per program (rel projection loaded once per BT): 115 -> 190 us (BT=16) / 200 us
  (BT=4) per 2048-token call (INKLING_PDL=0 profile). The loop form is slower; reverted.
- msg_up BC 128 -> 64 for decode: patterns kept, no measurable change (round costs equal, PDL prefetch hides it).
- Prefill profile notes (INKLING_PDL=0; PDL inflates profiled times by the wait): grouped MoE 199 ms of 470,
  attention 82, qkv_post 22, moe_sum 22 (1.35 TB/s on fp32 rows, near the read ceiling), store_kv negligible.
- Decode GEMVs at 1278-1353 GB/s vs a 1535 GB/s read ceiling (bench_readbw.py).
- msg_up num_warps 4 -> 2 (bench_msg.py: bit-identical without the norm tail): decode step 5.61 -> 5.46 ms,
  plain 178.5 tok/s, but the norm launches change bits (plain==sglang 19/64 instead of 9/64). With 2 warps only
  for norm-free launches the patterns hold but the gain is gone (175.6 tok/s). Reverted.
- bench_spec / bench_concurrency greedy DSpark outputs vary run to run: verify widths follow the calibrated
  round timings. bench_spec run-to-run noise is about +-3% on DSpark (332 vs 340 tok/s, same code); use bench_round graph
  timings for small changes.
- Grouped small-M MoE (verify T >= 4): all 78-81 configs bit-equal, engine picks within 3% of cold best
  (bench_grouped_cold.py); nothing to gain.
- FP8 GEMVs (drafter, dense_fp8): configs are not all bit-equal; cold best is 1-1.5 us per call faster
  (dense_fp8 wo 7.9 -> 6.4 us); not pursued.
