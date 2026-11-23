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
- bench_spec run-to-run noise is about +-3% on DSpark (332 vs 340 tok/s, same code); use bench_round graph
  timings for small changes.
