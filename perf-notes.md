# perf/continued notes (RTX PRO 6000 proxy, 32 GiB cap, bf16 unless noted)

Same-day baseline (`bench_spec.py 256`, `fec93b5`): plain 173.9 tok/s, DSpark 325.1, round costs 2/4/8 =
7.62 / 7.90 / 8.69 ms, 8k prefill 26.8k tok/s.

## Kept
- Router top-k: one int64-key max per pick (score bits << 32 | E-1-expert) instead of max + argmin + gather,
  weights re-read from the logit row. Logits unchanged (same 4-warp reduction), selections bitwise identical
  (plain==sglang 59/9/4/35/64/64, spec==plain unchanged). Router launch (`bench_router.py`, 24-layer rotation):
  T=1 5.88 -> 4.51 us, T=32 14.3 -> 12.8 us. Round costs 7.56 / 7.86 / 8.66 ms.

## Dropped
- Verify widths (2, 3, 4, 6, 8): round costs 3 = 8.36 ms (> width 4), 6 = 8.41 ms (vs 8.70 for 8); DSpark
  328.8 vs 325.1 tok/s same day, within noise. Width 3 never chosen.
- Router num_warps=1 (and BE 2/8 for T > 8): 5.9 -> 3.9 us per launch, DSpark 333 tok/s, but the different fp32
  reduction order of the logits flips near-tied experts (plain==sglang 19/64 and 8/64 instead of 9 and 35).
