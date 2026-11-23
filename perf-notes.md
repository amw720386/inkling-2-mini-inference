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

- Batched rounds (second pass): one host sync per round (width depends only on the batch, so the verify is
  queued right behind the draft), the multi-slot Markov head in one launch per step for all slots (4-slot
  draft 1.46 -> 1.08 ms), conv-state commits for all requests in one gather/scatter per state tensor, width 4
  (width 8 only for a pair with mean recent acceptance >= 1.5: in a 4-request run's tail width 8 lost), always
  speculate unless every request's recent acceptance is < 0.5 (a measured-rate switch to plain lost 3%), every
  prefilling prompt advances one chunk per loop (arrivals start together). sched_idle: GPU idle 8.5% -> 4.0%.
  In process 4 requests 535-548 tok/s; bench_concurrency 432 -> 460 (4), 383 -> 406 (2).
- Seeded requests pick widths from the calibration costs (fixed), not the online ones (reproducible).
- Small-M grouped MoE is bandwidth-bound: 1.38-1.42 TB/s cold at T >= 8 (92% of the read ceiling), hot no faster
  (grouped_hot_cold.py).
- Dropped: ragged per-request widths inside a common verify graph (MoE rows masked via the router): 477-514
  (confidences) / 416-428 (acceptance EMA) vs 508-518 fixed width 4; server token coalescing / batched stream
  decode (510 vs 522, noise); a 0.2-0.5 ms GIL switch interval (no consistent change).
- Server vs in-process, client in its own process (serve_gap.py): 2-4%.

- Server streaming: tokens reach the event loop once per scheduler step (coalesced call_soon_threadsafe) and
  each stream decodes once per batch of tokens. In-container A/B, client in its own process (serve_gap.py):
  507/512/523 -> 523/524/532 tok/s at 4 requests (direct scheduler 538-541). The earlier cross-container
  single run (510 vs 522) was noise.
- 2-request batches take width 8 when their mean recent acceptance is >= 1.0 (482 vs 472 tok/s for 1.5).
- Dropped (second pass): progress balancing (width 2 for requests far ahead, MoE rows masked): 513-524 vs
  532-537; one prefill forward for several fresh short prompts: 491-496 vs 501-518; faster 1-warp router /
  2-warp msg_up for T > 1: also changes short-prompt prefill, plain==sglang 19/64 instead of 9/64.
- SGLang today (same container type): 4 requests 528.5 (runs 411.5 / 528.5), 2 requests 278; its log shows
  accept length 2.7-2.9 at 4 running requests with 8 draft tokens.

- Pre-draft: a batched round launches the next round's multi-slot draft (same requests, new anchors and
  positions) before returning, so the GPU drafts while the host hands out tokens; the next round reuses it when
  nothing changed (a stale pre-draft only writes draft-block KV that the next draft rewrites). Through the
  server, in-container: 516/517/519 -> 532/532/538 tok/s at 4 requests, level with direct (536-538).

- Pre-draft for single-request rounds too (cleared by any prefill, so it never carries over to another
  request): through the server, 1 request, six prompts: 166/320/234/214/489/425 -> 177/333/240/212/528/423 tok/s.
- serve_bench on this branch (bf16): 4 requests 525.8 (runs 525.8 / 525.1), 2 requests 475.2, greedy decode
  317.8, T=1 253.3, GSM8K 0.85; dense_fp8 concurrency: 4 requests 547.2 (528.7 / 547.2), 2 requests 514.9.
  main: 454 / 473 / 280 / 262 (bf16), 474 / 502 (dense_fp8). SGLang: 528-545 / 278-338 / 241 / 201.


## Long context
- Attention, full-attention layers: key blocks whose nearest pair is >= rel_extent (1024) apart and that are
  wholly in the cache run in a loop of their own without the relative-bias gather, masks and chunk K/V loads
  (same block order, values bit-identical: the general loop added a masked 0.0 bias). bench_attn.py (last
  2048-token chunk, global layer): 28k context 6498 -> 4118 us, 8k context 1642 -> 1193 us, bit-equal to the
  previous kernel; local layers unchanged. Prefill bf16 30k 1998 -> 1892 ms (fp4_prefill 1537, both 1425);
  8k unchanged (467). longctx.py: 131k prefill 26.3 -> 15.0 s, decode at 131k plain 112 -> 118, DSpark 123 -> 189.

## Server default: --dense-fp8 --fp4-prefill
The server now runs FP8 dense weights and FP4 prefill experts by default (--no-dense-fp8 / --no-fp4-prefill
give the bf16 checkpoint math; fp4_prefill falls back to off when Triton < 3.6). The engine API defaults are
unchanged (bf16), so the plain==sglang parity checks still cover the bf16 path. Server benchmarks, same GPU:
bf16 profile 532.5 / 474.0 tok/s at 4 / 2 concurrent, greedy 325.5, T=1 259.8, TTFT 1k/4k/8k/30k
66 / 231 / 468 / 1884 ms, GSM8K 0.855; FP8 + FP4 profile 542.6 / 528.5, greedy 385.1, T=1 306.3,
TTFT 49 / 169 / 352 / 1551 ms, GSM8K 0.84. SGLang: 528-545 / 278-338, greedy 241, T=1 201,
TTFT 94 / 222 / 432 / 1670.
Three runs with the new default (serve_ours-default-fp8fp4/-default2/-explicit2): 4 concurrent 540.6 / 533.5 /
542.6, 2 concurrent 521.7 / 517.7 / 527.5, greedy 324 / 316 / 323, T=1 315 / 301 / 310, TTFT 48 / 168 / 341 / 1415.
Greedy prompt 0 swings 195-455 tok/s between runs: serve_bench uses ignore_eos and the engine reaches EOS after
~60 tokens on that prompt, so the rest is post-EOS text whose acceptance depends on the verify widths picked
(first_request.py: the engine alone gives 417-443 tok/s on it, no first-request stall). Same at 8a23f3d and
8d7256c, so it is not a regression.

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
