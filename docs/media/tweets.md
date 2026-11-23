# Launch posts (drafts, not posted)

Every number is from `results/serve_*.json` and `docs/benchmarks.md`. Attach a re-rendered `inkling-launch-1x1.mp4` to the
first post: the rendered one still shows withdrawn numbers.

## Thread

**1/**

We built an inference engine for Inkling-2-Mini, the 26B looped MoE, that fits on one 32 GB GPU.

Single-request decode: 322 tok/s, vs 232 for SGLang and 235 for vLLM on the same GPU and the same 32 GB budget
(medians over containers).

Python + Triton, open source. Video below.

github.com/amw720386/inkling-2-mini-inference

**2/**

Inkling-2-Mini runs its middle 12 layers three times, so every token makes 48 layer visits, and each visit picks 8
of 128 experts. That is 3.4B active parameters per token out of 26B.

We fuse each visit into ~12 Triton kernels and replay a whole decode step as one CUDA graph.

**3/**

DSpark speculative decoding: a small drafter reads four of the model's hidden states and drafts 7 tokens in one
pass. The model checks them at width 2, 4 or 8, picking the width with the most expected tokens per millisecond.

Sampling stays exact: rejection sampling, not a quality trade.

**4/**

It fits four 32k-token contexts in 32 GB. Or one 256k context, with an FP8 cache.

GSM8K is within noise of SGLang and vLLM (0.83 vs 0.845 / 0.84, n = 200).

**5/**

4 concurrent requests: 576 tok/s, vs 514 for SGLang and 437 for vLLM. It should carry over well to an RTX 5090 (same
chip, fits in 32 GB), not measured yet.

All numbers are from an RTX PRO 6000 Blackwell, 32 GB budget per engine.

## Single post

Inkling-2-Mini (26B looped MoE) on one 32 GB GPU: faster single-request decode than SGLang and vLLM on the
same card and budget (322 vs 232 / 235 tok/s). Fused Triton kernels, one CUDA graph per step, DSpark speculation. Open source. Video below.

github.com/amw720386/inkling-2-mini-inference
