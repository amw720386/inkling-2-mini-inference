# Launch posts (drafts, not posted)

Every number is from `results/serve_*.json` and `docs/benchmarks.md`. Attach `inkling-launch-1x1.mp4` to the first post.

## Thread

**1/**

We built an inference engine for Inkling-2-Mini, the 26B looped MoE, that runs on one GPU
in under 32 GB.

One greedy request on 24 MT-Bench prompts: 295 tok/s, vs 194 for SGLang and 212 for vLLM with its optimized flags, on
the same RTX PRO 6000 (medians of two containers; our dense weights are FP8, theirs bf16).

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

GSM8K is within noise of SGLang and vLLM (0.835-0.840 vs 0.845 / 0.840, n = 200).

**5/**

4 requests in flight: 539 tok/s, vs 466 for SGLang and 441 for vLLM. vLLM is level at 2 requests and on one random-token request,
and about 1% ahead on a 30k prompt's first token. An RTX 5090 (same chip, and the engine fits) should run it; not measured yet.

All numbers are from an RTX PRO 6000 Blackwell, whole card, every engine on its own flags.

## Single post

Inkling-2-Mini (26B looped MoE) on one GPU, under 32 GB: faster single-request decode than SGLang and vLLM on the same
RTX PRO 6000 (295 vs 194 / 212 tok/s on 24 MT-Bench prompts). Fused Triton kernels, one CUDA graph per step, DSpark speculation. Open source. Video below.

github.com/amw720386/inkling-2-mini-inference
