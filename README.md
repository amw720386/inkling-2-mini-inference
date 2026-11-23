# inkling-2-mini-inference

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Model: Inkling-2-Mini](https://img.shields.io/badge/model-Inkling--2--Mini-6f42c1.svg)](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
[![GPU: RTX PRO 6000 Blackwell](https://img.shields.io/badge/GPU-RTX%20PRO%206000%20Blackwell-76b900.svg)](docs/benchmarks.md)

<p align="center">
  <img src="docs/media/race.gif" width="900"
     alt="Real run, played at 1x: this engine, SGLang and vLLM decode the same prompt on one RTX PRO 6000">
  <br><sub>One GSM8K question on one RTX PRO 6000 under a 32 GB limit, played at 1×. Each engine writes its own answer (517, 536 and 641 tokens);
  an earlier version of this engine.</sub>
</p>

We're building a single-GPU inference engine for
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), the 26B-parameter looped MoE from Thinking
Machines (3.4B active per token), for one RTX PRO 6000 Blackwell, within 32 GB. It is Python and Triton on PyTorch, with an
OpenAI-compatible server. Decode uses fused kernels, CUDA graphs and DSpark speculative decoding. The model math
follows SGLang's implementation.

## Where it stands

Every engine on one RTX PRO 6000 Blackwell with the whole card (no memory cap) and its own recommended setup, at most 4 running requests,
same client, tok/s, medians over two containers. The prompts are 24 MT-Bench questions nothing was tuned on, thinking off, natural stopping with a
512-token limit (6 to 9 of the 24 answers reach it):

| | 1 request | 1 request, T=1 | 2 requests | 4 requests | random 1k/1k, 1 request | random 1k/1k, 4 requests | TTFT 4k | TTFT 30k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| this engine (2) | 295 | 284 | 353 | 539 | 300 | 652 | 168 ms | 1424 ms |
| SGLang (2) | 194 | 189 | 305 | 466 | 192 | 490 | 199 ms | 1585 ms |
| vLLM (2) | 210 | 203 | 327 | 435 | 303 | 404 | 184 ms | 1409 ms |
| vLLM, optimized flags (2) | 212 | 207 | 331 | 441 | 295 | 417 | 193 ms | 1486 ms |

Setups: this engine's server default (FP8 dense weights, FP4 prefill); SGLang's DSpark recipe for this card without its
multimodal, grammar and tool-parser flags; vLLM 0.34 nightly (`0.34.0rc1.dev1133+g647a76866`, includes #69168) with its DSpark recipe, and with the FP8 cache
and FLASHINFER attention of its Inkling-2-Mini pull request [#69168](https://github.com/vllm-project/vllm/pull/69168). "random 1k/1k" uses that pull request's
lengths, but our prompts are independent random ids ([why it isn't comparable with that PR](docs/benchmarks.md)).

- **Ahead:** decode on the MT-Bench prompts for one request (about 1.5x SGLang, 1.4x vLLM) and for four, and time to first token at 4k.
- **Level or behind:** two requests against vLLM (353 vs 331; its containers reach our slower one), random 1k/1k with one request (300 vs 303),
  and a 30k-token prompt's first token (vLLM is about 1% faster).
- Our default uses FP8 dense weights; SGLang and vLLM run theirs in bf16. Fully in bf16 (`--no-dense-fp8 --no-fp4-prefill`, 32 GB cap, one
  container of ours and of vLLM), this engine decodes 264 tok/s on the MT-Bench prompts against 192 and 211, is level with vLLM at two requests, and takes longer to
  the first token than both, which keep FP4 prefill (230 ms vs 223 and 184 at 4k).
- Containers differ by 6-7% for us (12% at two requests) and up to 20% for vLLM, TTFT by about 1%. Read decode gaps below 10% as level.
- **GSM8K** is level with both (0.835-0.840 vs 0.845 and 0.840, n = 200; standard error about 0.026), raw 5-shot completions without
  the chat template.

**32 GB limit.** We developed this engine under a 32 GB budget so it fits cards like the RTX 5090. With every engine held to it, on the
six prompts our scheduler was tuned on, greedy decode is 322 vs 232 (SGLang) and 235 (vLLM's default flags), and 576 vs 514 and 437 tok/s for
four requests. vLLM's optimized flags reach 260 for one request there. The full tables, the tuning caveat and the flags are in [docs/benchmarks.md](docs/benchmarks.md).

  <img src="docs/media/memory.gif" width="720" alt="The 32 GB budget: weights, four 32k KV slots, CUDA graphs; then one 256k FP8 slot">

**RTX 5090:** it should work there. A 5090 has the same GB202 chip and 31.8 GiB of memory, and we tested the engine with only
31.0 GiB left free on the card, from an empty kernel cache. The default and bf16 setups took four 31k-token prompts at once and
the 256k setup a 250k-token prompt, peaking at 28.6 GiB (`tests/gpu/fit_5090.py`). We haven't run it on a 5090, so its speed
there isn't measured: GeForce cards run bf16 prefill math at half the rate, so the FP8 / FP4 paths matter more.

![comparison](docs/img/comparison.png)

Details:

- [docs/benchmarks.md](docs/benchmarks.md): every number, how it was measured, quality and numerics, long context
- [docs/architecture.md](docs/architecture.md): how the engine works
- [docs/serving.md](docs/serving.md): the API, scheduling, flags and profiles
- [docs/experiments.md](docs/experiments.md): what changed the numbers, and what was tried and dropped
- [docs/related-work.md](docs/related-work.md): other Inkling-2-Mini engines and what this one borrows
- [results/](results/): the raw result files behind the tables

## How it works

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/media/decode-step-light.svg">
  <img src="docs/media/decode-step-dark.svg" alt="One decode step: prelude, three passes of the loop core, coda; 8 of 128 experts per visit; DSpark drafts and verify">
</picture>

Every token makes 48 layer visits (prelude, the loop core three times, coda), and each visit runs 8 of 128 experts.
Each visit is about 12 fused Triton kernels, and a whole decode step replays as one CUDA graph. DSpark drafts 7
tokens ahead and the model verifies them in the same kind of graph.

<p align="center">
  <img src="docs/media/loop.gif" width="440" alt="One token through the looped MoE">
  <img src="docs/media/dspark.gif" width="440" alt="DSpark: drafts proposed, verified, accepted or rejected">
</p>

More in [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/amw720386/inkling-2-mini-inference && cd inkling-2-mini-inference
# Linux, NVIDIA Blackwell (sm_120). Weights: hf download thinkingmachines/Inkling-2-Mini --local-dir /path/to/Inkling-2-Mini
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[server]"
pip install triton==3.6.0   # last: torch pins 3.5.1, and the FP4 prefill needs 3.6
python -m inkling_engine.server --model /path/to/Inkling-2-Mini --port 8000
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "reasoning_effort": "low"}'
```

The defaults are 4 concurrent requests, 32k context, DSpark, FP8 dense weights, FP4 prefill and image input. Add
`--no-dense-fp8 --no-fp4-prefill` for the bf16 checkpoint math, or `--kv-fp8 --max-ctx 262144 --slots 1` for long context ([serving.md](docs/serving.md)). The first start compiles
the Triton kernels, which takes about a minute; the kernel settings ship with the engine.

## Reproduce

Everything runs on [Modal](https://modal.com) (an `RTX-PRO-6000`):

```bash
modal run modal/download_weights.py                        # weights into a Modal volume
modal run modal/engine_app.py::pytest_gpu                  # kernel tests
modal run modal/engine_app.py::parity                      # logits vs transformers
modal run modal/serve_bench.py --engine ours --whole-card  # also: sglang, vllm, vllm-fp8kv; --tag 2 for a second container
python tools/results_table.py --whole-card && python tools/plot_comparison.py
```

## Next

- Better draft acceptance at 4 concurrent requests (the expert reads are already close to memory bandwidth)
- Smaller decode-step wins (fused per-visit kernels, verify cost)
- Measurements on an RTX 5090
- Re-run the comparison as SGLang and vLLM update their Inkling support

## Credits

The model math follows [SGLang](https://github.com/sgl-project/sglang) and transformers' `inkling2_loop`.
[ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) supplied the DSpark drafter reference
and several design ideas. [vLLM](https://github.com/vllm-project/vllm) supplied the FP4 prefill path. Details are
in [docs/related-work.md](docs/related-work.md).

Apache-2.0, see [LICENSE](LICENSE).
