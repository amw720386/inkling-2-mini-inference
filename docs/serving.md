# Serving

```bash
python -m inkling_engine.server --model /path/to/Inkling-2-Mini --port 8000
```

This starts an OpenAI-compatible server with 4 concurrent slots, 32k context, DSpark, FP8 dense weights, FP4 prefill
and image input enabled. `--no-dense-fp8 --no-fp4-prefill` gives the bf16 checkpoint math instead. The first start compiles and autotunes the Triton kernels, which takes a few minutes; Triton caches the results. Later
starts take under a minute.

## API

| | |
|---|---|
| Endpoints | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` (Prometheus) |
| Streaming | SSE, with a usage chunk at the end |
| Reasoning | split into `reasoning_content`; `reasoning_effort` is passed to the chat template |
| Tools | OpenAI tool calling (`tools`, `tool_calls`, tool results) |
| Images | `image_url` content parts (URL or base64) |
| Sampling | `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` |
| Defaults | requests without a `temperature` use the checkpoint's `generation_config.json` (sampling at 1.0, as SGLang does); `--default-temperature 0` makes greedy the default |
| Other | `stop`, `logprobs` / `top_logprobs`, `response_format` (JSON object or JSON schema, through xgrammar), `max_tokens` / `max_completion_tokens`, `ignore_eos` |
| Errors | a prompt longer than the context returns 400 with `code: context_length_exceeded` |

Chat requests with `response_format` reason freely. From the model's `<|content_text|>` marker on, every token is
masked to the grammar until the answer closes.

Requests with `logprobs`, a penalty or a `response_format` decode without speculation, because they need the plain
step's logits. Logprobs are the model's distribution before temperature and top-p.

## Scheduling

- **Slots:** `--slots` (default 4) requests run concurrently. Prompts prefill in 2048-token chunks interleaved with
  the other slots' decode steps.
- **Speculation:** a request alone in the engine speculates with DSpark at a width of 2, 4 or 8. Several requests
  speculate together, at width 4 for 3 or 4 requests (8 for a pair whose recent acceptance is high). They fall back
  to plain batched steps when every request's recent acceptance is low.
- **Prefix reuse:** each slot keeps the state at the end of its last prompt. The scheduler sends a request to the
  free slot holding the longest reusable prefix, so the next turn of a chat only prefills its new tokens (reported
  as `prompt_tokens_details.cached_tokens`).
- **Cancellation:** a client that disconnects mid-stream cancels its request.

## Flags

| Flag | Effect |
|---|---|
| `--slots N` | concurrent requests (default 4) |
| `--max-ctx N` | context per slot (default 32768) |
| `--dense-fp8` / `--no-dense-fp8` | on by default: row-scaled FP8 copies of the attention projections and LM head, FP8 GEMMs for their prefill. Faster decode and prefill; not bit-faithful to the bf16 checkpoint ([quality](benchmarks.md#quality)) |
| `--fp4-prefill` / `--no-fp4-prefill` | on by default: NVFP4 activations and FP4 tensor cores for the prefill expert GEMMs, as vLLM and SGLang do. Needs sm_120 and Triton ≥ 3.6; without them the server starts with it off |
| `--kv-fp8` | E4M3 cache for the full-attention layers (long contexts) |
| `--prefill-chunk N` | prompt tokens per prefill step (default 2048; 4096 is ~5% faster, but pauses other slots for longer) |
| `--no-spec` | no speculative decoding |
| `--no-vision` | text only (saves the 0.6 GB vision tower) |
| `--default-temperature T` | temperature for requests that do not set one |

## Profiles

| Goal | Flags | Notes |
|---|---|---|
| Speed | (none) | FP8 dense weights and FP4 prefill. |
| Checkpoint math | `--no-dense-fp8 --no-fp4-prefill` | bf16 dense layers and prefill experts, as transformers computes them |
| Long context | `--kv-fp8 --max-ctx 262144 --slots 1 --no-vision` | 256k context in 32 GiB |

The engine library (`inkling_engine.engine.Engine`) keeps bf16 as its default; only the server turns FP8 and FP4 on.

Environment knobs, for experiments:

- `INKLING_PDL=0` turns off programmatic dependent launch.
- `INKLING_DRAFT_SHORTLIST=0` makes the drafter score the whole vocabulary.
