# Serving

```bash
python -m inkling_engine.server --model /path/to/Inkling-2-Mini --port 8000
```

This starts an OpenAI-compatible server with 4 concurrent slots, 32k context, DSpark and image input enabled. The
first start compiles and autotunes the Triton kernels, which takes a few minutes; Triton caches the results. Later
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
- **Speculation:** a request alone in the engine speculates with DSpark. Several requests speculate together when
  the calibrated costs predict it beats a plain batched step.
- **Prefix reuse:** each slot keeps the state at the end of its last prompt. The scheduler sends a request to the
  free slot holding the longest reusable prefix, so the next turn of a chat only prefills its new tokens (reported
  as `prompt_tokens_details.cached_tokens`).
- **Cancellation:** a client that disconnects mid-stream cancels its request.

## Flags

| Flag | Effect |
|---|---|
| `--slots N` | concurrent requests (default 4) |
| `--max-ctx N` | context per slot (default 32768) |
| `--dense-fp8` | row-scaled FP8 copies of the attention projections and LM head, FP8 GEMMs for their prefill. Faster decode and prefill; not bit-faithful to the bf16 checkpoint (GSM8K unchanged) |
| `--fp4-prefill` | NVFP4 activations and FP4 tensor cores for the prefill expert GEMMs, as vLLM and SGLang do. Needs Triton ≥ 3.6 |
| `--kv-fp8` | E4M3 cache for the full-attention layers (long contexts) |
| `--prefill-chunk N` | prompt tokens per prefill step (default 2048; 4096 is ~5% faster, but pauses other slots for longer) |
| `--no-spec` | no speculative decoding |
| `--no-vision` | text only (saves the 0.6 GB vision tower) |
| `--default-temperature T` | temperature for requests that do not set one |

## Profiles

| Goal | Flags | Notes |
|---|---|---|
| Checkpoint math | (none) | bf16 dense layers, as transformers and SGLang compute them |
| Speed | `--dense-fp8` | about 25% faster decode, GSM8K unchanged |
| Long prompts on GeForce | `--dense-fp8 --fp4-prefill` | about 27% faster prefill on top. GeForce runs bf16 with fp32 accumulation at half rate, so on a 5090 this should matter more than on the proxy |
| Long context | `--kv-fp8 --max-ctx 262144 --slots 1 --no-vision` | 256k context in 32 GiB |

Environment knobs, for experiments:

- `INKLING_PDL=0` turns off programmatic dependent launch.
- `INKLING_DRAFT_SHORTLIST=0` makes the drafter score the whole vocabulary.
