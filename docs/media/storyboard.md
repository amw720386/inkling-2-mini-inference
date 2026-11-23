# Media storyboard

The plan for the launch video, the README GIFs and the architecture diagrams. Every number on screen comes from
`results/` or `docs/`, and the file it comes from is listed next to it. Change a number here first, then in the
scenes.

## Look

All media share `src/theme.css`: the colors, the fonts (Inter and JetBrains Mono, OFL, in `src/fonts/`) and the
type scale.

| Element | Spec |
|---|---|
| Background | `#05080f` (near-black navy) |
| Headline | Inter 300, 46 px at 1080 px frame height, `#e8edf4`. One short sentence per beat. |
| Caps label | JetBrains Mono 400, 15 px, letter-spacing 0.2em, uppercase, `#7d8899`. Sits above the headline. |
| Cards | `#0a111c` fill, 1 px `#1c2636` border, 10 px radius |
| Colors | blue `#3b82f6` = this engine and active experts; grey `#6b7280` = baselines and idle experts; orange `#f59e0b` = the loop and draft tokens; green `#22c55e` = correct or accepted, nothing else; red `#ef4444` = rejected drafts only |
| Margins | 64 px at 1080 px. The 16:9 cut uses the same sizes inside a 1920 x 1080 frame. |
| Motion | Ease in-out, nothing faster than 0.25 s. Text fades in over 0.4 s with a small (12 px) upward move. No music. |
| Glyphs | ✓, → and × are not in the font subset. Draw ✓ and → as inline SVG; × is in Latin-1 and renders. |

Rules for every frame:

- No internal names, hosts, accounts or paths. "MADE AT MAGNI" appears only on the end card.
- The hardware (RTX PRO 6000 Blackwell) is on screen in the race and on the how-it-works card.
- Frame-by-frame review before anything is merged: no clipped labels, no overlapping text, no jumps.

## Numbers (source of truth for every scene)

From `docs/benchmarks.md` (RTX PRO 6000 Blackwell on Modal, each engine held to 32 GB, medians over containers,
measured 2026-11-23; regenerate with `tools/results_table.py`):

| | this engine (default) | bf16 | SGLang | vLLM 0.34 |
|---|---:|---:|---:|---:|
| decode, greedy, 1 request (tok/s) | **344** | 273 | 232 | 235 |
| decode, temperature 1 (tok/s) | 264 | 239 | 192 | 212 |
| TTFT 8k (ms) | 341 | 467 | 434 | does not fit |
| 2 / 4 concurrent (tok/s) | 464 / **571** | 437 / 554 | 298 / 514 | 302 / 437 |
| GSM8K, n = 200 | 0.837 | 0.853 | 0.845 | 0.840 |

Earlier figures (363, 325, 280, 545 vs 474) are withdrawn. Still to do before release:

- Re-render the video and `race.gif`. The race data is an earlier engine's `--dense-fp8` run and its tok/s used the
  old timing; re-capture with `race/capture.py --engines ours,sglang,vllm`, then `pick.py` and `build_race.py`.
- Memory scene, `memory.gif` and the `memory-layout` diagrams show the bf16 profile. The FP8 default measures
  25.9 GiB after start, 27.4 GiB peak (`tests/gpu/memcheck.py`); its weights are smaller, so the segments need
  new sizes.
- Drop or redraw `scheduler.gif`: its idle-GPU claim is wrong (a trace shows the GPU busy 91-94% of the time).
- `render_modal.py` is meant to run `render.sh` on a Modal container. It has not been run yet.

Model facts (`docs/architecture.md`): 26.4B parameters, 3.4B active per token; 24 stored layers = prelude 6
(0-5) + loop core 12 (6-17) × 3 passes + coda 6 (18-23) = **48 layer visits per token**; 128 routed experts,
**top-8**, NVFP4 (layers 0-19) / MXFP8 (20-23); 20 sliding-window layers (window 1024) + 4 full-attention layers;
~12 fused kernels per visit, **one CUDA graph replay per decode step**. DSpark: reads the target's hidden states at
visits 5, 23, 41, 48, drafts **7 tokens** in one pass, verify width 2 / 4 / 8.

Memory (`docs/benchmarks.md`, `docs/architecture.md`): default server = **4 slots × 32k**, 26.4 GiB after start,
28.8 GiB peak (`tests/gpu/memcheck.py`). Long context = `--kv-fp8 --max-ctx 262144 --slots 1`: **256k on one card**,
25.3 GiB after start, 28.6 GiB peak; 131k-token prefill 26.3 s; decode at 131k 112 / 123 tok/s (plain / DSpark).
Segment sizes inside those totals (weights, KV, rings, graphs) are computed from `inkling_engine/cache.py`
(`CacheSpec.bytes`) and the checkpoint's file sizes; write the arithmetic in the scene's source.

Where we lose (must appear somewhere in the video): SGLang serves 4 concurrent requests faster (545 vs 474 tok/s).

## Hero video (1080 × 1080, ~23.7 s; 16:9 cut 1920 × 1080 with the same timing)

Each scene is rendered at both sizes, 30 fps, exact duration, as `scene-<n>-<1x1|16x9>.mp4`
(H.264, CRF 12, yuv420p), with its sources under `docs/media/src/`. `src/render.sh` joins them with 0.4 s cross-fades. A scene's first frame must not be empty (the cross-fade
would dip to black): show its caps label and headline from frame 0 and build the rest after.

### Scene 1: opener, 0.0-6.0 s

Dark 3D scene, slow camera dolly. The 24 stored layers as tile walls: prelude, loop core, coda. Each layer is a
wall of 128 expert tiles (grey, dim). A token (light beam) enters the prelude, passes through the loop core three
times (orange arc on each return), exits the coda. On every visit, 8 of the 128 tiles light up blue.

| t | caps label | headline |
|---|---|---|
| 0.3-2.2 s | `INKLING-2-MINI · 26B LOOPED MoE` | 26B parameters. 3.4B per token. |
| 2.2-4.1 s | `PRELUDE 6 · LOOP 12 × 3 · CODA 6` | Every token makes 48 layer visits. |
| 4.1-6.0 s | `128 EXPERTS · TOP-8 · ONE 32 GB GPU` | 8 of 128 experts light up each visit. |

### Scene 2: real-run race, 5.6-12.4 s (6.8 s clip)

Caps label `DEMO · REAL RUN · ONE RTX PRO 6000 · PLAYED AT 1×`, the prompt as a one-line headline, the prompt
text small below it. Three cards stacked (1:1) or side by side (16:9):

1. `THIS ENGINE · default` (blue bar)
2. `SGLANG` (grey bar)
3. `vLLM 0.34` (grey bar)

Each card: a progress bar filling with the real token timestamps, a live timer and tok/s in the top right, the
answer text streaming in, a green ✓ correct pill when the final answer matches the key. Played at 1× from the
captured per-token times (one greedy run per engine, same token-id prompt, same `max_tokens`, DSpark on for all,
the `bench/serve_bench.py` server flags). Pick a GSM8K test question whose answers are correct for all three and
that runs 1-5 s on the slowest engine. Never smooth or rescale the curves.

Punchline (computed from the captured run, not from the table): `Same answer, N× sooner than SGLang.`, with one
small line under it giving the same run's 6-prompt average ratio. Footer line, small: `RTX PRO 6000 Blackwell,
32 GB budget per engine, greedy, DSpark on all three.`

The race question and its selection rule are documented in the race scene's source.

### Scene 3: how it works, 12.0-18.0 s

Caps label `HOW`, headline: `One CUDA graph per step. DSpark drafts 7 tokens ahead.`

Three boxes joined by arrows, each with a caps label, a two-line body and a small tile glyph:

1. `DSPARK DRAFTER` / "drafts 7 tokens from 4 of the target's hidden states" / glyph: 7 orange tiles
2. `LOOPED TARGET` / "48 layer visits, ~12 fused Triton kernels each, one graph replay" / glyph: 3 rows of tiles, 8 of 128 lit blue
3. `VERIFY` / "keeps the longest matching prefix; width 2, 4 or 8 per round" / glyph: tiles green then one red

Legend line: `■ active expert  ■ draft token  ■ accepted`. Caveat, one line: `Measured on an RTX PRO 6000
Blackwell. 4 concurrent requests: 571 tok/s vs 514 (SGLang) and 437 (vLLM).`

### Scene 4: memory, 17.6-20.6 s

Caps label `32 GB · ONE CARD`. A 32 GiB bar (or 3D slab) filling: weights, then four 32k KV slots (blue), CUDA
graphs, free. Headline: `Four 32k contexts in 32 GB.` Then the slots merge into one long FP8 slot. Headline:
`Or 256k on one card.` Small numbers: `26.4 GiB after start · 28.8 GiB peak` → `25.3 / 28.6 GiB with --kv-fp8`.

### Scene 5: end card, 20.2-23.7 s

- Kicker: `INKLING-2-MINI · ONE 32 GB GPU`
- Headline: `Faster decode than SGLang and vLLM.`
- Three zero-based bars growing in: this engine 344, SGLang 232, vLLM 235 tok/s (beside the text in 16:9)
- Context: `One request, greedy: medians over containers, same GPU, same 32 GB budget, server default. 4 requests: 571 vs 514 (SGLang) and 437 (vLLM) tok/s. Python and Triton.`
- URL: `github.com/amw720386/inkling-2-mini-inference`
- Bottom: `MADE AT MAGNI`

## README GIFs (≤ 8 MB, 800-1000 px wide, clean loop)

| File | Content |
|---|---|
| `docs/media/race.gif` | the race scene, 960 px, loops after the punchline |
| `docs/media/loop.gif` | one token through prelude → loop × 3 → coda, experts lighting up |
| `docs/media/memory.gif` | 32 GiB packing: 4 × 32k slots, then the 256k FP8 single-slot layout |
| `docs/media/dspark.gif` | 7 drafts proposed, verified at width 8, prefix accepted (green), rest rejected (red), next round |

## Architecture diagrams (SVG, `-dark.svg` and `-light.svg`)

Under `docs/media/`:

- `request-path`: HTTP server → scheduler (slots, chunked prefill, prefix reuse) → prefill / decode → Triton kernels
- `decode-step`: embedding → prelude → loop × 3 (gated messages) → coda; per visit: projections, attention, router, top-8 experts; LM head → sampler → DSpark verify
- `memory-layout`: the 32 GiB budget, both profiles
- `kernel-fusion`: the ~12 launches of one layer visit (`docs/architecture.md` "Decode step"), with what each one fuses

README embeds with `<picture>` and `prefers-color-scheme` sources.

## Re-rendering

Sources live in `docs/media/src/`. One command re-renders everything: `docs/media/src/render.sh`.
