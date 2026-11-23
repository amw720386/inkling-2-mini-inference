"""Markdown comparison table from results/serve_*.json (python tools/results_table.py).

The README and docs/benchmarks.md tables are pasted from this output, so every number traces to a result file.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINES = [
    ("serve_ours.json", "this engine"),
    ("serve_ours-fp8.json", "this engine, `--dense-fp8`"),
    ("serve_ours-fp4.json", "this engine, `--fp4-prefill`"),
    ("serve_vllm.json", "vLLM"),
    ("serve_sglang.json", "SGLang"),
]


def get(r, *path, fmt="{:.0f}"):
    for p in path:
        if not isinstance(r, dict) or p not in r:
            return "-"
        r = r[p]
    return fmt.format(r)


def main():
    head = ["", "decode, greedy", "decode, T=1", "TTFT 1k", "TTFT 4k", "TTFT 8k", "TTFT 30k",
            "2 / 4 concurrent", "GSM8K", "peak GPU"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" + "---:|" * (len(head) - 1)]
    for fname, label in ENGINES:
        p = ROOT / "results" / fname
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if "error" in r:
            continue
        ttft = [get(r, "ttft", n, "ttft_ms") for n in ("1024", "4096", "8192", "30000")]
        conc = f"{get(r, 'concurrency', '2', 'aggregate_tok_s')} / {get(r, 'concurrency', '4', 'aggregate_tok_s')}"
        row = [label, get(r, "decode_greedy", "overall_tok_s"), get(r, "decode_t1", "overall_tok_s"),
               *[t + " ms" if t != "-" else t for t in ttft], conc, get(r, "gsm8k", "accuracy", fmt="{:.3f}"),
               f"{r['peak_gpu_mib'] / 1024:.1f} GiB"]
        lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
