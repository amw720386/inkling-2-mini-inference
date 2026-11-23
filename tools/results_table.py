"""Markdown comparison table from results/serve_*.json (python tools/results_table.py).

Each engine has one file per benchmark container (serve_ours.json, serve_ours-2.json, ...). A cell is the median over
the containers that measured it. The README and docs/benchmarks.md tables are pasted from this output, so every number
traces to a result file. `--spread` prints each container's value instead, `--whole-card` the table where every engine
runs on its own best flags with no memory cap, `--uncapped` the older prompts and TTFT without the cap.
"""

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINES = [("ours", "this engine"), ("ours-bf16", "this engine, bf16"), ("vllm", "vLLM"), ("sglang", "SGLang")]
# whole card: (files per container, label); a container may hold only some of the sections
WHOLE = [(("ours-uncapped", "ours-uncapped-heldout"), "this engine"),
         (("sglang-uncapped", "sglang-uncapped-heldout"), "SGLang"),
         (("vllm-uncapped", "vllm-uncapped-heldout"), "vLLM"),
         (("vllm-fp8kv-uncapped",), "vLLM, optimized flags")]
WHOLE_METRICS = [
    ("1 request", ("heldout", "greedy_k1", "tok_s"), ""), ("1 request, T=1", ("heldout", "t1_k1", "tok_s"), ""),
    ("2 requests", ("heldout", "greedy_k2", "tok_s"), ""), ("4 requests", ("heldout", "greedy_k4", "tok_s"), ""),
    ("random 1k/1k, 1 request", ("random1k", "1", "tok_s"), ""), ("random 1k/1k, 4 requests", ("random1k", "4", "tok_s"), ""),
    ("TTFT 4k", ("ttft", "4096", "ttft_ms"), " ms"), ("TTFT 30k", ("ttft", "30000", "ttft_ms"), " ms"),
]
UNCAPPED = [("ours-uncapped", "this engine"), ("vllm-uncapped", "vLLM"), ("sglang-uncapped", "SGLang")]
METRICS = [  # (column, path, unit)
    ("decode, greedy", ("decode_greedy", "overall_tok_s"), ""), ("decode, T=1", ("decode_t1", "overall_tok_s"), ""),
    ("2 concurrent", ("concurrency", "2", "aggregate_tok_s"), ""),
    ("4 concurrent", ("concurrency", "4", "aggregate_tok_s"), ""),
    ("TTFT 1k", ("ttft", "1024", "ttft_ms"), " ms"), ("TTFT 4k", ("ttft", "4096", "ttft_ms"), " ms"),
    ("TTFT 8k", ("ttft", "8192", "ttft_ms"), " ms"), ("TTFT 30k", ("ttft", "30000", "ttft_ms"), " ms"),
    ("GSM8K", ("gsm8k", "accuracy"), ""), ("peak GPU", ("peak_gpu_mib",), " GiB"),
]


def containers(engine):
    """Result files of one engine, one per container (`engine` may be a tuple of file prefixes)."""
    names = [engine] if isinstance(engine, str) else engine
    pat = re.compile(rf"serve_({'|'.join(map(re.escape, names))})(-\d+)?\.json$")  # "ours" does not match "ours-uncapped"
    files = sorted(p for p in (ROOT / "results").glob("serve_*.json") if pat.match(p.name))
    return [r for r in (json.loads(p.read_text()) for p in files) if "error" not in r]


def values(runs, path):
    out = []
    for r in runs:
        for p in path:
            r = r.get(p) if isinstance(r, dict) else None
        if r is not None:
            out.append(r / 1024 if path == ("peak_gpu_mib",) else r)
    return out


def median(engine, path):
    v = values(containers(engine), path)
    return statistics.median(v) if v else None


def table(rows, metrics, spread):
    lines = ["| | " + " | ".join(m[0] for m in metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for engine, label in rows:
        runs, cells = containers(engine), []
        for name, path, unit in metrics:
            v = values(runs, path)
            fmt = "{:.3f}" if name == "GSM8K" else "{:.1f}" if unit == " GiB" else "{:.0f}"
            if path == ("peak_gpu_mib",) and v:
                v = [max(v)]
            cells.append("-" if not v else " / ".join(fmt.format(x) for x in v) + unit if spread
                         else fmt.format(statistics.median(v)) + unit)
        n = len(values(runs, metrics[0][1])) if isinstance(engine, tuple) else len(runs)  # containers behind the first column
        lines.append(f"| {label} ({n}) | " + " | ".join(cells) + " |")
    return lines


def main():
    spread = "--spread" in sys.argv
    if "--whole-card" in sys.argv:
        print("\n".join(table(WHOLE, WHOLE_METRICS, spread)))
    else:
        print("\n".join(table(UNCAPPED if "--uncapped" in sys.argv else ENGINES, METRICS, spread)))


if __name__ == "__main__":
    main()
