"""Markdown comparison table from results/serve_*.json (python tools/results_table.py).

Each engine has one file per benchmark container (serve_ours.json, serve_ours-2.json, ...). A cell is the median over
the containers that measured it. The README and docs/benchmarks.md tables are pasted from this output, so every number
traces to a result file. `--spread` prints each container's value instead, `--uncapped` the runs without the 32 GB memory cap.
"""

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINES = [("ours", "this engine"), ("ours-bf16", "this engine, bf16"), ("vllm", "vLLM"), ("sglang", "SGLang")]
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
    """Result files of one engine, one per container."""
    pat = re.compile(rf"serve_{engine}(-\d+)?\.json$")  # "ours" does not match "ours-uncapped"
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


def main():
    spread = "--spread" in sys.argv
    print("| | " + " | ".join(m[0] for m in METRICS) + " |")
    print("|---|" + "---:|" * len(METRICS))
    for engine, label in (UNCAPPED if "--uncapped" in sys.argv else ENGINES):
        runs, cells = containers(engine), []
        for name, path, unit in METRICS:
            v = values(runs, path)
            fmt = "{:.3f}" if name == "GSM8K" else "{:.1f}" if unit == " GiB" else "{:.0f}"
            if path == ("peak_gpu_mib",) and v:
                v = [max(v)]
            cells.append("-" if not v else " / ".join(fmt.format(x) for x in v) + unit if spread
                         else fmt.format(statistics.median(v)) + unit)
        print(f"| {label} ({len(runs)}) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
