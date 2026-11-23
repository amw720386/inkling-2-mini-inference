"""Charts for the README from results/serve_*.json (python tools/plot_comparison.py).

Three panels, each with its own axis (different measures are never put on one scale): decode at batch 1, aggregate
throughput with 4 concurrent requests, and time to first token on a 4k prompt. Engines keep fixed colors.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from results_table import ENGINES, median  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
COLORS = {"ours": "#2a78d6", "ours-bf16": "#eda100", "vllm": "#1baf7a", "sglang": "#eb6834"}  # fixed per engine
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def rows_for(path):
    """(label, color, median over containers) of every engine that measured this metric."""
    out = []
    for engine, label in ENGINES:
        v = median(engine, path)
        if v is not None:
            out.append((label, COLORS[engine], v))
    return out


def panel(ax, title, rows, fmt, lower_better=False):
    labels = [lab for lab, _, _ in rows]
    vals = [v for _, _, v in rows]
    bars = ax.barh(range(len(rows)), vals, color=[c for _, c, _ in rows], height=0.62, edgecolor=SURFACE,
                   linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, "  " + fmt(v), va="center", ha="left",
                color=TEXT, fontsize=10)
    ax.set_yticks(range(len(rows)), labels, color=TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title + ("  (lower is better)" if lower_better else ""), loc="left", color=TEXT, fontsize=11,
                 pad=8)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)


def main():
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    tok = "{:.0f}".format
    panel(axes[0], "Decode, 1 request, greedy, DSpark (tok/s)", rows_for(("decode_greedy", "overall_tok_s")), tok)
    panel(axes[1], "Throughput, 4 concurrent requests (tok/s)",
          rows_for(("concurrency", "4", "aggregate_tok_s")), tok)
    panel(axes[2], "Time to first token, 4k-token prompt (ms)", rows_for(("ttft", "4096", "ttft_ms")), tok,
          lower_better=True)
    fig.text(0.01, 0.005, "Same RTX PRO 6000 Blackwell, 32 GB budget, same client (bench/serve_bench.py). "
             "Median over containers.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    out = ROOT / "docs" / "img" / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
