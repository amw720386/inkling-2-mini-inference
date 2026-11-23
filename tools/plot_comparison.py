"""Chart for the README from results/serve_*.json (python tools/plot_comparison.py): the whole-card comparison.

Three panels, each with its own axis (different measures are never put on one scale): one and four MT-Bench requests in
flight, and time to first token on a 4k prompt. Engines keep fixed colors.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from results_table import WHOLE, median  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
# fixed per engine; order validated (dataviz palette check)
COLORS = {"this engine": "#2a78d6", "vLLM, optimized flags": "#eda100", "vLLM": "#1baf7a", "SGLang": "#eb6834"}
ORDER = list(COLORS)
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def rows_for(path):
    """(label, color, median over containers) of every engine that measured this metric."""
    by = {label: median(files, path) for files, label in WHOLE}
    return [(label, COLORS[label], by[label]) for label in ORDER if by.get(label) is not None]


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
    panel(axes[0], "Decode, 1 request, 24 MT-Bench prompts, greedy (tok/s)", rows_for(("heldout", "greedy_k1", "tok_s")), tok)
    panel(axes[1], "Throughput, 4 requests in flight, same prompts (tok/s)", rows_for(("heldout", "greedy_k4", "tok_s")), tok)
    panel(axes[2], "Time to first token, 4k-token prompt (ms)", rows_for(("ttft", "4096", "ttft_ms")), tok,
          lower_better=True)
    fig.text(0.01, 0.005, "Same RTX PRO 6000 Blackwell, no memory cap, every engine on its own flags,\n"
             "same client (bench/serve_bench.py). Median of two containers.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    out = ROOT / "docs" / "img" / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
