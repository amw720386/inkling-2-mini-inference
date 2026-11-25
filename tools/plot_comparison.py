"""Comparison chart from results/serve_*.json (python tools/plot_comparison.py), in the look of docs/media.

Six panels, each with its own zero-based axis: MT-Bench requests in flight, random 1k/1k, and time to first token. A bar is
the median over the two whole-card containers, the thin line spans both. Panels where this engine is level or behind are
drawn too. Colors follow docs/media/src/theme.css (blue: this engine, grey: the others) and the fonts are embedded, as in
the architecture diagrams. Writes docs/img/comparison.svg; standard library only.
"""

import base64
import pathlib
from html import escape

from results_table import WHOLE, containers, median, values

ROOT = pathlib.Path(__file__).resolve().parent.parent
FONTS = ROOT / "docs" / "media" / "src" / "fonts"
ORDER = ["this engine", "vLLM, optimized flags", "vLLM", "SGLang"]
NAMES = ["This engine", "vLLM, optimized flags", "vLLM", "SGLang"]
THEME = {  # docs/media/src/theme.css; the same dark look on GitHub's light and dark pages, like the GIFs
    "bg": "#05080f", "card": "#0a111c", "border": "#1c2636", "text": "#e8edf4", "body": "#c3ccd8", "muted": "#7d8899",
    "blue": "#3b82f6", "value": "#60a5fa", "grey": "#6b7280"}
PANELS = [  # label, headline, unit, result path
    ("MT-Bench · greedy", "1 request", "tok/s", ("heldout", "greedy_k1", "tok_s")),
    ("MT-Bench · greedy", "2 requests", "tok/s", ("heldout", "greedy_k2", "tok_s")),
    ("MT-Bench · greedy", "4 requests", "tok/s", ("heldout", "greedy_k4", "tok_s")),
    ("Random 1k in / 1k out", "1 request", "tok/s", ("random1k", "1", "tok_s")),
    ("First token · lower is better", "4k prompt", "ms", ("ttft", "4096", "ttft_ms")),
    ("First token · lower is better", "30k prompt", "ms", ("ttft", "30000", "ttft_ms")),
]
W, MARGIN, GAP, PAD = 1320, 40, 20, 24
CARD_W = (W - 2 * MARGIN - GAP) / 2
CARD_H, TOP = 222, 126
NAME_W, VALUE_W, ROW = 178, 96, 33
SANS = "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "JBMono, 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def style():
    faces = [("Inter", 300, "inter-latin-300-normal"), ("Inter", 400, "inter-latin-400-normal"),
             ("JBMono", 400, "jetbrains-mono-latin-400-normal")]
    return "".join(f"@font-face{{font-family:{fam};font-weight:{w};src:url(data:font/woff2;base64,"
                   f"{base64.b64encode((FONTS / (f + '.woff2')).read_bytes()).decode()}) format(\"woff2\")}}" for fam, w, f in faces)


def text(x, y, s, size, fill, *, fam=SANS, weight=400, caps=False, anchor="start"):
    spacing = f' letter-spacing="{0.2 * size:.1f}"' if caps else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{fill}"'
            f'{spacing} text-anchor="{anchor}">{escape(s.upper() if caps else s)}</text>')


def panel(x, y, spec, t):
    label, head, unit, path = spec
    files = {lab: f for f, lab in WHOLE}
    med = [median(files[e], path) for e in ORDER]
    span = [values(containers(files[e]), path) for e in ORDER]
    scale = (CARD_W - 2 * PAD - NAME_W - VALUE_W) / max(max(s) for s in span)
    out = [f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" fill="{t["card"]}" stroke="{t["border"]}"/>',
           text(x + PAD, y + 34, label, 11, t["muted"], fam=MONO, caps=True),
           text(x + PAD, y + 64, head, 21, t["text"], weight=300)]
    x0 = x + PAD + NAME_W
    for i, (name, m, s) in enumerate(zip(NAMES, med, span)):
        ours, cy = i == 0, y + 96 + i * ROW
        out.append(text(x + PAD, cy + 5, name, 14, t["text"] if ours else t["body"]))
        out.append(f'<rect x="{x0}" y="{cy - 6}" width="{m * scale:.1f}" height="12" rx="3" fill="{t["blue"] if ours else t["grey"]}"/>')
        out.append(f'<line x1="{x0 + min(s) * scale:.1f}" x2="{x0 + max(s) * scale:.1f}" y1="{cy}" y2="{cy}" stroke="{t["text"]}" stroke-width="1.5"/>')
        vx = x0 + max(s) * scale + 12
        out.append(f'<text x="{vx:.1f}" y="{cy + 6}" font-family="{SANS}" font-size="17" fill="{t["value"] if ours else t["body"]}">{m:,.0f}'
                   f'<tspan font-family="{MONO}" font-size="10" letter-spacing="1.2" fill="{t["muted"]}" dx="7">{unit.upper()}</tspan></text>')
    return out


def main():
    t = THEME
    rows = (len(PANELS) + 1) // 2
    height = TOP + rows * (CARD_H + GAP) + 44
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" role="img">',
           f"<title>This engine, vLLM and SGLang on one RTX PRO 6000 Blackwell</title><style>{style()}</style>",
           f'<rect width="{W}" height="{height}" fill="{t["bg"]}"/>',
           text(MARGIN, 52, "Same card · same client · no memory cap", 12, t["muted"], fam=MONO, caps=True),
           text(MARGIN, 92, "Faster decode on chat prompts at one and four requests. Level with vLLM at two.", 29, t["text"], weight=300)]
    for i, spec in enumerate(PANELS):
        out += panel(MARGIN + (i % 2) * (CARD_W + GAP), TOP + (i // 2) * (CARD_H + GAP), spec, t)
    out.append(text(MARGIN, height - 28, "RTX PRO 6000 Blackwell, every engine on its own flags. Bar: median of two containers; line: both. "
                    "This engine runs FP8 dense weights, the others bf16.", 14, t["muted"]))
    out.append("</svg>")
    path = ROOT / "docs" / "img" / "comparison.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
