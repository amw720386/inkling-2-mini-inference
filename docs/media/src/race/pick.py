"""Pick the race question from the captured runs and write race/race_capture.json.

    python docs/media/src/race/pick.py /tmp/race_raw_ours-fp8.json /tmp/race_raw_sglang.json /tmp/race_raw_vllm.json

Fixed rule, applied to the warm repeat (rep 1 of 0-1) of every engine, never per engine:
  1. the final answer is correct on all three engines;
  2. the slowest engine finishes in 2.0-5.0 s (request sent to last token);
  3. this engine's completion and the fastest baseline's (the ratio's denominator) have the same token count within
     5%, so the ratio measures speed, not length (greedy outputs differ between FP8 and bf16 engines, more so on long
     answers); the third card shows its own token count. --strict requires all three within 5%;
  4. of those, the question whose speedup (fastest baseline / this engine) is closest to the 6-prompt decode average
     (--target results: the published 363.1 / 241.6 = 1.50; --target capture: the same 6 prompts re-measured in this
     capture), so the race shows a typical question, not the best one.
Prints every candidate, so the choice can be checked.
"""
from __future__ import annotations

import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REP = 1
NAMES = {"ours-fp8": ("This engine", 'This engine · <span class="flag">--dense-fp8</span>'),
         "ours": ("This engine", "This engine · bf16"),
         "sglang": ("SGLang", "SGLang · DSpark"), "vllm": ("vLLM 0.34", "vLLM 0.34 · DSpark")}


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def text_of(run):
    return "".join(c[1] for c in run["chunks"])


def thinking_len(split):
    """Characters of visible thinking text before the answer, from the token strings with their markers."""
    text = split.get("text") or ""
    if "<|content_thinking|>" in text:  # servers that honor skip_special_tokens=False
        body = text.split("<|content_thinking|>", 1)[1]
        if "<|end_message|>" in body:
            return len(re.sub(r"<\|[a-z_]+\|>", "", body.split("<|end_message|>", 1)[0]))
    toks = split.get("tokens") or []
    out, in_think = [], False
    for t in toks:
        if t == "<|content_thinking|>":
            in_think = True
        elif t == "<|end_message|>" and in_think:
            return len("".join(out))
        elif in_think:
            out.append(t)
    return None


def decode_rate(run):
    c, n = run["chunks"], run["usage"]["completion_tokens"]
    d = c[-1][0] - c[0][0]
    return (n - 1) / d if d > 0 else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--strict", action="store_true", help="all three engines within 5%% tokens")
    ap.add_argument("--target", choices=["results", "capture"], default="results",
                    help="ratio target: the published 6-prompt averages, or the same 6 prompts re-measured in this capture")
    a = ap.parse_args()
    raws = {}
    for p in a.raw:
        key = pathlib.Path(p).stem.replace("race_raw_", "")
        raws[key] = json.load(open(p))
        assert "error" not in raws[key], (key, raws[key].get("error", "")[-500:])
    ours_key = next(k for k in raws if k.startswith("ours"))
    base_keys = [k for k in raws if not k.startswith("ours")]
    prompts = {p["id"]: p for p in json.load(open(HERE / "race_prompts.json"))}
    res = {k: json.load(open(ROOT / f"results/serve_{k}.json"))["decode_greedy"]["overall_tok_s"] for k in raws}

    runs = {k: {(r["id"], r["rep"]): r for r in v["runs"]} for k, v in raws.items()}
    bench = {}
    for k in raws:  # the decode benchmark, re-measured by this capture (sanity check against results/)
        per = []
        for rep in (0, 1):
            rs = [runs[k][(f"bench{i}", rep)] for i in range(6)]
            tot_t = sum(r["usage"]["completion_tokens"] - 1 for r in rs)
            tot_s = sum((r["usage"]["completion_tokens"] - 1) / decode_rate(r) for r in rs)
            per.append(round(tot_t / tot_s, 1))
        bench[k] = {"overall_tok_s_reps": per, "results_file_tok_s": res[k]}
        print(f"bench6 {k:9s} this capture {per} tok/s   results/ {res[k]}")
    mean = {k: sum(v["overall_tok_s_reps"]) / 2 for k, v in bench.items()}
    t_res = res[ours_key] / max(res[k] for k in base_keys)
    t_cap = mean[ours_key] / max(mean[k] for k in base_keys)
    target = t_cap if a.target == "capture" else t_res
    print(f"ratio target: results {t_res:.3f}, this capture {t_cap:.3f}; using {a.target}")

    cands = []
    for pid, p in prompts.items():
        if not pid.startswith("gsm"):
            continue
        rs = {k: runs[k].get((pid, REP)) for k in raws}
        if any(r is None or not r["chunks"] for r in rs.values()):
            continue
        starts, ok = {}, {}
        for k, r in rs.items():
            t = text_of(r)
            # thinking/answer split: the engine's own untimed request with special tokens kept gives the answer text;
            # its first 30 characters, found in the timed stream, mark where the answer starts there (the untimed
            # rerun can differ slightly from the timed one, so the texts are not compared whole)
            sp = raws[k].get("split", {}).get(pid, {})
            tl = thinking_len(sp)
            vis = re.sub(r"<\|[a-z_]+\|>", "", sp.get("text") or "") if sp.get("text") else "".join(
                x for x in (sp.get("tokens") or []) if not re.fullmatch(r"<\|[a-z_]+\|>", x))
            head = vis[tl:tl + 30] if tl is not None else ""
            j = t.rfind(head) if len(head) >= 10 else -1
            s = j if j > 0 else None
            starts[k] = s
            ok[k] = s is not None and last_number(t[s:]) == p["label"]
        ends = {k: r["chunks"][-1][0] for k, r in rs.items()}
        ratio = min(ends[k] for k in base_keys) / ends[ours_key]
        row = {"id": pid, "ends": ends, "ratio": round(ratio, 3), "correct": ok,
               "tokens": {k: r["usage"]["completion_tokens"] for k, r in rs.items()}, "starts": starts}
        cands.append(row)
        print(f"{pid:6s} ends " + " ".join(f"{k}={v:.2f}" for k, v in ends.items()) + f"  ratio {ratio:.2f}  correct "
              + " ".join(f"{k}={int(v)}" for k, v in ok.items()) + f"  tokens {row['tokens']}")
    def same_length(c):
        if a.strict:
            ks = list(c["tokens"])
        else:
            ks = [ours_key, min(base_keys, key=lambda k: c["ends"][k])]
        t = [c["tokens"][k] for k in ks]
        return max(t) <= 1.05 * min(t)

    good = [c for c in cands if all(c["correct"].values()) and 2.0 <= max(c["ends"].values()) <= 5.0 and same_length(c)]
    assert good, "no question passes the rule"
    best = min(good, key=lambda c: abs(c["ratio"] - target))
    print(f"\n{len(good)} of {len(cands)} pass; target ratio {target:.3f}; picked {best['id']} ratio {best['ratio']}")

    p = prompts[best["id"]]
    engines = []
    for k in [ours_key] + base_keys:
        r = runs[k][(best["id"], REP)]
        engines.append({"key": k, "name": NAMES[k][0], "label": NAMES[k][1], "chunks": r["chunks"],
                        "completion_tokens": r["usage"]["completion_tokens"], "finish_reason": r["finish_reason"],
                        "answer_start": best["starts"][k], "gpu": raws[k].get("gpu"), "startup_s": raws[k].get("startup_s")})
    out = {"question_id": best["id"], "question": p["question"], "label": p["label"], "prompt_ids": p["input_ids"],
           "headline": "One GSM8K question. Three engines.",
           "layout": {"t0": 1.2, "duration": 8.0, "punch_delay": 0.3, "gif_duration": 6.5},
           "rule": __doc__.split("\n\n", 2)[1].strip(), "rep": REP, "target_ratio": round(target, 3), "target_from": a.target, "strict": a.strict,
           "selection_note": None if a.strict else (
               "Rule 3 was relaxed to this form after the capture was seen: the strict form (all three engines within 5% "
               "tokens) left only gsm108-e0.99 (1.39 s race, 1.18x). The pick's ratio can sit above the run's 6-prompt "
               "ratio; the scene shows both."),
           "engines": engines, "bench6": bench,
           "candidates": [{k: v for k, v in c.items() if k != "starts"} for c in cands]}
    (HERE / "race_capture.json").write_text(json.dumps(out, ensure_ascii=False))
    print("wrote", HERE / "race_capture.json")


if __name__ == "__main__":
    main()
