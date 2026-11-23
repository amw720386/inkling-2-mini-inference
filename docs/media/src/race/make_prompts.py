"""Write race/race_prompts.json: the 6 decode-benchmark prompts and GSM8K test questions 0-39 as token ids,
at one or more thinking-effort levels.

    python docs/media/src/race/make_prompts.py --tokenizer /path/to/Inkling-2-Mini/tokenizer.json

GSM8K questions use the same chat rendering as tests/reference/sglang_ref.json (thinking effort 0 by default; 0.7 is the template's "medium"), greedy, up to
1024 new tokens, stopping at the model's end of turn. The decode prompts are the benchmark's (256 tokens, ignore_eos).
"""
import argparse, json, pathlib, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
CHAT = "<|message_system|><|content_text|>Thinking effort level: {e}<|end_message|><|message_user|><|content_text|>{q}<|end_message|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--efforts", default="0", help="comma list of thinking-effort levels, as the chat template writes them")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--out", default=str(HERE / "race_prompts.json"))
    a = ap.parse_args()
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(a.tokenizer)
    ref = json.load(open(ROOT / "tests/reference/sglang_ref.json"))
    items = ref["items"] if isinstance(ref, dict) else ref
    for it in items[:4]:  # the rendering reproduces the reference prompts' token ids
        assert tok.encode(it["rendered_prompt"], add_special_tokens=False).ids == it["input_ids"]
    P = [] if a.no_bench else [{"id": f"bench{i}", "input_ids": x["input_ids"], "max_tokens": 256, "ignore_eos": True} for i, x in enumerate(items[:6])]
    with urllib.request.urlopen(GSM) as f:
        test = [json.loads(x) for x in f.read().decode().splitlines() if x.strip()]
    for eff in a.efforts.split(","):
      for i, e in enumerate(test[:a.n]):
        P.append({"id": f"gsm{i}" if eff == "0" else f"gsm{i}-e{eff}", "effort": eff,
                  "input_ids": tok.encode(CHAT.format(e=eff, q=e["question"]), add_special_tokens=False).ids,
                  "max_tokens": 1024, "ignore_eos": False, "question": e["question"],
                  "label": e["answer"].split("####")[-1].strip().replace(",", "")})
    json.dump(P, open(a.out, "w"))
    print(len(P), "prompts")


if __name__ == "__main__":
    main()
