"""Write tests/reference/heldout_prompts.json: MT-Bench first turns that nothing in this repo was tuned on.

    python tools/build_heldout.py --tokenizer /path/to/Inkling-2-Mini/tokenizer.json

The first three questions of each of MT-Bench's eight categories (24 prompts; FastChat, Apache-2.0), in the chat rendering
of tests/reference/sglang_ref.json (thinking effort 0), as token ids. The scheduler's batch widths and the draft temperature were chosen on
sglang_ref.json's six prompts, never on these.
"""

import argparse
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
CHAT = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{q}<|end_message|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--per-category", type=int, default=3)
    a = ap.parse_args()
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(a.tokenizer)
    with urllib.request.urlopen(URL) as f:
        questions = [json.loads(x) for x in f.read().decode().splitlines() if x.strip()]
    seen, items = {}, []
    for q in questions:
        c = q["category"]
        if seen.get(c, 0) < a.per_category:
            seen[c] = seen.get(c, 0) + 1
            ids = tok.encode(CHAT.format(q=q["turns"][0]), add_special_tokens=False).ids
            items.append({"question_id": q["question_id"], "category": c, "input_ids": ids})
    out = ROOT / "tests" / "reference" / "heldout_prompts.json"
    out.write_text(json.dumps({"source": URL, "items": items}))
    print(len(items), "prompts,", sum(len(i["input_ids"]) for i in items), "tokens ->", out)


if __name__ == "__main__":
    main()
