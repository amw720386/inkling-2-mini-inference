"""GSM8K few-shot accuracy through the engine (Modal: parity --args "gsm8k_eval.py [n] [bf16|fp8]").

Same protocol as SGLang's few_shot_gsm8k used for the baseline: 5-shot "Question: ...\\nAnswer: ..." prompts
from the train split, greedy, answers cut at the next "Question", last number compared with the label.
"""

import json
import re
import sys
import time
import urllib.request

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
mode = sys.argv[3] if len(sys.argv) > 3 else "bf16"
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))

BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"


def load(split):
    with urllib.request.urlopen(BASE + f"{split}.jsonl") as f:
        return [json.loads(line) for line in f.read().decode().splitlines() if line.strip()]


def label(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("Question")[0])
    return nums[-1].replace(",", "").rstrip(".") if nums else None


train, test = load("train"), load("test")
shots = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train[:5])
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
eng = Engine(path, max_ctx=8192, spec=True, dense_fp8=(mode == "fp8"))
stop = {eng.cfg.eos_token_id}
correct, gen_tokens, t0 = 0, 0, time.time()
for i, e in enumerate(test[:n]):
    ids = tok.encode(shots + f"Question: {e['question']}\nAnswer:", add_special_tokens=False)
    out_text, toks = "", []
    for t in eng.stream(ids, 320, stop):
        toks.append(t)
        if len(toks) % 16 == 0:
            out_text = tok.decode(toks)
            if "Question" in out_text:
                break
    out_text = tok.decode(toks)
    gen_tokens += len(toks)
    correct += int(pred(out_text) == label(e["answer"]))
dt = time.time() - t0
print(f"GSM8K {mode}: {correct}/{n} = {correct / n:.3f}  ({gen_tokens} tokens in {dt:.0f}s, {gen_tokens / dt:.0f} tok/s incl. prefill)")
