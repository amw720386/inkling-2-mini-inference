"""One OpenAI-API benchmark for every server (this engine, SGLang, vLLM), run on the same GPU.

    python bench/serve_bench.py --base http://127.0.0.1:8000/v1 --model <name> --prompts tests/reference/sglang_ref.json

Everything goes through /v1/completions with token-id prompts, so tokenization and chat templates cannot differ:
  decode       6 reference prompts, 256 tokens, ignore_eos, greedy and temperature 1 (seeded); decode tok/s =
               (tokens - 1) / (last token - first token), measured client-side over the stream
  ttft         random-token prompts of 1k / 8k / 30k tokens, max_tokens 1, a fresh prompt per run (no prefix cache)
  concurrency  k = 1, 2, 4 simultaneous 256-token requests, aggregate output tok/s
  gsm8k        200 test questions, 5-shot "Question: / Answer:" prompts from the train split, greedy, stop at
               "Question", last number vs the label (the protocol of SGLang's few_shot_gsm8k)
Prints and writes one JSON object.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import threading
import time
import urllib.request

import requests


def stream(base, model, prompt, max_tokens, temperature=0.0, seed=None, stop=None, ignore_eos=True):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "ignore_eos": ignore_eos, "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    if stop:
        body["stop"] = stop
    t0 = time.perf_counter()
    first = last = None
    n_chunks, usage, text = 0, None, []
    with requests.post(base + "/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    n_chunks += 1
                    text.append(ch["text"])
    n = (usage or {}).get("completion_tokens") or n_chunks
    return {"ttft": (first or time.perf_counter()) - t0, "decode_s": (last or 0) - (first or 0), "tokens": n,
            "text": "".join(text)}


def decode_rate(r):
    return (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] > 0 and r["tokens"] > 1 else 0.0


def bench_decode(base, model, prompts, temperature):
    out = []
    for i, ids in enumerate(prompts):
        r = stream(base, model, ids, 256, temperature, seed=100 + i if temperature > 0 else None)
        out.append({"prompt_tokens": len(ids), "tokens": r["tokens"], "tok_s": round(decode_rate(r), 1)})
    tot_t = sum(o["tokens"] - 1 for o in out)
    tot_s = sum((o["tokens"] - 1) / o["tok_s"] for o in out if o["tok_s"] > 0)
    return {"per_prompt": out, "overall_tok_s": round(tot_t / tot_s, 1)}


def bench_ttft(base, model, lengths, reps=2):
    rng = random.Random(0)
    res = {}
    for n in lengths:
        times = []
        for _ in range(reps + 1):  # first run warms up
            ids = [rng.randrange(1000, 60000) for _ in range(n)]
            times.append(stream(base, model, ids, 1)["ttft"])
        t = statistics.median(times[1:])
        res[str(n)] = {"ttft_ms": round(t * 1000, 1), "prefill_tok_s": round(n / t)}
    return res


def bench_concurrency(base, model, prompts, ks=(1, 2, 4), reps=2):
    res = {}
    for k in ks:
        best = 0.0
        for rep in range(reps):
            outs = [None] * k
            th = [threading.Thread(target=lambda i=i: outs.__setitem__(
                i, stream(base, model, prompts[(i + rep) % len(prompts)], 256))) for i in range(k)]
            t0 = time.perf_counter()
            [x.start() for x in th]
            [x.join() for x in th]
            dt = time.perf_counter() - t0
            best = max(best, sum(o["tokens"] for o in outs) / dt)
        res[str(k)] = {"aggregate_tok_s": round(best, 1)}
    return res


GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"


def bench_gsm8k(base, model, n=200):
    def load(split):
        with urllib.request.urlopen(GSM + f"{split}.jsonl") as f:
            return [json.loads(x) for x in f.read().decode().splitlines() if x.strip()]

    train, test = load("train"), load("test")
    shots = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train[:5])
    correct = 0
    t0 = time.perf_counter()
    for e in test[:n]:
        r = stream(base, model, shots + f"Question: {e['question']}\nAnswer:", 320, stop=["Question"], ignore_eos=False)
        nums = re.findall(r"-?\d[\d,]*\.?\d*", r["text"].split("Question")[0])
        p = nums[-1].replace(",", "").rstrip(".") if nums else None
        correct += int(p == e["answer"].split("####")[-1].strip().replace(",", ""))
    return {"n": n, "accuracy": round(correct / n, 3), "seconds": round(time.perf_counter() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="tests/reference/sglang_ref.json")
    ap.add_argument("--ttft", default="1024,4096,8192,30000")
    ap.add_argument("--conc", default="1,2,4", help="concurrency levels")
    ap.add_argument("--skip", default="", help="comma list of sections to skip")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = json.load(open(a.prompts))
    items = items["items"] if isinstance(items, dict) else items
    prompts = [it["input_ids"] for it in items[:6]]
    skip = set(a.skip.split(",")) - {""}
    res = {"model": a.model}
    stream(a.base, a.model, prompts[0], 32)  # warm-up
    if "decode" not in skip:
        res["decode_greedy"] = bench_decode(a.base, a.model, prompts, 0.0)
        res["decode_t1"] = bench_decode(a.base, a.model, prompts, 1.0)
    if "ttft" not in skip:
        res["ttft"] = bench_ttft(a.base, a.model, [int(x) for x in a.ttft.split(",")])
    if "concurrency" not in skip:
        res["concurrency"] = bench_concurrency(a.base, a.model, prompts, [int(x) for x in a.conc.split(",")])
    if "gsm8k" not in skip:
        res["gsm8k"] = bench_gsm8k(a.base, a.model)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
