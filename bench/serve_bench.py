"""One OpenAI-API benchmark for every server (this engine, SGLang, vLLM), run on the same GPU.

    python bench/serve_bench.py --base http://127.0.0.1:8000/v1 --model <name> --prompts tests/reference/sglang_ref.json

Everything but gsm8k (which sends text) goes through /v1/completions with token-id prompts, so tokenization and chat templates cannot differ:
  decode       6 reference prompts, 256 tokens, ignore_eos, greedy and temperature 1 (seeded), one request at a time;
               tok/s = completion tokens / (request sent -> stream finished), median of --decode-reps passes
  ttft         random-token prompts of 1k / 4k / 8k / 30k tokens, max_tokens 1, a fresh prompt per run (no prefix cache)
  concurrency  k = 1, 2, 4 simultaneous 256-token requests, aggregate output tok/s, median of 6 prompt rotations
  heldout      24 MT-Bench prompts nothing was tuned on, natural stopping (at most 512 tokens), k = 1, 2, 4 in flight; also
               random1k: random 1k-token prompts, 1k tokens out (the workload of vLLM's Inkling-2-Mini pull request)
  gsm8k        200 test questions, 5-shot "Question: / Answer:" prompts from the train split, greedy, stop at
               "Question", last number vs the label (the protocol of SGLang's few_shot_gsm8k)
Rates never use the arrival times of text: a server may send nothing for a special token, so the first or last
visible chunk can be many tokens away from the first or last generated one. Prints and writes one JSON object.
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
    first = end = None
    n_chunks, usage, text = 0, None, []
    with requests.post(base + "/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=None):  # each piece as it arrives
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            end = time.perf_counter()
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    first = first or end
                    n_chunks += 1
                    text.append(ch["text"])
    n = (usage or {}).get("completion_tokens") or n_chunks
    return {"ttft": (first or time.perf_counter()) - t0, "total_s": (end or time.perf_counter()) - t0, "tokens": n,
            "text": "".join(text)}


def bench_decode(base, model, prompts, temperature, reps=3):
    runs, per = [], [[] for _ in prompts]
    for rep in range(reps):
        toks = secs = 0.0
        for i, ids in enumerate(prompts):
            r = stream(base, model, ids, 256, temperature, seed=100 + i + 10 * rep if temperature > 0 else None)
            per[i].append(round(r["tokens"] / r["total_s"], 1))
            toks, secs = toks + r["tokens"], secs + r["total_s"]
        runs.append(round(toks / secs, 1))
    return {"per_prompt": [{"prompt_tokens": len(p), "tok_s": x} for p, x in zip(prompts, per)], "runs": runs,
            "overall_tok_s": round(statistics.median(runs), 1)}


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


def bench_concurrency(base, model, prompts, ks=(1, 2, 4), reps=6):
    res = {}
    for k in ks:
        runs = []
        for rep in range(-1, reps):  # rep -1 warms up this batch size
            outs = [None] * k
            th = [threading.Thread(target=lambda i=i: outs.__setitem__(
                i, stream(base, model, prompts[(i + rep) % len(prompts)], 256))) for i in range(k)]
            t0 = time.perf_counter()
            [x.start() for x in th]
            [x.join() for x in th]
            dt = time.perf_counter() - t0
            if rep >= 0:
                runs.append(round(sum(o["tokens"] for o in outs) / dt, 1))
        res[str(k)] = {"aggregate_tok_s": round(statistics.median(runs), 1), "runs": runs}
    return res


def closed_loop(base, model, prompts, k, max_tokens, temperature, ignore_eos, seed0=None):
    """k workers take the prompts in order until all are done: (tokens, seconds, tokens per request)."""
    q, lock, out = list(enumerate(prompts)), threading.Lock(), [None] * len(prompts)

    def work():
        while True:
            with lock:
                if not q:
                    return
                i, ids = q.pop(0)
            out[i] = stream(base, model, ids, max_tokens, temperature,
                            seed=None if seed0 is None else seed0 + i, ignore_eos=ignore_eos)
    th = [threading.Thread(target=work) for _ in range(k)]
    t0 = time.perf_counter()
    [x.start() for x in th]
    [x.join() for x in th]
    return sum(o["tokens"] for o in out), time.perf_counter() - t0, [o["tokens"] for o in out]


def bench_heldout(base, model, prompts, ks, passes=3):
    """Chat prompts nothing was tuned on, natural stopping, k in flight: a warm-up pass, then `passes` measured ones."""
    res = {}
    for temp in (0.0, 1.0):
        for k in (ks if temp == 0.0 else (1,)):
            closed_loop(base, model, prompts, k, 512, temp, False, None if temp == 0 else 1000)
            runs = []
            for p in range(passes):
                n, dt, lens = closed_loop(base, model, prompts, k, 512, temp, False, None if temp == 0 else 2000 + 100 * p)
                runs.append(round(n / dt, 1))
            res[f"{'greedy' if temp == 0 else 't1'}_k{k}"] = {"tok_s": round(statistics.median(runs), 1), "runs": runs,
                                                             "tokens": n, "tokens_per_request": lens}
    return res


def bench_random(base, model, ks, n_in=1024, n_out=1024):
    """The workload of vLLM's Inkling-2-Mini pull request: random 1k-token prompts, 1k tokens out, ignore_eos, 8k requests."""
    rng, res = random.Random(1), {}
    mk = lambda n: [[rng.randrange(1000, 60000) for _ in range(n_in)] for _ in range(n)]  # noqa: E731
    closed_loop(base, model, mk(2), 1, n_out, 0.0, True)
    for k in ks:
        n, dt, _ = closed_loop(base, model, mk(8 * k), k, n_out, 0.0, True)
        res[str(k)] = {"tok_s": round(n / dt, 1)}
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
    ap.add_argument("--reps", type=int, default=6, help="concurrency repetitions per level (median reported)")
    ap.add_argument("--decode-reps", type=int, default=3, help="passes over the decode prompts (median reported)")
    ap.add_argument("--heldout", default="tests/reference/heldout_prompts.json")
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
        res["decode_greedy"] = bench_decode(a.base, a.model, prompts, 0.0, a.decode_reps)
        res["decode_t1"] = bench_decode(a.base, a.model, prompts, 1.0, a.decode_reps)
    if "ttft" not in skip:
        res["ttft"] = bench_ttft(a.base, a.model, [int(x) for x in a.ttft.split(",")])
    if "concurrency" not in skip:
        res["concurrency"] = bench_concurrency(a.base, a.model, prompts, [int(x) for x in a.conc.split(",")],
                                               reps=a.reps)
    if "heldout" not in skip:
        held = [it["input_ids"] for it in json.load(open(a.heldout))["items"]]
        res["heldout"] = bench_heldout(a.base, a.model, held, [int(x) for x in a.conc.split(",")])
        res["random1k"] = bench_random(a.base, a.model, [int(x) for x in a.conc.split(",")])
    if "gsm8k" not in skip:
        res["gsm8k"] = bench_gsm8k(a.base, a.model)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
