"""Same-client chat benchmark for any OpenAI-compatible server (this engine, ingine, SGLang, vLLM).

    python chat_bench.py --base http://127.0.0.1:8000/v1 --model NAME --out result.json

Everything goes through /v1/chat/completions, so it works with servers that have no /v1/completions:
  mtbench   the first three questions of each MT-Bench category (24), reasoning_effort "none", at most 512 tokens,
            natural stopping, k = 1, 2, 4 requests in flight (closed loop), greedy and temperature 1 (seeded);
            a warm-up pass, then the median of 3 passes; rate = completion tokens / wall time of the pass
  ttft      a fresh random-word prompt of about 1k / 4k / 16k / 30k tokens, max_tokens 1 (prompt_tokens as the server counts them)
  agentic   decode rate after a long prompt: 8k-token context then 256 tokens out, greedy
Each request's TTFT and decode rate (tokens after the first / time after the first chunk) are kept too.
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

import requests

MT = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


def chat(base, model, messages, max_tokens, temperature=0.0, seed=None, effort="none"):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}, "reasoning_effort": effort}
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    first = end = None
    usage = None
    with requests.post(base + "/chat/completions", json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=None):
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            end = time.perf_counter()
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    first = end
    n = (usage or {}).get("completion_tokens", 0)
    total = (end or time.perf_counter()) - t0
    return {"tokens": n, "prompt_tokens": (usage or {}).get("prompt_tokens"), "total_s": total,
            "ttft": (first or end or t0) - t0, "decode_tok_s": (n - 1) / (end - first) if first and end > first and n > 1 else None}


def closed_loop(base, model, msgs, k, max_tokens, temperature, seed0):
    q, lock, out = list(enumerate(msgs)), threading.Lock(), [None] * len(msgs)

    def work():
        while True:
            with lock:
                if not q:
                    return
                i, m = q.pop(0)
            out[i] = chat(base, model, m, max_tokens, temperature, None if seed0 is None else seed0 + i)
    th = [threading.Thread(target=work) for _ in range(k)]
    t0 = time.perf_counter()
    [x.start() for x in th]
    [x.join() for x in th]
    return sum(o["tokens"] for o in out), time.perf_counter() - t0, out


def mtbench(base, model, ks, passes=3):
    with urllib.request.urlopen(MT) as f:
        qs = [json.loads(x) for x in f.read().decode().splitlines() if x.strip()]
    seen, msgs = {}, []
    for q in qs:
        if seen.get(q["category"], 0) < 3:
            seen[q["category"]] = seen.get(q["category"], 0) + 1
            msgs.append([{"role": "user", "content": q["turns"][0]}])
    res = {}
    for temp in (0.0, 1.0):
        for k in (ks if temp == 0 else (1,)):
            closed_loop(base, model, msgs, k, 512, temp, None if temp == 0 else 1000)
            runs, per = [], None
            for p in range(passes):
                n, dt, out = closed_loop(base, model, msgs, k, 512, temp, None if temp == 0 else 2000 + 100 * p)
                runs.append(round(n / dt, 1))
                per = out
            dec = [o["decode_tok_s"] for o in per if o["decode_tok_s"]]
            res[f"{'greedy' if temp == 0 else 't1'}_k{k}"] = {
                "tok_s": round(statistics.median(runs), 1), "runs": runs,
                "tokens_per_request": [o["tokens"] for o in per],
                "per_request_decode_tok_s_median": round(statistics.median(dec), 1) if dec else None,
                "ttft_ms_median": round(1000 * statistics.median(o["ttft"] for o in per), 1)}
            print(k, temp, res[f"{'greedy' if temp == 0 else 't1'}_k{k}"]["tok_s"], flush=True)
    return res


WORDS = None


def words(n, rng):
    global WORDS
    if WORDS is None:
        WORDS = ("time year people way day man thing woman life child world school state family student group country "
                 "problem hand part place case week company system program question work government number night point "
                 "home water room mother area money story fact month lot right study book eye job word business issue "
                 "side kind head house service friend father power hour game line end member law car city community name").split()
    return " ".join(rng.choice(WORDS) for _ in range(n))


def ttft(base, model, lengths, reps=2):
    rng, res = random.Random(0), {}
    for n in lengths:
        ts, pt = [], None
        for _ in range(reps + 1):
            r = chat(base, model, [{"role": "user", "content": words(n, rng)}], 1)
            ts.append(r["total_s"])
            pt = r["prompt_tokens"]
        res[str(n)] = {"ttft_ms": round(1000 * statistics.median(ts[1:]), 1), "prompt_tokens": pt}
        print("ttft", n, res[str(n)], flush=True)
    return res


def long_context_decode(base, model, n=8000, out=256):
    rng = random.Random(7)
    txt = words(n, rng)
    m = [{"role": "user", "content": txt + "\n\nSummarize the themes above in a long essay."}]
    chat(base, model, m, 32)
    r = chat(base, model, [{"role": "user", "content": words(n, rng) + "\n\nSummarize the themes above in a long essay."}], out)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--ttft", default="1000,4000,16000,30000")
    ap.add_argument("--skip", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    res = {"model": a.model}
    chat(a.base, a.model, [{"role": "user", "content": "Hello"}], 16)
    if "mtbench" not in skip:
        res["mtbench"] = mtbench(a.base, a.model, [int(x) for x in a.conc.split(",")])
    if "ttft" not in skip:
        res["ttft"] = ttft(a.base, a.model, [int(x) for x in a.ttft.split(",")])
    if "long" not in skip:
        res["long_decode"] = long_context_decode(a.base, a.model)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res)[:3000])


if __name__ == "__main__":
    main()
