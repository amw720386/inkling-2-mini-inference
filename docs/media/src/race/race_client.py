"""Race capture client: streams /v1/completions with token-id prompts and records every SSE chunk's arrival time.

    python race_client.py --base http://127.0.0.1:8000/v1 --model NAME --prompts race_prompts.json --out raw.json

Same request shape as bench/serve_bench.py (token ids, greedy, DSpark on in the server). For each prompt it keeps
t_send, and per chunk (t since send, text). Optional --split adds an untimed non-stream request with logprobs=0 to get
the token strings with special markers (thinking/answer split); timing runs never ask for logprobs.
"""
import argparse, json, time
import requests


def stream(base, model, ids, max_tokens, ignore_eos):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "ignore_eos": ignore_eos, "stream_options": {"include_usage": True}}
    chunks, usage, fin = [], None, None
    t0 = time.perf_counter()
    with requests.post(base + "/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        # chunk_size=None: yield each chunked-transfer piece as it arrives (the default 512 batches events)
        for line in r.iter_lines(chunk_size=None):
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            now = time.perf_counter() - t0
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    fin = ch["finish_reason"]
                if ch.get("text"):
                    chunks.append([round(now, 6), ch["text"]])
    return {"chunks": chunks, "usage": usage, "finish_reason": fin, "t_end": round(time.perf_counter() - t0, 6)}


def split_tokens(base, model, ids, max_tokens):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 0,
            "skip_special_tokens": False}
    r = requests.post(base + "/completions", json=body, timeout=600).json()
    ch = r["choices"][0]
    return {"text": ch.get("text"), "tokens": (ch.get("logprobs") or {}).get("tokens"), "usage": r.get("usage")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--split", action="store_true")
    a = ap.parse_args()
    P = json.load(open(a.prompts))
    res = {"model": a.model, "runs": []}
    stream(a.base, a.model, P[0]["input_ids"], 32, True)  # warm-up, as bench/serve_bench.py
    stream(a.base, a.model, P[-1]["input_ids"], 64, False)
    for rep in range(a.reps):
        for p in P:
            r = stream(a.base, a.model, p["input_ids"], p["max_tokens"], p["ignore_eos"])
            res["runs"].append({"id": p["id"], "rep": rep, **r})
            print(p["id"], rep, len(r["chunks"]), (r["usage"] or {}).get("completion_tokens"), r["t_end"], flush=True)
    if a.split:
        res["split"] = {}
        for p in P:
            try:
                res["split"][p["id"]] = split_tokens(a.base, a.model, p["input_ids"], p["max_tokens"])
            except Exception as e:  # noqa: BLE001
                res["split"][p["id"]] = {"error": str(e)[:300]}
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
