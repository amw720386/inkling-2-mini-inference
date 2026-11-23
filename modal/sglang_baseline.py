"""SGLang reference baseline for Inkling-2-Mini on a Modal RTX PRO 6000 (GB202 / sm_120),
memory-capped to 32 GB.

Usage:
  modal run modal/sglang_baseline.py --mode probe --config balanced --mem-frac 0.26
  modal run modal/sglang_baseline.py --mode full  --config balanced --mem-frac 0.26
  modal run modal/sglang_baseline.py --mode full  --config dspark   --mem-frac 0.26

All results are written to the `inkling2-mini-weights` volume under /weights/results/ and
returned to the local entrypoint, which saves small JSON files locally.

Numbers produced here are RTX PRO 6000 results, memory-capped to 32 GB.
"""
import json
import os
import pathlib

import modal

MODEL_DIR = "/weights/Inkling-2-Mini"
RESULTS_DIR = "/weights/results"
PORT = 30000
IMAGE_TAG = "lmsysorg/sglang:inkling-2-mini"

vol = modal.Volume.from_name("inkling2-mini-weights")
image = modal.Image.from_registry(IMAGE_TAG)
app = modal.App("inkling2-mini-sglang-baseline", image=image)

COMMON_FLAGS = [
    "--model-path", MODEL_DIR,
    "--tp", "1", "--ep", "1",
    "--attention-backend", "fa4",
    "--dtype", "bfloat16",
    "--page-size", "128",
    "--grammar-backend", "xgrammar",
    "--enable-multimodal",
    "--reasoning-parser", "inkling",
    "--tool-call-parser", "inkling2",
    "--host", "0.0.0.0", "--port", str(PORT),
]
CONFIGS = {
    "balanced": {
        "flags": ["--context-length", "65536"],
        "env": {},
        "ctx": 65536,
    },
    "dspark": {
        "flags": ["--speculative-algorithm", "DSPARK", "--speculative-num-draft-tokens", "8",
                  "--context-length", "32768", "--max-running-requests", "4"],
        "env": {
            "SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL": "1",
            "SGLANG_OPT_DSV41_DSPARK_MID_FOLD": "1",
            "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE": "1",
            "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS": "1",
        },
        "ctx": 32768,
    },
}

CHAT_REF_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that checks whether a number is prime.",
    "Explain why the sky is blue to a 10-year-old.",
    "Translate 'Good morning, how are you?' into Spanish, French, and German.",
]
RAW_REF_PROMPTS = [
    "The quick brown fox",
    "def fibonacci(n):\n",
    "In 1905, Albert Einstein published",
    "1, 1, 2, 3, 5, 8, 13,",
]


# ----------------------------------------------------------------------------- helpers
class MemSampler:
    """Samples nvidia-smi memory.used (MiB) every 200 ms; tracks per-phase peaks."""

    def __init__(self):
        import subprocess
        import threading
        self.peak = 0
        self.phase = "boot"
        self.phase_peak = {}
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        for line in self.proc.stdout:
            try:
                v = int(line.strip())
            except ValueError:
                continue
            self.peak = max(self.peak, v)
            self.phase_peak[self.phase] = max(self.phase_peak.get(self.phase, 0), v)

    def set_phase(self, p):
        self.phase = p

    def stop(self):
        self.proc.terminate()


def start_server(cfg_name, mem_frac, log_path):
    import subprocess
    cfg = CONFIGS[cfg_name]
    cmd = ["sglang", "serve"] + COMMON_FLAGS + cfg["flags"] + ["--mem-fraction-static", str(mem_frac)]
    env = dict(os.environ)
    env.update(cfg["env"])
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, cmd


def wait_ready(proc, log_path, timeout=1800):
    import time
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited rc=%s\n%s" % (proc.returncode, tail(log_path, 120)))
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health_generate", timeout=30)
            if r.status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("server not ready in time\n" + tail(log_path, 120))


def tail(path, n):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except Exception as e:  # noqa
        return str(e)


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def chat_ids(tok, user_text):
    """Chat template with reasoning effort 'none' => 'Thinking effort level: 0'."""
    msgs = [{"role": "user", "content": user_text}]
    try:
        ids = tok.apply_chat_template(msgs, tokenize=True, reasoning_effort="none")
        if hasattr(ids, "keys"):  # BatchEncoding (UserDict) in recent transformers
            ids = ids["input_ids"]
        ids = [int(x) for x in ids]
        txt = tok.apply_chat_template(msgs, tokenize=False, reasoning_effort="none")
        return list(ids), txt
    except Exception as e:  # fallback: manual render of the same template
        print("apply_chat_template failed, manual render:", e)
        txt = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
               f"<|message_user|><|content_text|>{user_text}<|end_message|>")
        return tok.encode(txt, add_special_tokens=False), txt


def generate(payload, timeout=300):
    import requests
    r = requests.post(f"http://127.0.0.1:{PORT}/generate", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"/generate {r.status_code}: {r.text[:2000]}")
    return r.json()


def stream_generate(input_ids, max_new_tokens, ignore_eos=True):
    """Returns dict with ttft, decode tok/s, n tokens, meta."""
    import time
    import requests
    payload = {"input_ids": input_ids, "stream": True,
               "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens,
                                   "ignore_eos": ignore_eos}}
    t0 = time.perf_counter()
    t_first = None
    last_meta = None
    with requests.post(f"http://127.0.0.1:{PORT}/generate", json=payload, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(f"/generate {r.status_code}: {r.text[:2000]}")
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            meta = obj.get("meta_info", {})
            if t_first is None and meta.get("completion_tokens", 0) >= 1:
                t_first = time.perf_counter()
            last_meta = meta
    t_end = time.perf_counter()
    n = last_meta.get("completion_tokens", 0)
    res = {"prompt_tokens": last_meta.get("prompt_tokens"), "completion_tokens": n,
           "ttft_s": t_first - t0, "e2e_s": t_end - t0,
           "decode_tok_s": (n - 1) / (t_end - t_first) if n > 1 and t_end > t_first else None,
           "cached_tokens": last_meta.get("cached_tokens")}
    for k in ("spec_verify_ct", "spec_accept_length", "spec_accept_rate", "spec_accept_token_num",
              "spec_draft_token_num"):
        if k in last_meta:
            res[k] = last_meta[k]
    if res.get("spec_verify_ct"):
        res["avg_tokens_per_verify"] = n / res["spec_verify_ct"]
    return res


def long_token_pool(tok, n_needed):
    base = open("/sgl-workspace/sglang/python/sglang/test/long_prompt.txt", errors="replace").read()
    ids = tok.encode(base, add_special_tokens=False)
    pool = []
    while len(pool) < n_needed:
        pool.extend(ids)
    return pool[:n_needed]


def unique_prefix(tok, i):
    import random
    rnd = random.Random(1000 + i)
    words = " ".join(rnd.choice(["alpha", "river", "stone", "cloud", "maple", "orbit", "violet",
                                 "ember", "quartz", "harbor", "falcon", "meadow"]) for _ in range(24))
    return tok.encode(f"Document {i}: {words}.\n", add_special_tokens=False)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def run_cmd(cmd, timeout=3600):
    import subprocess
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                       executable="/bin/bash")
    return p.returncode, p.stdout[-20000:], p.stderr[-8000:]


# ----------------------------------------------------------------------------- benchmarks
def bench_decode(tok, ctx_len):
    out = {}
    short_ids, _ = chat_ids(tok, (
        "Write a long, detailed essay about the history of the printing press, covering its "
        "invention, its spread across Europe, its effect on literacy, religion, science and "
        "politics, and its legacy in the digital age. Use several paragraphs with concrete "
        "examples, names and dates, and finish with a short conclusion."))
    pool = long_token_pool(tok, 8000)
    long_text = tok.decode(pool)
    long_ids, _ = chat_ids(tok, long_text + "\n\nSummarize the document above in detail, section by section.")
    for name, ids in (("short", short_ids), ("long_8k", long_ids)):
        stream_generate(ids, 64)  # warmup
        runs = [stream_generate(ids, 512) for _ in range(3)]
        out[name] = {"prompt_tokens": len(ids), "runs": runs,
                     "median_decode_tok_s": median([r["decode_tok_s"] for r in runs])}
        print("decode", name, out[name]["prompt_tokens"], out[name]["median_decode_tok_s"], flush=True)
    return out


def bench_prefill(tok, ctx_len, targets=None):
    out = {}
    targets = targets or {"8k": 8192, "32k": 32000}
    for name, n in targets.items():
        runs = []
        for i in range(4):  # first run = warmup; unique prefix defeats radix cache
            pre = unique_prefix(tok, hash((name, i)) % 100000)
            ids = pre + long_token_pool(tok, n - len(pre))
            r = stream_generate(ids, 1)
            if r["prompt_tokens"] != len(ids):
                raise RuntimeError(f"request rejected/truncated: sent {len(ids)} got {r['prompt_tokens']}")
            r["prefill_tok_s"] = len(ids) / r["ttft_s"]
            runs.append(r)
        meas = runs[1:]
        out[name] = {"prompt_tokens": n, "warmup": runs[0], "runs": meas,
                     "median_ttft_s": median([r["ttft_s"] for r in meas]),
                     "median_prefill_tok_s": median([r["prefill_tok_s"] for r in meas])}
        print("prefill", name, out[name]["median_ttft_s"], out[name]["median_prefill_tok_s"], flush=True)
    return out


def bench_throughput(cfg_name):
    out = {}
    for conc, nprompts in ((4, 32), (16, 64)):
        of = f"/tmp/bs_{cfg_name}_c{conc}.jsonl"
        cmd = (f"python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port {PORT} "
               f"--dataset-name random --random-input-len 1024 --random-output-len 512 "
               f"--random-range-ratio 1.0 --num-prompts {nprompts} --max-concurrency {conc} "
               f"--seed 1 --output-file {of}")
        rc, so, se = run_cmd(cmd, timeout=2400)
        res = {"cmd": cmd, "rc": rc, "stdout_tail": so[-6000:], "stderr_tail": se[-3000:]}
        if rc == 0 and os.path.exists(of):
            lines = [l for l in open(of) if l.strip()]
            if lines:
                d = json.loads(lines[-1])
                res["result"] = {k: v for k, v in d.items() if not isinstance(v, list)}
        out[f"c{conc}"] = res
        print("throughput", conc, rc, (res.get("result") or {}).get("output_throughput"), flush=True)
    return out


def bench_gsm8k():
    cmd = (f"python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --host 127.0.0.1 "
           f"--port {PORT}")
    rc, so, se = run_cmd(cmd, timeout=2400)
    print("gsm8k", rc, so[-600:], flush=True)
    return {"cmd": cmd, "rc": rc, "stdout_tail": so[-4000:], "stderr_tail": se[-3000:]}


def reference_outputs(tok, with_logprobs=True):
    items = []
    for p in CHAT_REF_PROMPTS:
        ids, txt = chat_ids(tok, p)
        items.append({"kind": "chat", "prompt": p, "rendered_prompt": txt, "input_ids": ids})
    for p in RAW_REF_PROMPTS:
        ids = tok.encode(p, add_special_tokens=False)
        items.append({"kind": "raw", "prompt": p, "rendered_prompt": p, "input_ids": ids})
    for it in items:
        payload = {"input_ids": it["input_ids"],
                   "sampling_params": {"temperature": 0.0, "max_new_tokens": 64, "ignore_eos": True}}
        if with_logprobs:
            payload.update({"return_logprob": True, "top_logprobs_num": 20, "logprob_start_len": 0,
                            "return_text_in_logprobs": False})
        r = generate(payload)
        m = r["meta_info"]
        if with_logprobs:
            # output_token_logprobs: [[logprob, token_id, None], ...]
            it["output_ids"] = [t[1] for t in m["output_token_logprobs"]]
            it["output_logprobs"] = [t[0] for t in m["output_token_logprobs"]]
            it["output_top_logprobs"] = [[[e[1], e[0]] for e in pos] for pos in m["output_top_logprobs"]]
            it["input_token_logprobs"] = [t[0] for t in m.get("input_token_logprobs", [])]
            it["input_top_logprobs"] = [[[e[1], e[0]] for e in (pos or [])]
                                        for pos in m.get("input_top_logprobs", [])]
        else:
            it["output_ids"] = r.get("output_ids") or tok.encode(r["text"], add_special_tokens=False)
        it["output_text"] = r["text"]
    # determinism re-check (ids only)
    rerun_match = []
    for it in items:
        r = generate({"input_ids": it["input_ids"],
                      "sampling_params": {"temperature": 0.0, "max_new_tokens": 64, "ignore_eos": True}})
        ids2 = r.get("output_ids")
        rerun_match.append(ids2 == it["output_ids"] if ids2 is not None else r["text"] == it["output_text"])
    return {"items": items, "rerun_identical": rerun_match}


# ----------------------------------------------------------------------------- modal entry
@app.function(gpu="RTX-PRO-6000", volumes={"/weights": vol}, timeout=3 * 3600, cpu=8, memory=65536)
def run(mode: str, config: str, mem_frac: float):
    import subprocess
    import time
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {"mode": mode, "config": config, "mem_frac": mem_frac, "image": IMAGE_TAG,
              "image_env": {k: os.environ.get(k) for k in ("SGLANG_IMAGE_TAG", "SGLANG_BUILD_COMMIT")}}
    result["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"],
        capture_output=True, text=True).stdout
    print(result["nvidia_smi"], flush=True)
    result["versions"] = run_cmd("python3 -c 'import sglang,torch;print(torch.__version__, torch.version.cuda);"
                                 "import importlib.metadata as m;print(m.version(\"sglang\"))'")[1]
    mem = MemSampler()
    log_path = f"/tmp/server_{config}.log"
    proc, cmd = start_server(config, mem_frac, log_path)
    result["server_cmd"] = " ".join(f"{k}={v}" for k, v in CONFIGS[config]["env"].items()) + " " + " ".join(cmd)
    try:
        result["startup_s"] = wait_ready(proc, log_path)
        time.sleep(3)
        result["mem_after_ready_mib"] = mem.peak
        print("ready in", result["startup_s"], "mem", mem.peak, flush=True)
        if mode != "probe" and mem.peak > 32000:
            raise RuntimeError(f"memory after ready {mem.peak} MiB exceeds 32 GB cap; lower --mem-frac")
        tok = get_tokenizer()
        if mode == "probe":
            mem.set_phase("probe")
            result["probe_decode"] = stream_generate(chat_ids(tok, "Write a long story about a dragon.")[0], 256)
            pre = unique_prefix(tok, 1)
            ids = pre + long_token_pool(tok, 32000 - len(pre))
            result["probe_prefill_32k"] = stream_generate(ids, 8)
            print(json.dumps(result["probe_decode"]), json.dumps(result["probe_prefill_32k"]), flush=True)
            result["help_bench_serving"] = run_cmd("python3 -m sglang.benchmark.serving --help | grep -A3 -E 'dataset-name|random'")[1][-3000:]
            result["help_gsm8k"] = run_cmd("python3 -m sglang.test.few_shot_gsm8k --help")[1][-3000:]
        elif mode == "prefill31k":  # DSpark at 32 GB cap admits at most ~31.7k tokens
            mem.set_phase("prefill")
            result["prefill"] = bench_prefill(tok, CONFIGS[config]["ctx"], {"31k": 31000})
        elif mode == "extra":  # throughput + gsm8k only
            mem.set_phase("throughput")
            result["throughput"] = bench_throughput(config)
            mem.set_phase("gsm8k")
            result["gsm8k"] = bench_gsm8k()
        else:
            if config == "balanced":
                mem.set_phase("reference")
                result["reference"] = reference_outputs(tok, with_logprobs=True)
            else:
                mem.set_phase("reference")
                try:
                    result["reference_ids_only"] = reference_outputs(tok, with_logprobs=False)
                except Exception as e:
                    result["reference_ids_only"] = {"error": repr(e)}
            mem.set_phase("decode")
            result["decode"] = bench_decode(tok, CONFIGS[config]["ctx"])
            mem.set_phase("prefill")
            result["prefill"] = bench_prefill(tok, CONFIGS[config]["ctx"])
            mem.set_phase("throughput")
            result["throughput"] = bench_throughput(config)
            mem.set_phase("gsm8k")
            result["gsm8k"] = bench_gsm8k()
    except Exception as e:
        import traceback
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result["peak_mem_mib"] = mem.peak
        result["phase_peak_mib"] = mem.phase_peak
        proc.terminate()
        try:
            proc.wait(60)
        except Exception:
            proc.kill()
        mem.stop()
        result["server_log_tail"] = tail(log_path, 200)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f"{RESULTS_DIR}/{mode}_{config}_{stamp}.json", "w") as f:
            json.dump(result, f)
        subprocess.run(f"cp {log_path} {RESULTS_DIR}/server_{mode}_{config}_{stamp}.log", shell=True)
        vol.commit()
    return result


@app.local_entrypoint()
def main(mode: str = "probe", config: str = "balanced", mem_frac: float = 0.26):
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results" / "sglang"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run.remote(mode, config, mem_frac)
    ref = res.get("reference")
    if ref:
        ref_path = pathlib.Path(__file__).resolve().parent.parent / "tests" / "reference" / "sglang_ref.json"
        ref_doc = {"model": "thinkingmachines/Inkling-2-Mini", "hf_revision": "ed40e7da625f594fc9c4990f5f66be24873c22d7",
                   "engine": "SGLang " + IMAGE_TAG, "server_cmd": res.get("server_cmd"),
                   "gpu": "RTX PRO 6000 (sm_120) proxy, memory-capped to 32 GB",
                   "sampling": {"temperature": 0.0, "max_new_tokens": 64, "ignore_eos": True,
                                "top_logprobs_num": 20, "logprob_start_len": 0},
                   "notes": "top logprobs are [token_id, logprob] pairs; chat prompts use the HF chat "
                            "template with reasoning_effort='none' (Thinking effort level: 0); raw "
                            "prompts are tokenizer.encode(text, add_special_tokens=False).",
                   **ref}
        ref_path.write_text(json.dumps(ref_doc))
        print("wrote", ref_path)
    slim = {k: v for k, v in res.items() if k not in ("reference",)}
    (out_dir / f"{mode}_{config}.json").write_text(json.dumps(slim, indent=1))
    printable = {k: v for k, v in slim.items() if k not in ("server_log_tail", "reference_ids_only")}
    print(json.dumps(printable, indent=1)[-15000:])
    if res.get("error"):
        print("SERVER LOG TAIL:\n", res.get("server_log_tail", "")[-8000:])
