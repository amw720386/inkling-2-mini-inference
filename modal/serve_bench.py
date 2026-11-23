"""Same-GPU, same-client comparison: this engine vs SGLang vs vLLM serving Inkling-2-Mini.

    modal run modal/serve_bench.py --engine ours          # FP8 dense + FP4 prefill, DSpark (the server default)
    modal run modal/serve_bench.py --engine ours-bf16     # --no-dense-fp8 --no-fp4-prefill
    modal run modal/serve_bench.py --engine ours --tag 2  # another container: results/serve_ours-2.json
    modal run modal/serve_bench.py --engine ours --cap-gib 0   # this engine without the memory cap (0 = none)
    modal run modal/serve_bench.py --engine sglang        # lmsysorg/sglang:inkling-2-mini, DSpark recipe
    modal run modal/serve_bench.py --engine vllm          # vllm/vllm-openai nightly, DSpark recipe

Each server runs on an RTX PRO 6000 Blackwell (GB202 / sm_120) held to a 32 GB budget, and
`bench/serve_bench.py` measures it through /v1/completions. Results land in results/serve_<engine>.json locally.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import modal

MODEL = "/weights/Inkling-2-Mini"
ROOT = pathlib.Path(__file__).resolve().parent.parent
GPU = "RTX-PRO-6000"
vol = modal.Volume.from_name("inkling2-mini-weights")
cache_vol = modal.Volume.from_name("inkling2-engine-cache", create_if_missing=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine_app import image as engine_image  # noqa: E402

files = [(ROOT / "bench/serve_bench.py", "/root/bench_client.py"), (ROOT / "modal/engine_app.py", "/root/engine_app.py"),
         (ROOT / "tests/reference/sglang_ref.json", "/root/sglang_ref.json")]


def with_files(img):
    for src, dst in files:
        img = img.add_local_file(str(src), dst)
    return img


ours_image = with_files(engine_image)
sglang_image = with_files(modal.Image.from_registry("lmsysorg/sglang:inkling-2-mini"))
vllm_image = with_files(modal.Image.from_registry("vllm/vllm-openai:nightly")
                        .dockerfile_commands(["RUN ln -sf $(command -v python3) /usr/local/bin/python"])
                        .entrypoint([]))
app = modal.App("inkling2-serve-bench")
COMMON = dict(gpu=GPU, volumes={"/weights": vol, "/cache": cache_vol}, timeout=3 * 3600, memory=65536)

OURS_LAUNCH = """
import sys, torch
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, {cap} / total))
from inkling_engine.server import main
sys.argv = ["server"] + {args!r}
main()
"""


class MemPeak:
    """nvidia-smi memory.used peak (MiB) over the whole run."""

    def __init__(self):
        self.peak, self.stop = 0, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                self.peak = max(self.peak, int(out.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)


def serve_and_bench(cmd, env, port, model_name, version_cmd=None, client_args=()):
    import requests

    log = open("/tmp/server.log", "w")
    mem = MemPeak()
    proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            if requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2).ok:
                break
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is not None or time.time() - t0 > 3600:
            log.flush()
            raise RuntimeError("server did not start:\n" + open("/tmp/server.log").read()[-6000:])
        time.sleep(3)
    startup = time.time() - t0
    out = "/tmp/result.json"
    cp = subprocess.run([sys.executable, "/root/bench_client.py", "--base", f"http://127.0.0.1:{port}/v1",
                         "--model", model_name, "--prompts", "/root/sglang_ref.json", "--out", out, *client_args],
                        capture_output=True, text=True)
    rc = cp.returncode
    print(cp.stdout[-4000:], cp.stderr[-4000:], flush=True)
    proc.terminate()
    mem.stop = True
    log.flush()
    ok = rc == 0 and os.path.exists(out)
    res = json.load(open(out)) if ok else {
        "error": cp.stderr[-3000:] + "\n--- server log ---\n" + open("/tmp/server.log").read()[-4000:]}
    acc_lines = [x for x in open("/tmp/server.log", errors="replace").read().splitlines() if "accept" in x.lower()]
    res["server_accept_log"] = acc_lines[-60:]  # e.g. SGLang's per-batch speculative accept length
    res["startup_s"] = round(startup)
    res["peak_gpu_mib"] = mem.peak
    if version_cmd:
        res["version"] = subprocess.run(version_cmd, capture_output=True, text=True, shell=True).stdout.strip()[-300:]
    return res


@app.function(image=ours_image, **COMMON)
def run_ours(extra: list[str], client: tuple = (), cap_gib: float = 32.0):
    args = ["--model", MODEL, "--port", "8000"] + extra
    cmd = [sys.executable, "-c", OURS_LAUNCH.format(cap=int(cap_gib * 2**30) if cap_gib else 2**50, args=args)]
    return serve_and_bench(cmd, {"PYTHONPATH": "/root"}, 8000, "inkling-2-mini",
                           "git -C /root describe --always 2>/dev/null; python -c 'import torch,triton;print(torch.__version__, triton.__version__)'",
                           client_args=client)


SGLANG_FLAGS = ["--model-path", MODEL, "--tp", "1", "--ep", "1", "--attention-backend", "fa4", "--dtype", "bfloat16",
                "--page-size", "128", "--reasoning-parser", "inkling", "--host", "0.0.0.0", "--port", "30000",
                "--speculative-algorithm", "DSPARK", "--speculative-num-draft-tokens", "8",
                "--context-length", "32768", "--max-running-requests", "4", "--mem-fraction-static", "0.232"]
SGLANG_ENV = {"SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL": "1", "SGLANG_OPT_DSV41_DSPARK_MID_FOLD": "1",
              "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE": "1", "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS": "1"}


@app.function(image=sglang_image, **COMMON)
def run_sglang(extra: list[str], client: tuple = ()):
    cmd = ["python3", "-m", "sglang.launch_server"] + SGLANG_FLAGS + extra
    return serve_and_bench(cmd, SGLANG_ENV, 30000, MODEL, "python3 -c 'import sglang;print(sglang.__version__)'",
                           client_args=client)


VLLM_FLAGS = ["--tensor-parallel-size", "1", "--reasoning-parser", "inkling",
              "--speculative-config", '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}',
              # under 32 GB vLLM keeps ~7.3 GiB of KV (~6.9k tokens) after weights and graphs: context capped
              "--max-model-len", "6144", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.33",
              "--host", "0.0.0.0", "--port", "8000"]


@app.function(image=vllm_image, **COMMON)
def run_vllm(extra: list[str], client: tuple = ()):
    cmd = ["vllm", "serve", MODEL] + VLLM_FLAGS + extra
    return serve_and_bench(cmd, {}, 8000, MODEL, "python3 -c 'import vllm;print(vllm.__version__)'",
                           client_args=("--ttft", "1024,4096", *client))


@app.local_entrypoint()
def main(engine: str = "ours", extra: str = "", tag: str = "", client: str = "", cap_gib: float = 32.0):
    xs = extra.split() if extra else []
    cl = tuple(client.split()) if client else ()
    fn = {"ours": lambda: run_ours.remote(xs, cl, cap_gib), "ours-bf16": lambda: run_ours.remote(["--no-dense-fp8", "--no-fp4-prefill"] + xs, cl, cap_gib),
          "sglang": lambda: run_sglang.remote(xs, cl), "vllm": lambda: run_vllm.remote(xs, cl)}[engine]
    res = fn()
    out = ROOT / "results" / f"serve_{engine}{'-' + tag if tag else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "error"}, indent=1)[:4000])
    if "error" in res:
        print("ERROR:", res["error"][-3000:])
