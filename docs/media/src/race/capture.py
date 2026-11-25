"""Per-token race capture on Modal: one server boot per engine, the same client, the same token-id prompts.

From the repo root (weights in the volume of modal/download_weights.py):
    python docs/media/src/race/make_prompts.py --tokenizer /path/to/tokenizer.json
    modal run docs/media/src/race/capture.py --engines ours,sglang,vllm-fp8kv --outdir /tmp
    python docs/media/src/race/pick.py /tmp/race_raw_*.json
Reuses modal/serve_bench.py's images and its --whole-card flags (no memory cap, every engine's own recipe; vLLM with its
optimized flags). Each run writes race_raw_<engine>.json: per prompt,
the arrival time of every streamed chunk.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time
import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path.cwd() / "modal"))
import serve_bench as sb  # noqa: E402

extra = [(HERE / "race_client.py", "/root/race_client.py"), (pathlib.Path(os.environ.get("RACE_PROMPTS", HERE / "race_prompts.json")), "/root/race_prompts.json"),
         (pathlib.Path(sb.__file__), "/root/serve_bench.py")]


def add(img):
    for s, d in extra:
        img = img.add_local_file(str(s), d)
    return img


app = modal.App("inkling2-race-capture")
COMMON = dict(gpu=sb.GPU, volumes={"/weights": sb.vol, "/cache": sb.cache_vol}, timeout=50 * 60, memory=65536)


def model_dir():
    return sb.MODEL


def host_info():
    """GPU clocks/perf state, host id and the Triton cache size before the server starts (cold = 0 files)."""
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
    return {"nvidia_smi_q": sh("nvidia-smi -q -d CLOCK,PERFORMANCE | head -80"),
            "gpu_uuid": sh("nvidia-smi --query-gpu=uuid,pci.bus_id,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader").strip(),
            "hostname": sh("hostname").strip(), "modal_task": os.environ.get("MODAL_TASK_ID", ""),
            "triton_cache_files": int(sh("find /cache/triton -type f 2>/dev/null | wc -l").strip() or 0)}


def serve(cmd, env, port, split, version_cmd):
    import requests
    info0 = host_info()
    log = open("/tmp/server.log", "w")
    proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            if r.ok:
                name = r.json()["data"][0]["id"]
                break
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is not None or time.time() - t0 > 35 * 60:
            log.flush()
            return {"error": open("/tmp/server.log").read()[-6000:]}
        time.sleep(3)
    startup = time.time() - t0
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cp = subprocess.run([sys.executable, "/root/race_client.py", "--base", f"http://127.0.0.1:{port}/v1", "--model", name,
                         "--prompts", "/root/race_prompts.json", "--out", "/tmp/raw.json"] + (["--split"] if split else []),
                        capture_output=True, text=True)
    print(cp.stdout[-3000:], cp.stderr[-3000:], flush=True)
    proc.terminate()
    log.flush()
    if cp.returncode or not os.path.exists("/tmp/raw.json"):
        return {"error": cp.stderr[-3000:] + "\n--- server ---\n" + open("/tmp/server.log").read()[-4000:]}
    res = json.load(open("/tmp/raw.json"))
    res["version"] = subprocess.run(version_cmd, capture_output=True, text=True, shell=True).stdout.strip()[-300:]
    res.update(host=info0, smi_after=host_info()["nvidia_smi_q"], startup_s=round(startup), gpu=gpu, cmd=cmd, server_log_tail=open("/tmp/server.log").read()[-3000:])
    return res


def subst(flags, m):
    return [m if f == sb.MODEL else f for f in flags]


@app.function(image=add(sb.ours_image), **COMMON)
def run_ours(xs: list[str]):
    m = model_dir()
    args = ["--model", m, "--port", "8000"] + xs
    cmd = [sys.executable, "-c", sb.OURS_LAUNCH.format(cap=2**50, args=args)]  # 2**50: no cap
    r = serve(cmd, {"PYTHONPATH": "/root"}, 8000, True, "python -c 'import torch,triton;print(torch.__version__, triton.__version__)'")
    sb.cache_vol.commit()
    return r


@app.function(image=add(sb.sglang_image), **COMMON)
def run_sglang():
    m = model_dir()
    return serve(["python3", "-m", "sglang.launch_server"] + subst(sb.SGLANG_FLAGS, m) + ["--mem-fraction-static", "0.8"],
                 sb.SGLANG_ENV, 30000, True, "python3 -c 'import sglang;print(sglang.__version__)'")


@app.function(image=add(sb.vllm_image), **COMMON)
def run_vllm():
    m = model_dir()
    own = ["--kv-cache-dtype", "fp8", "--attention-backend", "FLASHINFER", "--gpu-memory-utilization", "0.9", "--max-model-len", "32768"]
    return serve(["vllm", "serve", m] + sb.VLLM_FLAGS + own, {}, 8000, True, "python3 -c 'import vllm;print(vllm.__version__)'")


@app.local_entrypoint()
def main(engines: str = "ours,sglang,vllm-fp8kv", outdir: str = "/tmp"):
    fns = {"ours": lambda: run_ours.spawn([]),
           "sglang": lambda: run_sglang.spawn(), "vllm-fp8kv": lambda: run_vllm.spawn()}
    calls = {e: fns[e]() for e in engines.split(",")}
    for e, c in calls.items():
        try:
            r = c.get()
        except Exception as ex:  # noqa: BLE001
            r = {"error": repr(ex)}
        p = pathlib.Path(outdir) / f"race_raw_{e}.json"
        p.write_text(json.dumps(r), encoding="utf-8")
        print(e, "ERROR" if "error" in r else "ok", p, (r.get("error") or "")[-1500:], flush=True)
