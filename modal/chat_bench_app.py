"""Same-client chat benchmark of this engine's server on an RTX PRO 6000 (Modal).

    modal run modal/chat_bench_app.py --tag ab --env "INKLING_SINGLE_USER=0;"   # without, then with the one-request path
    modal run modal/chat_bench_app.py --tag quick --skip ttft,long --conc 1

The server and bench/chat_bench.py run in the same container; results land in results/chat_<tag>.json.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import modal

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine_app import COMMON, WEIGHTS, image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("inkling2-chat-bench", image=image.add_local_file(str(ROOT / "modal/engine_app.py"), "/root/engine_app.py"))


@app.function(**{**COMMON, "timeout": 4 * 3600}, memory=65536)
def bench_many(envs: list, server_args: list, bench_args: list) -> list:
    """One server per environment, one after another in this container (a fair A/B)."""
    return [_bench(e, server_args, bench_args) for e in envs]


def _bench(env: dict, server_args: list, bench_args: list) -> dict:
    import requests

    e = dict(os.environ, **env)
    srv = subprocess.Popen([sys.executable, "-m", "inkling_engine.server", "--model", WEIGHTS, "--port", "8000",
                            "--no-vision", *server_args], env=e, cwd="/root")
    t0 = time.time()
    while True:
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=2).ok:
                break
        except Exception:  # noqa: BLE001
            pass
        if srv.poll() is not None:
            raise SystemExit("server exited")
        time.sleep(2)
    print(f"server up in {time.time() - t0:.0f} s", flush=True)
    out = "/tmp/chat.json"
    subprocess.check_call([sys.executable, "/root/bench/chat_bench.py", "--base", "http://127.0.0.1:8000/v1",
                           "--model", "inkling-2-mini", "--out", out, *bench_args])
    srv.terminate()
    srv.wait()
    return json.load(open(out))


@app.local_entrypoint()
def main(tag: str = "ours", env: str = "", server: str = "", skip: str = "", conc: str = "1,2,4"):
    """env: environments separated by ';', variables by ',' (e.g. "INKLING_SS=0;" runs Triton, then defaults)."""
    envs = [dict(kv.split("=", 1) for kv in grp.split(",") if kv) for grp in env.split(";")]
    args = ["--conc", conc] + (["--skip", skip] if skip else [])
    results = bench_many.remote(envs, server.split() if server else [], args)
    for i, (envd, res) in enumerate(zip(envs, results)):
        res["env"] = envd
        p = ROOT / "results" / f"chat_{tag}{'' if len(envs) == 1 else f'_{i}'}.json"
        p.write_text(json.dumps(res, indent=1))
        print("==", envd)
        for k, v in res.get("mtbench", {}).items():
            print(" ", k, v.get("tok_s"), "per-request decode", v.get("per_request_decode_tok_s_median"), "ttft",
                  v.get("ttft_ms_median"))
        print("  ttft", {k: v["ttft_ms"] for k, v in res.get("ttft", {}).items()}, "long", res.get("long_decode"))
