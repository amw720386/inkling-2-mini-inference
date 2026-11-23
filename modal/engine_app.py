"""Modal entry points for engine development on an sm_120 GPU.

The target is an RTX PRO 6000 Blackwell (GB202 / sm_120, 96 GB), used within a 32 GB budget.

  modal run modal/engine_app.py::parity          # engine vs transformers logits
  modal run modal/engine_app.py::pytest_gpu --args "-k moe"   # GPU unit tests
"""

from __future__ import annotations

import os

import modal

WEIGHTS = "/weights/Inkling-2-Mini"
TRANSFORMERS_SHA = "2f1bc5e321b9a736200a590d6e7743ccdeb60772"

vol = modal.Volume.from_name("inkling2-mini-weights")
cache_vol = modal.Volume.from_name("inkling2-engine-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "safetensors>=0.5",
        "numpy",
        "pytest",
        "tokenizers>=0.22",
        "tiktoken",
        "jinja2",
        "accelerate",
        "fastapi",
        "pillow",
        "uvicorn",
        "requests",
        "xgrammar==0.2.8",
        f"transformers @ git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_SHA}",
    )
)
# Triton 3.6 (newer than torch 2.9.1's pinned 3.5.1): NVFP4 tl.dot_scaled on sm_120 for fp4_prefill.
# INKLING_TRITON=<version> tries another release.
image = image.pip_install(f"triton=={os.environ.get('INKLING_TRITON', '3.6.0')}")
image = (
    image.env({"TRITON_CACHE_DIR": "/cache/triton", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("inkling_engine", ignore=["**/__pycache__", "**/*.pyc"])  # also data/ files
    .add_local_dir("tests", "/root/tests", ignore=["**/__pycache__"])
    .add_local_dir("bench", "/root/bench", ignore=["**/__pycache__"])
    .add_local_dir("tools", "/root/tools", ignore=["**/__pycache__"])
)
# `parity` looks scripts up by file name in these folders
SCRIPT_DIRS = ["/root/tests/gpu", "/root/bench", "/root/bench/experiments", "/root/tools"]

app = modal.App("inkling2-engine", image=image)
GPU = "RTX-PRO-6000"
COMMON = dict(gpu=GPU, volumes={"/weights": vol, "/cache": cache_vol}, timeout=3600)


@app.function(**COMMON, memory=65536)
def run_parity(args: list[str]):
    import subprocess
    import sys

    import os

    env = dict(os.environ)
    while args and "=" in args[0] and not args[0].endswith(".py"):  # KEY=VALUE environment overrides
        k, v = args[0].split("=", 1)
        env[k] = v
        args = args[1:]
    name = args[0].split("/")[-1] if args and args[0].endswith(".py") else "parity.py"
    script = next((f"{d}/{name}" for d in SCRIPT_DIRS if os.path.exists(f"{d}/{name}")), f"/root/tests/gpu/{name}")
    rest = args[1:] if args and args[0].endswith(".py") else args
    return subprocess.call([sys.executable, script, WEIGHTS, *rest], env=env)


@app.function(**COMMON, memory=65536)
def run_pytest(args: list[str]):
    import subprocess
    import sys

    return subprocess.call([sys.executable, "-m", "pytest", "-x", "-q", "/root/tests/gpu", *args],
                           env={"WEIGHTS": WEIGHTS, **__import__("os").environ})


@app.local_entrypoint()
def parity(args: str = ""):
    raise SystemExit(run_parity.remote(args.split()))


@app.local_entrypoint()
def pytest_gpu(args: str = ""):
    raise SystemExit(run_pytest.remote(args.split()))
