"""CPU-only probe of the SGLang image and downloaded model metadata."""
import modal

vol = modal.Volume.from_name("inkling2-mini-weights")
image = modal.Image.from_registry("lmsysorg/sglang:inkling-2-mini")
app = modal.App("inkling2-mini-inspect", image=image)


@app.function(volumes={"/weights": vol}, timeout=900, cpu=2, memory=4096)
def inspect():
    import subprocess
    cmds = [
        "ls /sgl-workspace/sglang/python/sglang/benchmark /sgl-workspace/sglang/python/sglang/cli",
        "grep -rn 'def normalize_base_url' -A12 /sgl-workspace/sglang/python/sglang/",
        "grep -n 'add_argument' /sgl-workspace/sglang/python/sglang/test/few_shot_gsm8k.py",
        "grep -rln 'random-range-ratio' /sgl-workspace/sglang/python/sglang/ | head",
        "sglang --help 2>&1 | head -40",
    ]
    for c in cmds:
        print("=" * 20, c)
        print(subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash").stdout[-12000:])


@app.local_entrypoint()
def main():
    inspect.remote()
