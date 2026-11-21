"""CPU-only: download thinkingmachines/Inkling-2-Mini into the Modal Volume `inkling2-mini-weights`.

Usage: modal run modal/download_weights.py
Weights never touch the local machine.
"""
import modal

REPO = "thinkingmachines/Inkling-2-Mini"
vol = modal.Volume.from_name("inkling2-mini-weights", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]>=0.26")
app = modal.App("inkling2-mini-download", image=image)


@app.function(volumes={"/weights": vol}, timeout=3600, cpu=4, memory=8192,
              env={"HF_HUB_ENABLE_HF_TRANSFER": "1"})
def download():
    import os
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, local_dir=f"/weights/{REPO.split('/')[-1]}")
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            total += sz
            print(f"{sz/1e6:12.1f} MB  {os.path.relpath(p, path)}")
    print(f"TOTAL {total/1e9:.2f} GB at {path}")
    vol.commit()


@app.local_entrypoint()
def main():
    download.remote()
