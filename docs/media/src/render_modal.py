"""Run render.sh on a Modal CPU container and write the results to docs/media.

    modal run docs/media/src/render_modal.py                # everything
    modal run docs/media/src/render_modal.py --what video   # video | gifs | diagrams
    modal run docs/media/src/render_modal.py --stills "scenes/endcard.html 3d/memory.html"  # PNG checks only
"""
import io
import pathlib
import subprocess
import tarfile

import modal

HERE = pathlib.Path(__file__).resolve().parent
CHROME_LIBS = ("libnss3 libatk-bridge2.0-0 libatk1.0-0 libgbm1 libasound2 libxkbcommon0 libxcomposite1 libxdamage1 "
               "libxrandr2 libxfixes3 libcups2 libpango-1.0-0 libcairo2 libdrm2 libdbus-1-3").split()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "nodejs", "npm", "ca-certificates", *CHROME_LIBS)
    .run_commands("npx --yes @puppeteer/browsers install chrome-headless-shell@stable --path /root/.cache/chs")
    .add_local_dir(str(HERE), "/root/src", ignore=["node_modules", "__pycache__"])
)
app = modal.App("inkling2-media-render")


def sh(cmd):
    cp = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(cp.stdout[-3000:], cp.stderr[-3000:], flush=True)
    if cp.returncode:
        raise SystemExit(f"failed: {cmd}")


def pack(directory):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for p in sorted(pathlib.Path(directory).rglob("*")):
            if p.is_file():
                t.add(p, arcname=str(p.relative_to(directory)))
    return buf.getvalue()


@app.function(image=image, cpu=16, memory=32768, timeout=3600)
def render(what: str, stills: str):
    sh("mkdir -p /root/media/src && cp -r /root/src/. /root/media/src/ && cd /root/media/src && npm ci --no-audit --no-fund")
    if stills:
        out = pathlib.Path("/root/stills")
        for page in stills.split():
            for size in ("1080x1080", "1920x1080"):
                w, h = size.split("x")
                d = out / f"{pathlib.Path(page.split('?')[0]).stem}-{size}"
                sh(f"cd /root/media/src && node render.js '{page}' {d} --w {w} --h {h} --stills 0.5,1.5,2.5,3.4,5,6.7")
        return pack(out)
    sh(f"mkdir -p /root/out && cd /root/media/src && WORK=/root/work bash render.sh {what}")
    sh("mkdir -p /root/pack && find /root/media -maxdepth 1 -type f "
       "\\( -name '*.mp4' -o -name '*.gif' -o -name '*.svg' -o -name '*.png' \\) -exec cp {} /root/pack/ \\;")
    return pack("/root/pack")


@app.local_entrypoint()
def main(what: str = "all", stills: str = "", out: str = ""):
    data = render.remote(what, stills)
    dest = pathlib.Path(out) if out else HERE.parent
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(dest, filter="data")
    print("wrote", len(data), "bytes of files to", dest)
