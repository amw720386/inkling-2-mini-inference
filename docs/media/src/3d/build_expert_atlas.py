"""Build expert-atlas.png: one small thumbnail per routed expert (24 layers x 128), from the real checkpoint.

What a thumbnail shows, for expert e in layer L:
  brightness  the expert's w13 (gate/up) amax (layers 0-19, `w13_weight_amax`) as log2, normalised within the layer
              (2nd-98th percentile), so experts with larger weights read brighter; layers 20-23 store no amax, so
              there it is the mean E8M0 block exponent over the rows read
  texture     the expert's w13 block scales as log2 magnitude, rows 0-15 of the [2304, cols] scale matrix (output
              channels) by its first 24 scale columns (16-input blocks with FP8 E4M3 scales in the NVFP4 layers 0-19,
              32-input blocks with E8M0 exponents in the MXFP8 layers 20-23), contrast stretched per thumbnail
  pixel = 0.6 * brightness + 0.4 * texture, quantised to 16 grey levels (a 4-bit PNG, < 1 MB).
Only those bytes are fetched, with HTTP range requests against the public checkpoint (Inkling-2-Mini, Apache-2.0):
3,072 + 20 reads, about 7 MB, not the 19.5 GB file. Standard library only.

  python build_expert_atlas.py            # writes expert-atlas.png next to this file
"""
import concurrent.futures as cf
import json
import math
import struct
import urllib.request
import zlib
from pathlib import Path

URL = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors"
LAYERS, EXPERTS = 24, 128
ROWS = 16            # rows of the scale matrix read per expert (output channels 0-15)
TW, TH = 24, 16      # thumbnail: 24 scale columns x 16 rows, one pixel per scale
UA = {"User-Agent": "media-atlas-builder/1.0"}


def resolve(url):
    """Follow the Hub redirect once so the range requests go straight to the file host."""
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.geturl()


def get(url, a, b):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == b - a:
                return data
        except OSError:
            pass
    raise RuntimeError(f"range {a}-{b} failed")


def e4m3(byte):
    s = -1.0 if byte & 0x80 else 1.0
    e, m = (byte >> 3) & 0xF, byte & 0x7
    if e == 0xF and m == 0x7:
        return 0.0
    return s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def png_gray(path, w, h, px):
    """4-bit greyscale PNG; px holds 0..15, w even."""
    row = lambda y: bytes((px[y * w + x] << 4) | px[y * w + x + 1] for x in range(0, w, 2))
    raw = b"".join(b"\x00" + row(y) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 4, 0, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    url = resolve(URL)
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    hdr = json.loads(get(url, 8, 8 + n))
    base = 8 + n
    jobs = []
    for L in range(LAYERS):
        t = hdr[f"model.llm.layers.{L}.mlp.experts.w13_weight_scale"]
        e_, rows, cols = t["shape"]
        assert e_ == EXPERTS, t
        start = base + t["data_offsets"][0]
        for e in range(EXPERTS):
            off = start + e * rows * cols
            jobs.append((L, e, t["dtype"], cols, off, off + ROWS * cols))
    with cf.ThreadPoolExecutor(16) as ex:
        blobs = list(ex.map(lambda j: get(url, j[4], j[5]), jobs))

    W, H = EXPERTS * TW, LAYERS * TH
    px = [0] * (W * H)
    by_layer = {}
    for (L, e, dtype, cols, _, _), blob in zip(jobs, blobs):
        if dtype == "F8_E4M3":
            vals = [math.log2(abs(e4m3(b))) if e4m3(b) else -20.0 for b in blob]
        else:  # E8M0: the byte is a biased power-of-two exponent
            vals = [float(b - 127) for b in blob]
        # one pixel per scale: rows 0..TH-1, columns 0..TW-1
        img = [vals[y * cols + x] for y in range(TH) for x in range(TW)]
        by_layer.setdefault(L, []).append((e, img))
    def stretch(vals):   # 2nd..98th percentile -> 0..1
        srt = sorted(vals)
        lo, hi = srt[len(srt) // 50], srt[-len(srt) // 50 - 1]
        return [min(1.0, max(0.0, (v - lo) / (hi - lo))) if hi > lo else 0.5 for v in vals]

    for L, items in by_layer.items():
        name = f"model.llm.layers.{L}.mlp.experts.w13_weight_amax"
        if name in hdr:
            t = hdr[name]
            raw = get(url, base + t["data_offsets"][0], base + t["data_offsets"][1])
            amax = struct.unpack(f"<{EXPERTS}f", raw)
            level = [math.log2(amax[e]) for e, _ in items]
        else:
            level = [sum(img) / len(img) for _, img in items]
        bright = stretch(level)
        for (e, img), b in zip(items, bright):
            tex = stretch(img)
            for y in range(TH):
                for x in range(TW):
                    v = 0.6 * b + 0.4 * tex[y * TW + x]
                    px[(L * TH + y) * W + e * TW + x] = max(0, min(15, round(v * 15)))
    out = Path(__file__).with_name("expert-atlas.png")
    png_gray(out, W, H, px)
    print(f"wrote {out} ({W}x{H}, {out.stat().st_size} bytes) from {len(jobs)} range reads")


if __name__ == "__main__":
    main()
