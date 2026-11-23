"""Token ranking for the drafter's shortlist head (Modal: parity --args "build_shortlist.py").

Token frequency over code (Python sources and headers in the container) and English text (WikiText-2 raw and the
GSM8K training set, fetched as parquet, plus /usr/share/doc), then the remaining tokens by id. Writes 65,536 little-endian uint32 token ids to /cache/draft_shortlist.bin (copy it into
inkling_engine/data/). With the argument "mix" it also counts Wikipedia text of 12 other languages (one row group of
a parquet shard each, up to 2M characters), ranks by 0.35 code + 0.35 English + 0.30 other languages (per-bucket
frequencies, each language equal) and writes /cache/draft_shortlist_mix.bin. Same method as ingine's build_draft_shortlist.py; no model outputs, so evaluations on model
traces are not biased.
"""

import collections
import glob
import random
import struct
import sys

from tokenizers import Tokenizer

ROWS = 65536
path = sys.argv[1]
MIX = len(sys.argv) > 2 and sys.argv[2] == "mix"
LANGS = ["fr", "de", "es", "pt", "it", "ru", "zh", "ja", "ko", "ar", "hi", "tr"]
random.seed(0)
tok = Tokenizer.from_file(f"{path}/tokenizer.json")


def collect(patterns, budget):
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    random.shuffle(files)
    out, n = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if t.strip():
            out.append(t[:200000])
            n += len(out[-1])
            if n > budget:
                break
    return out, len(files)


code, nc = collect(["/usr/local/lib/python3.11/**/*.py", "/usr/include/**/*.h", "/root/inkling_engine/**/*.py"], 15_000_000)
eng, ne = collect(["/usr/share/doc/**/*.md", "/usr/share/doc/**/*.txt", "/usr/share/doc/**/README*",
                   "/usr/share/doc/**/*.html", "/usr/local/lib/python3.11/**/*.md", "/usr/local/lib/python3.11/**/*.rst",
                   "/usr/local/lib/python3.11/**/*.txt", "/usr/share/common-licenses/*"], 25_000_000)


def hf_parquet(url, cols):
    import io
    import subprocess

    import requests

    try:
        import pyarrow.parquet as pq
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
        import pyarrow.parquet as pq
    t = pq.read_table(io.BytesIO(requests.get(url, timeout=120).content))
    return ["\n".join(str(t.column(c)[i]) for c in cols) for i in range(t.num_rows)]


wiki = hf_parquet("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
                  "train-00000-of-00001.parquet", ["text"])
gsm = hf_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet",
                 ["question", "answer"])
eng += ["".join(wiki[i:i + 200]) for i in range(0, len(wiki), 200)] + ["\n\n".join(gsm[i:i + 50]) for i in range(0, len(gsm), 50)]
print(f"code files {nc} ({sum(map(len, code)) / 1e6:.1f} MB used), text files {ne} + WikiText-2 + GSM8K ({sum(map(len, eng)) / 1e6:.1f} MB)")
def wiki_lang(lang, budget=2_000_000):
    """Text of the first row group of the first Wikipedia parquet shard of `lang`, fetched with range requests."""
    import fsspec
    import pyarrow.parquet as pq
    import requests

    urls = requests.get(f"https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.{lang}/train",
                        timeout=60).json()
    with fsspec.open(urls[0], "rb", block_size=2**20) as f:
        col = pq.ParquetFile(f).read_row_group(0, columns=["text"]).column("text")
        return chr(10).join(str(x) for x in col.to_pylist())[:budget]


def count(texts):
    c = collections.Counter()
    for e in tok.encode_batch(texts, add_special_tokens=False):
        c.update(e.ids)
    return c


cnt, name = count(code) + count(eng), "draft_shortlist.bin"
ranked = [i for i, _ in cnt.most_common()]
if MIX:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fsspec[http]", "aiohttp"])
    buckets = [(0.35, count(code)), (0.35, count(eng))]
    for lang in LANGS:
        t = wiki_lang(lang)
        buckets.append((0.30 / len(LANGS), count([t[i:i + 100_000] for i in range(0, len(t), 100_000)])))
        print(f"{lang}: {len(t)} chars, {sum(buckets[-1][1].values())} tokens, {len(buckets[-1][1])} distinct", flush=True)
    score = collections.Counter()
    for w, c in buckets:
        n = sum(c.values())
        for t, f in c.items():
            score[t] += w * f / n
    cnt, name = score, "draft_shortlist_mix.bin"
    ranked = [i for i, _ in score.most_common()]
seen = set(ranked)
ranked += [i for i in range(tok.get_vocab_size()) if i not in seen]
with open(f"/cache/{name}", "wb") as f:
    f.write(struct.pack(f"<{ROWS}I", *ranked[:ROWS]))
print(f"{len(cnt)} distinct tokens counted -> /cache/{name}")
