"""Token ranking for the drafter's shortlist head (Modal: parity --args "build_shortlist.py").

Token frequency over code (Python sources and headers in the container) and English text (WikiText-2 raw and the
GSM8K training set, fetched as parquet, plus /usr/share/doc), then the remaining tokens by id. Writes 65,536 little-endian uint32 token ids to /cache/draft_shortlist.bin (copy it into
inkling_engine/data/). Same method as ingine's build_draft_shortlist.py; no model outputs, so evaluations on model
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
cnt = collections.Counter()
for batch in (code, eng):
    for e in tok.encode_batch(batch, add_special_tokens=False):
        cnt.update(e.ids)
ranked = [i for i, _ in cnt.most_common()]
seen = set(ranked)
ranked += [i for i in range(tok.get_vocab_size()) if i not in seen]
with open("/cache/draft_shortlist.bin", "wb") as f:
    f.write(struct.pack(f"<{ROWS}I", *ranked[:ROWS]))
print(f"{sum(cnt.values())} tokens, {len(cnt)} distinct -> /cache/draft_shortlist.bin")
