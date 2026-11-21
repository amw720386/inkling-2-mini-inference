"""Engine vs transformers parity on real weights (run on Modal: modal/engine_app.py::parity).

Checks:
  hf      engine full-prefill logits vs transformers Inkling2LoopForCausalLM
  decode  engine prefill+token-by-token decode vs engine full prefill (cache path)
  long    chunked prefill past the sliding window vs single-shot prefill

bf16 results depend on batch shape (GEMM tiling), and this looped top-8 MoE amplifies
one-ulp differences through routing flips. So "decode" and "long" measure the bf16 noise
floor, and "hf" passes when engine-vs-transformers stays within it. Bit-exact state
handling is checked separately in fp64 by state_exact.py.
"""

from __future__ import annotations

import gc
import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.batch import Batch, SeqChunk  # noqa: E402
from inkling_engine.cache import CacheSpec, InklingCache  # noqa: E402
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.model import InklingModel  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

PROMPT = (
    "The history of numerical weather prediction begins in the 1920s, when Lewis Fry Richardson "
    "attempted to compute a six-hour forecast by hand. His result was wildly wrong, but the method "
    "was sound. Explain, step by step, why his forecast failed and what modern models do differently."
)


def compare(name, a: torch.Tensor, b: torch.Tensor):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (lb.exp() * (lb - la)).sum(-1)
    print(f"[{name}] rows={a.shape[0]} max|d|={diff.max().item():.4f} mean|d|={diff.mean().item():.5f} "
          f"top1={top1:.4f} KL(mean)={kl.mean().item():.2e} KL(max)={kl.max().item():.2e}")
    return top1, kl.mean().item()


def tokens(path, n_min):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True)
    ids = [int(i) for i in enc["input_ids"]]
    assert len(ids) > 16, ids
    filler = tok.encode(" The quick brown fox jumps over the lazy dog, and the dog sleeps.")
    long_ids = list(ids)
    while len(long_ids) < n_min:
        long_ids += filler
    return ids, long_ids[:n_min], tok


def hf_logits(path, ids):
    from transformers import Inkling2LoopForCausalLM

    t0 = time.time()
    model = Inkling2LoopForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"hf load {time.time() - t0:.0f}s")
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda")).logits[0].float().cpu()
        gen = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=24, do_sample=False)[0, len(ids):].tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, gen


def engine_prefill(model, cache, ids, slot=0, start=0, want="all", chunk=None):
    out = []
    chunk = chunk or len(ids)
    for s in range(0, len(ids), chunk):
        part = ids[s:s + chunk]
        b = Batch.build([SeqChunk(slot, start + s, len(part), want)], [part])
        logits, _ = model.forward(b, cache)
        out.append(logits)
    return torch.cat(out)


def engine_generate(model, cache, ids, n, slot=1):
    cache.reset_slot(slot)
    logits = engine_prefill(model, cache, ids, slot=slot, want="last")
    gen = [int(logits[-1].argmax())]
    for i in range(n - 1):
        b = Batch.build([SeqChunk(slot, len(ids) + i, 1)], [[gen[-1]]])
        logits, _ = model.forward(b, cache)
        gen.append(int(logits[-1].argmax()))
    return gen


def main():
    path = sys.argv[1]
    which = set(sys.argv[2:]) or {"hf", "decode", "long"}
    cfg = InklingConfig.from_pretrained(path)
    ids, long_ids, tok = tokens(path, 1600)
    print(f"prompt tokens={len(ids)} long={len(long_ids)}")

    hf = None
    if "hf" in which:
        hf, hf_gen = hf_logits(path, ids)
        print("hf greedy:", repr(tok.decode(hf_gen)))

    t0 = time.time()
    w = load_weights(cfg, path)
    print(f"engine load {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    cache = InklingCache(cfg, CacheSpec(slots=2, max_ctx=4096, ring=cfg.sliding_window + 64))
    model = InklingModel(cfg, w)

    cache.reset_slot(0)
    t0 = time.time()
    full = engine_prefill(model, cache, ids)
    print(f"engine prefill {len(ids)} tok in {time.time() - t0:.1f}s (reference ops)")
    ok = True
    if hf is not None:
        top1, kl = compare("engine-vs-hf", full.cpu(), hf)
        ok &= top1 >= 0.90 and kl < 5e-2
        gen = engine_generate(model, cache, ids, 24)
        print("engine greedy:", repr(tok.decode(gen)))
        print("greedy match:", sum(a == b for a, b in zip(gen, hf_gen)), "/", len(gen))

    if "decode" in which:
        cache.reset_slot(0)
        split = len(ids) // 2
        parts = [engine_prefill(model, cache, ids[:split])[-1:]]
        for i in range(split, len(ids) - 1):
            b = Batch.build([SeqChunk(0, i, 1)], [[ids[i]]])
            parts.append(model.forward(b, cache)[0])
        inc = torch.cat(parts)
        top1, kl = compare("decode-vs-prefill", inc, full[split - 1:len(ids) - 1])
        print("  (bf16 noise floor, informational)")

    if "long" in which:
        cache.reset_slot(0)
        one = engine_prefill(model, cache, long_ids, want="all")
        cache.reset_slot(0)
        chunked = engine_prefill(model, cache, long_ids, want="all", chunk=600)
        top1, kl = compare("chunked-vs-single(1600)", chunked, one)
        print("  (bf16 noise floor, informational)")

    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
