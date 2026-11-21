"""FP8 full-attention KV: accuracy vs bf16, and 256k-context memory/speed under a 32 GiB cap.

Modal: modal run modal/engine_app.py::parity --args "longctx.py [long_tokens]"
"""

import gc
import json
import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
long_n = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
GiB = 2**30

ref = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
base = ref[0]["input_ids"]
torch.manual_seed(0)
prompt4k = (base + torch.randint(1000, 60000, (4000,)).tolist())[:4000] + base[-8:]


def kl(a, b):
    a, b = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    return (b.exp() * (b - a)).sum().item()


# --- accuracy at 4k: bf16 vs fp8 cache
res = {}
for fp8 in (False, True):
    eng = Engine(path, max_ctx=8192, spec=False, kv_fp8=fp8)
    logits, _ = eng.prefill(prompt4k)
    out, _ = eng.generate(prompt4k, 64, {eng.cfg.eos_token_id})
    res[fp8] = (logits.float().cpu(), out)
    del eng
    gc.collect()
    torch.cuda.empty_cache()
(lb, ob), (lf, of) = res[False], res[True]
same = next((i for i, (a, b) in enumerate(zip(ob, of)) if a != b), min(len(ob), len(of)))
print(f"4k prompt: KL(bf16 || fp8 KV) = {kl(lf, lb):.2e}, argmax {'same' if lb.argmax() == lf.argmax() else 'DIFF'}, "
      f"greedy match {same}/{len(ob)}")

# --- 256k context engine
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
eng = Engine(path, max_ctx=262144, spec=True, kv_fp8=True)
print(f"max_ctx 262144, kv_fp8, DSpark: ready in {time.time() - t0:.0f}s, allocated "
      f"{torch.cuda.memory_allocated() / GiB:.2f} GiB, peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB")
ids = (base * (long_n // len(base) + 1))[:long_n - 16] + base[-16:]
torch.cuda.synchronize()
t0 = time.time()
logits, _ = eng.prefill(ids)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"prefill {len(ids)} tokens: {dt:.1f}s ({len(ids) / dt:.0f} tok/s), peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB")
for spec in (False, True):
    out, st = eng.generate(ids, 128, {eng.cfg.eos_token_id}, spec=spec)
    print(f"decode at {len(ids)} ctx ({'DSpark' if spec else 'plain'}): {st.decode_tok_s:.1f} tok/s"
          + (f", accept len {(st.accepted + st.rounds) / max(st.rounds, 1):.2f}" if spec else "")
          + f" (prompt reused: {st.reused_tokens} tokens)")
print(f"peak allocated {torch.cuda.max_memory_allocated() / GiB:.2f} GiB")
