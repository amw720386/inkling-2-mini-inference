"""End-to-end generation: plain vs DSpark greedy, vs SGLang's reference outputs, with timing.

Modal: modal run modal/engine_app.py::parity --args "bench_spec.py [n_new]"
Memory capped to 32 GiB (RTX 5090 emulation on an RTX PRO 6000; proxy numbers).
"""

import json
import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 256
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
print(torch.cuda.get_device_name(0), "| memory capped to 32 GiB")

t0 = time.time()
fp8 = len(sys.argv) > 3 and sys.argv[3] == "fp8"
print("dense_fp8 =", fp8)
eng = Engine(path, max_ctx=32768, spec=True, dense_fp8=fp8)
print(f"engine ready in {time.time() - t0:.0f}s, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

ref = json.load(open("/root/tests/reference/sglang_ref.json"))
items = ref["items"] if isinstance(ref, dict) and "items" in ref else ref
stop = {eng.cfg.eos_token_id}
tot = {"plain": [0, 0.0], "spec": [0, 0.0]}
for it in items[:6]:
    ids = it["input_ids"]
    ref_out = it.get("output_ids", [])
    plain, sp = eng.generate(ids, n_new, stop, spec=False)
    spec, ss = eng.generate(ids, n_new, stop, spec=True)
    same = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), min(len(plain), len(spec)))
    sg = next((i for i, (a, b) in enumerate(zip(plain, ref_out)) if a != b), min(len(plain), len(ref_out)))
    acc_len = (ss.accepted + ss.rounds) / max(ss.rounds, 1)
    print(f"[{it.get('kind', '?'):4s} prompt {len(ids):4d}] plain {sp.decode_tok_s:6.1f} tok/s | spec {ss.decode_tok_s:6.1f} "
          f"tok/s (accept len {acc_len:.2f}, widths {ss.widths}) | spec==plain for {same}/{len(plain)} | "
          f"plain==sglang for {sg}/{len(ref_out)} | prefill {sp.prefill_s * 1000:.0f} ms")
    for k, s in (("plain", sp), ("spec", ss)):
        tot[k][0] += s.generated
        tot[k][1] += s.decode_s

for k, (n, s) in tot.items():
    print(f"overall {k}: {n / s:.1f} tok/s")
print("round costs (ms):", {m: round(c, 2) for m, c in eng.costs.items()})

# long-context decode (8k prompt of natural-ish text repeated)
long_ids = (items[0]["input_ids"] * 200)[:8000]
plain, sp = eng.generate(long_ids, 256, stop, spec=False)
spec, ss = eng.generate(long_ids, 256, stop, spec=True)
print(f"8k prompt: prefill {8000 / sp.prefill_s:.0f} tok/s | plain {sp.decode_tok_s:.1f} tok/s | "
      f"spec {ss.decode_tok_s:.1f} tok/s (accept len {(ss.accepted + ss.rounds) / max(ss.rounds, 1):.2f})")
print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

# sampled decoding (temperature 1, the model's default): plain vs DSpark rejection sampling
from inkling_engine.sampling import Sampling  # noqa: E402

tot = {"plain": [0, 0.0], "spec": [0, 0.0]}
acc_r = [0, 0]
for i, it in enumerate(items[:6]):
    for k, use in (("plain", False), ("spec", True)):
        _, s = eng.generate(it["input_ids"], n_new, stop, spec=use, sampling=Sampling(temperature=1.0, seed=100 + i))
        tot[k][0] += s.generated
        tot[k][1] += s.decode_s
        if use:
            acc_r[0] += s.accepted
            acc_r[1] += s.rounds
print(f"temperature 1.0: plain {tot['plain'][0] / tot['plain'][1]:.1f} tok/s | DSpark {tot['spec'][0] / tot['spec'][1]:.1f} "
      f"tok/s (accept len {(acc_r[0] + acc_r[1]) / max(acc_r[1], 1):.2f})")
