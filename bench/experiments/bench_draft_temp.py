"""Sampled DSpark acceptance vs the draft temperature (Modal: parity --args "bench_draft_temp.py [n_new]").

Rejection sampling is exact for any draft distribution q, so the drafter may sample at a different temperature
than the target; a sharper q can match a peaked target better. Temperature-1 requests, 6 prompts x 4 seeds; scales as a comma list (third argument).
"""

import json
import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402

path = sys.argv[1]
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 256
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=32768, spec=True)
ref = json.load(open("/root/tests/reference/sglang_ref.json"))
items = ref["items"] if isinstance(ref, dict) and "items" in ref else ref
stop = {eng.cfg.eos_token_id}
for scale in [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1.0", "0.8", "0.6", "0.4"])]:
    eng.draft_temp_scale = scale
    n = t = acc = rounds = 0
    for seed in (100, 200, 300, 400):
        for i, it in enumerate(items[:6]):
            _, s = eng.generate(it["input_ids"], n_new, stop, spec=True, sampling=Sampling(temperature=1.0, seed=seed + i))
            n += s.generated
            t += s.decode_s
            acc += s.accepted
            rounds += s.rounds
    print(f"draft temperature x{scale}: {n / t:.1f} tok/s, accept len {(acc + rounds) / max(rounds, 1):.3f}")
