"""Temperature-1 seeded DSpark (bench_spec's sampled section) for several SEEDED_GAMMA values, one engine.
Modal: parity --args "seeded_gamma.py"
"""

import json
import sys

import torch

sys.path.insert(0, "/root")
import inkling_engine.engine as em  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = em.Engine(sys.argv[1], max_ctx=32768, spec=True)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
stop = {eng.cfg.eos_token_id}
print("seeded costs", {m: round(c, 2) for m, c in eng.seeded_costs.items()})
for g in (2.0, 1.5, 1.0, 0.7, 2.0, 1.0):
    em.SEEDED_GAMMA = g
    n, s, acc, rounds, widths = 0, 0.0, 0, 0, {}
    for i, it in enumerate(items[:6]):
        _, st = eng.generate(it["input_ids"], 256, stop, spec=True, sampling=Sampling(temperature=1.0, seed=100 + i))
        n, s, acc, rounds = n + st.generated, s + st.decode_s, acc + st.accepted, rounds + st.rounds
        for m, c in st.widths.items():
            widths[m] = widths.get(m, 0) + c
    print(f"SEEDED_GAMMA {g}: {n / s:.1f} tok/s, accept len {(acc + rounds) / rounds:.2f}, widths {widths}", flush=True)
