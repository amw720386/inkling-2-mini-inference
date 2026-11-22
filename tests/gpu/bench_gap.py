"""GPU idle time inside generation (host gaps between graph replays): wall time vs summed kernel time.

Modal: parity --args "bench_gap.py". Greedy, 256 new tokens, plain and DSpark.
"""

import json
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=32768, spec=True)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
ids = items[1]["input_ids"]
stop = {eng.cfg.eos_token_id}
for spec in (False, True):
    eng.generate(ids, 64, stop, spec=spec)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t0 = time.time()
        out, st = eng.generate(ids, 256, stop, spec=spec)
        torch.cuda.synchronize()
        wall = time.time() - t0
    busy = sum(e.self_device_time_total for e in prof.key_averages()) / 1e6  # us -> s (kernels and copies)
    dec = st.decode_s
    print(f"{'DSpark' if spec else 'plain '}: {len(out)} tokens, wall {wall * 1000:.0f} ms (decode {dec * 1000:.0f} ms), "
          f"GPU busy {busy * 1000:.0f} ms -> idle {100 * (1 - busy / wall):.1f}%"
          + (f", rounds {st.rounds}, {1000 * (wall - busy) / max(st.rounds, 1):.2f} ms idle per round" if spec else
             f", {1000 * (wall - busy) / len(out):.3f} ms idle per token"))
