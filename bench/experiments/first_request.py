"""First greedy request after start-up vs repeats (server profile): finds one-time stalls (lazy compiles/captures).
Modal: parity --args "first_request.py"
"""

import json
import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4, dense_fp8=True, fp4_prefill=True)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
stop = {eng.cfg.eos_token_id}
ids = items[0]["input_ids"]
eng.generate(ids, 32, stop, spec=True)  # serve_bench's warm-up
for rep in range(3):
    for i in (0, 1):
        t0, gaps, last = time.perf_counter(), [], time.perf_counter()
        _, st = eng.generate(items[i]["input_ids"], 256, stop, spec=True)
        print(f"rep {rep} prompt {i}: {st.generated / st.decode_s:.1f} tok/s, rounds {st.rounds}, "
              f"accepted {st.accepted}, widths {st.widths}, wall {time.perf_counter() - t0:.3f}s", flush=True)
