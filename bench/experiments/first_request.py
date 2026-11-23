"""Greedy prompts past EOS through the scheduler (server profile), repeated: per run the token hash, first EOS
position, accept length, widths and longest gap between tokens tell different text apart from a one-time stall.
Modal: parity --args "first_request.py [reps]"   (BF16=1: bf16 profile)
"""

import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402

reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
fp = os.environ.get("BF16") != "1"
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True, slots=4, dense_fp8=fp, fp4_prefill=fp)
sched = Scheduler(eng)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
eos = eng.cfg.eos_token_id
sched.generate(items[0]["input_ids"], 32, (), Sampling(temperature=0.0))  # serve_bench's warm-up
for rep in range(reps):
    for i in (0, 1):
        toks, ts = [], []

        def on_token(t):
            if t is not None:
                toks.append(t)
                ts.append(time.perf_counter())

        req = sched.submit(items[i]["input_ids"], 256, (), Sampling(temperature=0.0), on_token)
        req.done.wait()
        if req.error:
            raise req.error
        st, gaps = req.stats, [b - a for a, b in zip(ts, ts[1:])]
        g = max(gaps)
        print(f"rep {rep} prompt {i}: {(len(toks) - 1) / (ts[-1] - ts[0]):6.1f} tok/s, hash "
              f"{hashlib.sha1(str(toks).encode()).hexdigest()[:8]}, first eos {toks.index(eos) if eos in toks else None}, "
              f"rounds {st.rounds}, accept len {(st.accepted + st.rounds) / max(st.rounds, 1):.2f}, widths {st.widths}, "
              f"max gap {g * 1000:.1f} ms at {gaps.index(g)}", flush=True)
