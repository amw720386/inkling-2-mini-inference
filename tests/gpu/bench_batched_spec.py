"""Steady-state throughput of k concurrent requests: batched plain, always-batched speculation, adaptive."""

import json
import sys
import threading
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402

path = sys.argv[1]
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=8192, spec=True, slots=4)
sched = Scheduler(eng)
items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
N = 384


def t(fn, n=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


print("graph ms: " + "  ".join(f"verify({k},{m})={t(eng.verify[(k, m)].graph.replay):.2f}"
                               for k in (1, 2, 4) for m in (2, 4, 8)))
print("graph ms: " + "  ".join(f"decode(bs={k})={t(eng.decode_graphs[k].graph.replay):.2f}" for k in (1, 2, 4)))
for k in (2, 4):
    for batched in (False, True, "adaptive"):
        sched.batched_spec = bool(batched)
        eng.verify_costs_saved = getattr(eng, 'verify_costs_saved', eng.verify_costs)
        eng.verify_costs = eng.verify_costs_saved if batched == "adaptive" else {}
        sched.counters = {kk: ([0, 0.0] if kk != "batch_sizes" else {}) for kk in sched.counters}
        res = {}
        th = [threading.Thread(target=lambda i=i: res.__setitem__(i, sched.generate(items[i]["input_ids"], N, {eng.cfg.eos_token_id})))
              for i in range(k)]
        t0 = time.time()
        for x in th:
            x.start()
        for x in th:
            x.join()
        dt = time.time() - t0
        toks = sum(len(res[i][0]) for i in res)
        rounds = sum(res[i][1].rounds for i in res)
        acc = sum(res[i][1].accepted for i in res)
        print(f"{k} concurrent, batched_spec={batched}: {toks} tokens in {dt:.2f}s -> {toks / dt:.0f} tok/s"
              + (f" (accept len {(acc + rounds) / max(rounds, 1):.2f})" if batched else ""))
        print("   steps:", {kk: (v if kk == "batch_sizes" else (v[0], round(v[1], 2))) for kk, v in sched.counters.items()})
