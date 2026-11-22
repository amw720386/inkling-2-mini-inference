"""Cost of each piece of a speculative round (graph replays), plus a per-kernel profile of verify(8).

Modal: modal run modal/engine_app.py::parity --args "bench_round.py [profile_width]"
"""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
prof_w = int(sys.argv[2]) if len(sys.argv) > 2 else 0
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=32768, spec=True)
ids = list(range(1000, 1600))
eng.prefill(ids)
P = len(ids)


def t(fn, n=50):
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


print(f"decode (m=1)      {t(lambda: eng.decode_graph.graph.replay()):.2f} ms")
eng.drafter.anchor.fill_(1000)
eng.drafter.P.fill_(P)
g = eng.drafter.graphs[(next(b for b in eng.drafter.buckets if b >= P + 8), 0)]
print(f"draft             {t(lambda: g.replay()):.2f} ms")
print(f"drafter commit(4) {t(lambda: eng.drafter.commit_graphs[(4, 0)].replay()):.2f} ms")
for m, vg in eng.verify[0].items():
    print(f"verify m={m}        {t(lambda: vg.graph.replay()):.2f} ms")
print(f"commit_spec       {t(lambda: eng.cache.commit_spec(0, 2)):.3f} ms")

if prof_w:
    from torch.profiler import ProfilerActivity, profile

    vg = eng.verify[0][prof_w]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            vg.graph.replay()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16, max_name_column_width=40))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16, max_name_column_width=40))
