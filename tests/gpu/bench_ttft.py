"""Prefill latency (time to first token) by prompt length (Modal: parity --args "bench_ttft.py").

Wall time of Engine.prefill (fresh slot, includes the drafter commit) vs the GPU time of the same work, to see
how much of short-prompt latency is host-side launch overhead.
"""

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(path, max_ctx=32768, spec=True)
for n in (8, 24, 32, 48, 100, 300, 1000, 2000):
    ids = list(range(5000, 5000 + n))
    walls, gpus = [], []
    for rep in range(4):
        eng.slot_ids[0], eng.snaps[0] = [], None  # fresh prefill each time
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.time()
        e0.record()
        eng.prefill(ids)
        e1.record()
        torch.cuda.synchronize()
        walls.append((time.time() - t0) * 1000)
        gpus.append(e0.elapsed_time(e1))
    w, g = sorted(walls)[1], sorted(gpus)[1]
    print(f"prompt {n:5d}: wall {w:6.1f} ms, gpu span {g:6.1f} ms")

# chunk graphs vs eager prefill: same logits (within split-order noise) and the same greedy continuation
graphs = eng.chunk_graphs
for n in (5, 24, 2048 + 20):  # the last: a 2048-token eager chunk, then a 20-token graph chunk
    ids = list(range(7000, 7000 + n))
    outs = {}
    for mode in ("graph", "eager"):
        eng.chunk_graphs = graphs if mode == "graph" else {}
        eng.slot_ids[0], eng.snaps[0] = [], None
        lg, _ = eng.prefill(ids)
        gen, _ = eng.generate(ids + [int(lg.argmax())], 24, set(), spec=False)
        outs[mode] = (lg.float(), gen)
    eng.chunk_graphs = graphs
    d = (outs["graph"][0] - outs["eager"][0]).abs().max().item()
    same = next((i for i, (a, b) in enumerate(zip(outs["graph"][1], outs["eager"][1])) if a != b), 24)
    print(f"prompt {n}: graph vs eager logits max |d| {d:.3f}, argmax equal {int(outs['graph'][0].argmax()) == int(outs['eager'][0].argmax())}, greedy continuation equal for {same}/24")
