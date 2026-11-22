"""Memory of the server's default configuration under the 32 GiB cap (Modal: parity --args "memcheck.py")."""

import gc
import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
G = 2**30
for kw in (dict(max_ctx=32768, slots=4, vision=True), dict(max_ctx=262144, slots=1, kv_fp8=True, vision=True)):
    torch.cuda.reset_peak_memory_stats()
    eng = Engine(sys.argv[1], spec=True, **kw)
    a, pk = torch.cuda.memory_allocated() / G, torch.cuda.max_memory_allocated() / G
    n = min(kw["max_ctx"] - 64, 30000)
    eng.prefill(list(range(1000, 1000 + n)))
    for slot in range(1, kw["slots"]):  # every slot holds a prompt snapshot (prefix reuse)
        eng.prefill_slot(slot, list(range(2000, 2000 + 4000)))
    print(f"{kw}: after start {a:.2f} GiB (peak {pk:.2f}); after a {n}-token prefill peak "
          f"{torch.cuda.max_memory_allocated() / G:.2f} GiB, reserved {torch.cuda.memory_reserved() / G:.2f} GiB")
    import weakref

    ref = weakref.ref(eng)
    del eng
    gc.collect()  # the engine holds reference cycles (graphs, drafter); free it before the next profile
    torch.cuda.empty_cache()
    if ref() is not None:
        print("engine still referenced by:", [type(r).__name__ for r in gc.get_referrers(ref())][:10])
    print(f"after cleanup: allocated {torch.cuda.memory_allocated() / G:.2f} GiB, reserved {torch.cuda.memory_reserved() / G:.2f} GiB")
