"""Autotuner choices the engine ends up with (after start-up and a few decode / spec steps).

Modal: modal run modal/engine_app.py::parity --args "print_configs.py"
"""

import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.ops import attn_triton, gemv_triton, moe_triton  # noqa: E402

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True)
eng.generate(list(range(1000, 1100)), 16, spec=False)
eng.generate(list(range(1000, 1100)), 16, spec=True)
for mod in (moe_triton, gemv_triton, attn_triton):
    for name in dir(mod):
        k = getattr(mod, name)
        if hasattr(k, "cache") and hasattr(k, "configs") and isinstance(k.cache, dict):
            for key, cfg in k.cache.items():
                print(f"{mod.__name__.split('.')[-1]}.{name} {key} -> {cfg}")
