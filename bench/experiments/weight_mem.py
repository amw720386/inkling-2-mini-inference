"""GPU memory after loading the weights, the drafter and the caches (Modal: parity --args "weight_mem.py")."""
import os
import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine import engine as E  # noqa: E402
from inkling_engine.fused_model import FusedModel  # noqa: E402

G = 2**30
bf16 = os.environ.get("BF16") == "1"
compact, drafter = FusedModel.compact, E.Drafter.__init__


def c(self):
    compact(self)
    torch.cuda.empty_cache()
    print(f"weights after compact(): {torch.cuda.memory_allocated() / G:.3f} GiB", flush=True)


def d(self, *a, **k):
    b = torch.cuda.memory_allocated()
    drafter(self, *a, **k)
    print(f"drafter: +{(torch.cuda.memory_allocated() - b) / G:.3f} GiB", flush=True)


FusedModel.compact, E.Drafter.__init__ = c, d
eng = E.Engine(sys.argv[1], max_ctx=32768, slots=4, spec=True, vision=True, dense_fp8=not bf16, fp4_prefill=not bf16)
print(f"BF16={bf16}: engine allocated {torch.cuda.memory_allocated() / G:.3f} GiB", flush=True)
