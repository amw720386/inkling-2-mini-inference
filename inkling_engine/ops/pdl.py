"""Programmatic dependent launch for the engine's Triton kernels.

Every kernel starts with `if PDL: gdc_wait()`: launched with the PDL attribute, it is scheduled while its
predecessor drains and waits for that predecessor's completion (and memory flush) before reading anything.
Kernels do not trigger their dependents early (the trigger is implicit at completion): an early trigger let
successors occupy SMs and spin while GEMVs still needed them (decode 6.40 -> 6.81 ms). With the implicit
trigger, an in-process A/B/A (tests/gpu/bench_pdl.py, RTX PRO 6000 proxy) measured no difference beyond
noise (decode 6.32 vs 6.34 ms), so it is off by default: INKLING_PDL=1 turns it on.
A kernel launched without the PDL attribute behaves as before (the wait is a no-op).
`enable(kernel)` makes every launch of a JIT kernel (also through an autotuner) pass launch_pdl=True.
INKLING_PDL=0 turns it off.
"""

import os

PDL_ON = os.environ.get("INKLING_PDL", "0") == "1"
ACTIVE = [PDL_ON]  # runtime switch read at launch (graphs keep what they were captured with)


def enable(*kernels):
    if not PDL_ON:
        return
    from triton.runtime.jit import JITFunction

    for k in kernels:
        fn = k
        while not isinstance(fn, JITFunction):  # autotuner(s) -> the JITFunction they launch
            fn = fn.fn
        if getattr(fn, "run_pdl_wrapped", False):
            continue
        orig = fn.run

        def run(*args, _orig=orig, **kwargs):
            if ACTIVE[0]:
                kwargs.setdefault("launch_pdl", True)
            return _orig(*args, **kwargs)

        fn.run = run
        fn.run_pdl_wrapped = True
