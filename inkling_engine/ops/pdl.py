"""Programmatic dependent launch for the engine's Triton kernels (on by default; INKLING_PDL=0 turns it off).

Every kernel starts with `if PDL: gdc_wait()`: launched with the PDL attribute, it is scheduled while its predecessor
runs and waits for that predecessor's completion (and memory flush) before reading its outputs. Right after its own
wait each kernel triggers its dependents, so a successor is resident and ready as soon as possible, but no kernel
can launch two steps ahead (a trigger before the wait cascaded: successors occupied SMs and spun, decode 6.40 ->
6.81 ms). The weight-streaming GEMVs use the head start: before waiting they issue L2 prefetches for the first
1 KiB of each of their weight rows (`prefetch_rows_l2`), which do not depend on the predecessor; small single-program
kernels (norms, depth attention, routing) leave most of the GPU idle, so that DRAM traffic overlaps them.
In-process A/B/A (bench/bench_pdl.py, RTX PRO 6000 proxy): decode step 6.08 -> 5.91 ms, verify(8) 8.68 -> 8.27.
A kernel launched without the PDL attribute behaves as before (wait and trigger are no-ops).
`enable(kernel)` makes every launch of a JIT kernel (also through an autotuner) pass launch_pdl=True.
"""

import os

import triton
import triton.language as tl

PDL_ON = os.environ.get("INKLING_PDL", "1") == "1"
ACTIVE = [PDL_ON]  # runtime switch read at launch (graphs keep what they were captured with)
# L2 prefetch per weight row before the wait; 1 KiB measured best (2 / 4 KiB: smaller gains on verify)
PREFETCH_BYTES = tl.constexpr(int(os.environ.get("INKLING_PDL_PREFETCH", "1024")))


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


@triton.jit
def prefetch_rows_l2(W, rows, rmask, row_bytes, PREF_BYTES: tl.constexpr):
    """Issue L2 prefetches for the first PREF_BYTES of each weight row (128-byte lines). Weights do not depend on
    the preceding kernel, so a PDL-launched kernel can fetch them before gdc_wait while its predecessor runs."""
    LINES: tl.constexpr = PREF_BYTES // 128
    wb = W.to(tl.pointer_type(tl.int8), bitcast=True)
    line = tl.minimum(tl.arange(0, LINES) * 128, row_bytes - 1)  # never past the row (PREF_BYTES may exceed it)
    off = tl.where(rmask, rows, 0).to(tl.int64)[:, None] * row_bytes + line[None, :]
    tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [wb + off], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def prefetch_l2(ptrs):
    """L2 prefetch of each address in a pointer tensor (any element type)."""
    tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32,
                              is_pure=False, pack=1)
