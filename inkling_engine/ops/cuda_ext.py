"""CUDA C++ kernels of the one-request decode path (from inkling-2-mini-single-stream) (inkling_engine/csrc), compiled for sm_120a at first use.

The build is cached in INKLING_EXT_DIR (default /cache/torch_ext when /cache is writable, else
~/.cache/inkling/torch_ext; one folder per hash of the sources), so a container that
mounts the cache volume compiles once per source change.
"""

from __future__ import annotations

import hashlib
import os

import torch

from .pdl import ACTIVE as _PDL

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
_SOURCES = ["bindings.cpp", "dense_gemv.cu", "msg_gates.cu", "post_attn.cu"]
_HEADERS = ["common.cuh"]
_mod = None


def mod():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load

        paths = [os.path.join(_SRC, s) for s in _SOURCES]
        h = hashlib.sha1(b"".join(open(p, "rb").read() for p in paths + [os.path.join(_SRC, x) for x in _HEADERS])).hexdigest()[:12]
        root = os.environ.get("INKLING_EXT_DIR") or (
            "/cache/torch_ext" if os.access("/cache", os.W_OK) else os.path.expanduser("~/.cache/inkling/torch_ext"))
        build = os.path.join(root, h)
        os.makedirs(build, exist_ok=True)
        _mod = load(name=f"inkling_ss_{h}", sources=paths, build_directory=build, verbose=False,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo",
                                       "--expt-relaxed-constexpr", "-std=c++17"])
    return _mod


_EMPTY = {}


def _factor_empty(dev):
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, dtype=torch.float32, device=dev)
    return _EMPTY[dev]


def _e(dev):
    return _factor_empty(dev)

def fp8_gemv(x: torch.Tensor, w, out: torch.Tensor, bounds=None, variant: int = 0, grid: int = 0):
    """out[m, r] = bf16(w.scale[r] * w.w8[r] . x_i[m]). x: [I, M, K] (stride 0 on dim 0 for one shared input) or
    [M, K]; bounds: 4 row boundaries choosing the input of each 16-row tile (rows < b0: 0, < b1: 1, < b2: 2, < b3: 1,
    else 2); None: every row reads input 0."""
    if x.dim() == 2:
        x = x[None]
    r = w.w8.shape[0]
    b = list(bounds) if bounds is not None else [r, r, r, r]
    mod().fp8_gemv(x, x.stride(0), x.stride(1), w.w8, w.scale, out, x.shape[1], b, bool(_PDL[0]), variant, grid)
    return out

_MSG_WS = {}


def msg_gates_cuda(hs, ms, lw, gate0: int, n_gate: int, g_buf, out, eps: float, *, norm_w=None, tmp=None):
    """Same contract as fused_triton.msg_gates for T <= 8 (hs / ms fp32 [n_gate, T, H] views; out bf16 normed with
    norm_w (needs fp32 tmp) or fp32 without)."""
    m = mod()
    pdl = bool(_PDL[0])
    key = (hs.device.index, n_gate)
    if key not in _MSG_WS:
        _MSG_WS[key] = (torch.zeros(n_gate, 8, 36, dtype=torch.float32, device=hs.device),
                        torch.zeros(n_gate, dtype=torch.int32, device=hs.device))
    ss, counter = _MSG_WS[key]
    assert lw.ctl_down.dtype == torch.bfloat16 and lw.ctl_down.is_contiguous() and lw.ctl_gain.is_contiguous()
    assert norm_w is None or (norm_w.dtype == torch.bfloat16 and norm_w.stride(-1) == 1)
    m.msg_down(hs, ms, lw.ctl_down, gate0, g_buf, eps, pdl)
    nw = norm_w if norm_w is not None else _factor_empty(hs.device)
    m.msg_up(hs, ms, g_buf, lw.ctl_gain, lw.ctl_forget, gate0, nw, tmp if tmp is not None else out, ss, counter, out,
             eps, pdl)


_F32 = {}


def f32c(t: torch.Tensor) -> torch.Tensor:
    """fp32 contiguous copy of a (static) weight, cached by address."""
    k = (t.data_ptr(), t.dtype, tuple(t.shape))
    if k not in _F32:
        _F32[k] = t.float().contiguous()
    return _F32[k]


def post_attn_cuda(o, conv_w, state, cand, blk, ns: int, part: int, boundary: bool, resid, query, norm_w, out, cfg,
                   T: int, slot=None):
    """Fused post_attn for one sequence. out: bf16 x (norm_w given) or fp32 mi. state: this visit's conv state
    [3, H], or with `slot` (int32 device tensor) every slot's [S, 3, H] and the kernel picks slot[0]; cand:
    candidate states [T, 3, H] (verify) or None (decode)."""
    e = _e(o.device)
    norm = norm_w is not None
    mod().post_attn(o, f32c(conv_w), state, cand if cand is not None else e, blk, ns, part, boundary, resid, query,
                    f32c(norm_w) if norm else e, out if norm else e, e if norm else out, cfg.conv_out_scale,
                    cfg.rms_eps, T, bool(_PDL[0]), slot if slot is not None else e,
                    state.stride(0) if slot is not None else 0)
