"""Reference (pure torch) implementations of every engine op.

These mirror the dtype flow of transformers' modeling_inkling2_loop.py except that
attention softmax runs in fp32 (as every fused kernel does). Fast kernels are tested
against these.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..quant import ExpertWeight

# compute dtype for reference math; tests set float64 for batch-invariant checks
COMPUTE = torch.float32


# ----------------------------------------------------------------------------- norms
def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float, out_dtype=torch.bfloat16):
    xf = x.to(COMPUTE)
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = weight.to(COMPUTE) * y
    return y.to(out_dtype)


def unweighted_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.to(COMPUTE)
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Per-head (last dim) clip to `threshold`, after an optional learned scale. Returns bf16."""
    xf = x.to(COMPUTE)
    if scale is not None:
        xf = xf * scale.to(COMPUTE)
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    return (xf * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(torch.bfloat16)


def log_scale(x: torch.Tensor, positions: torch.Tensor, alpha: float, n_floor: float) -> torch.Tensor:
    """x: [T, H, D] bf16; positions [T]."""
    tau = 1.0 + alpha * torch.log(((positions + 1).to(COMPUTE) / n_floor).clamp(min=1.0))
    return (x.to(COMPUTE) * tau[:, None, None]).to(torch.bfloat16)


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


# ---------------------------------------------------------------------- depth attn
def attn_res(sources: list[torch.Tensor], scaled_queries: torch.Tensor) -> torch.Tensor:
    """Block attention residual over depth.

    sources: S tensors [T, H] (any float dtype); scaled_queries: fp32 [Q, H] already
    multiplied by H**-0.5. Keys and values are the unweighted-RMS-normed sources.
    Returns fp32 [T, Q, H].
    """
    stacked = torch.stack([s.to(COMPUTE) for s in sources], dim=1)  # [T, S, H]
    normed = unweighted_rms(stacked)
    logits = torch.einsum("tsh,qh->tqs", normed, scaled_queries.to(COMPUTE))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("tqs,tsh->tqh", probs, normed)


def message_gate(h: torch.Tensor, m: torch.Tensor, down: torch.Tensor, gain: torch.Tensor,
                 forget: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(1 - tanh(F g)) * h + tanh(G g) * m, g = down(cat(rms(h), rms(m))). fp32 out."""
    h, m = h.to(COMPUTE), m.to(COMPUTE)
    g = torch.cat([unweighted_rms(h, eps), unweighted_rms(m, eps)], dim=-1)
    g = F.linear(g.to(down.dtype).to(COMPUTE), down.to(COMPUTE)).to(down.dtype).to(COMPUTE)
    f = torch.tanh(F.linear(g, forget.to(COMPUTE)))
    gg = torch.tanh(F.linear(g, gain.to(COMPUTE)))
    return (1 - f) * h + gg * m


# ---------------------------------------------------------------------- short convs
def short_conv(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, out_scale: float):
    """Causal depthwise conv over one sequence.

    x [T, C] bf16, state [K-1, C] (previous inputs, oldest first), weight [C, K] where
    weight[:, K-1] multiplies the current token. Returns (bf16 [T, C], new_state).
    """
    k = weight.shape[1]
    ctx = torch.cat([state.to(COMPUTE), x.to(COMPUTE)], dim=0)  # [K-1+T, C]
    win = ctx.unfold(0, k, 1)  # [T, C, K]
    y = (win * weight.to(COMPUTE)[None]).sum(-1)
    y = (y + x.to(COMPUTE)) * out_scale
    return y.to(torch.bfloat16), ctx[-(k - 1):].to(state.dtype)


def dyn_conv(x: torch.Tensor, state: torch.Tensor, static_w: torch.Tensor, dyn_w: torch.Tensor,
             n_heads: int, out_scale: float):
    """Dynamic short conv (global layers' K/V).

    x [T, C] bf16, state [K-1, C]; static_w [K, C] with static_w[0] weighting the current
    token; dyn_w [T, n_heads * K] (bf16 projection of the layer input), shared within a
    head. Returns (bf16 [T, C], new_state).
    """
    t, c = x.shape
    k = static_w.shape[0]
    ctx = torch.cat([state.to(COMPUTE), x.to(COMPUTE)], dim=0)
    win = ctx.unfold(0, k, 1).flip(-1)  # [T, C, K], win[..., lag]
    w = static_w.to(COMPUTE).T.reshape(n_heads, c // n_heads, k)[None] + dyn_w.to(COMPUTE).reshape(t, n_heads, 1, k)
    y = (win.reshape(t, n_heads, c // n_heads, k) * w).sum(-1).reshape(t, c)
    y = (y + x.to(COMPUTE)) * out_scale
    return y.to(torch.bfloat16), ctx[-(k - 1):].to(state.dtype)


# ------------------------------------------------------------------------ attention
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor,
              k_pos: torch.Tensor, rel_logits: torch.Tensor, sinks: torch.Tensor,
              window: int | None, scale: float) -> torch.Tensor:
    """One sequence. q [T, H, D], k/v [S, KVH, D] bf16, q_pos [T], k_pos [S].

    rel_logits [T, H, R] bf16: bias for backward distance d in [0, R); zero beyond.
    sinks [T, H]: one extra zero-value logit per query. Returns bf16 [T, H*D].
    """
    t, h, d = q.shape
    kvh = k.shape[1]
    rep = h // kvh
    kf = k.to(COMPUTE).repeat_interleave(rep, dim=1)  # [S, H, D]
    vf = v.to(COMPUTE).repeat_interleave(rep, dim=1)
    logits = torch.einsum("thd,shd->hts", q.to(COMPUTE), kf) * scale
    dist = q_pos[:, None] - k_pos[None, :]  # [T, S]
    r = rel_logits.shape[-1]
    idx = dist.clamp(0, r - 1)[None].expand(h, -1, -1)  # [H, T, S]
    bias = rel_logits.to(COMPUTE).permute(1, 0, 2).gather(-1, idx)
    bias = bias.masked_fill(((dist < 0) | (dist >= r))[None], 0.0)
    logits = logits + bias
    allowed = dist >= 0
    if window is not None:
        allowed &= dist < window
    logits = logits.masked_fill(~allowed[None], float("-inf"))
    logits = torch.cat([logits, sinks.to(COMPUTE).T[:, :, None]], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    out = torch.einsum("hts,shd->thd", probs, vf)
    return out.reshape(t, h * d).to(torch.bfloat16)


# ------------------------------------------------------------------------------ MoE
def route(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, top_k: int, scale: float):
    logits = F.linear(x.to(COMPUTE), w.to(COMPUTE))
    scores = torch.sigmoid(logits)
    idx = torch.topk(scores + bias.to(COMPUTE), top_k, dim=-1, sorted=False).indices
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
    return wts * scale, idx


def softcapped_swiglu(gu: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    """gu [..., 2I] with interleaved (gate, up) columns. bf16 out."""
    g, u = gu[..., 0::2].to(COMPUTE), gu[..., 1::2].to(COMPUTE)
    return (softcap(g, gate_cap) * torch.sigmoid(g) * softcap(u, up_cap)).to(torch.bfloat16)


def moe(x: torch.Tensor, wts: torch.Tensor, idx: torch.Tensor, w13: ExpertWeight, w2: ExpertWeight,
        gate_cap: float, up_cap: float) -> torch.Tensor:
    out = torch.zeros(x.shape, dtype=COMPUTE, device=x.device)
    for e in torch.unique(idx).tolist():
        tok, slot = torch.where(idx == e)
        a = F.linear(x[tok], w13.dequant(slice(e, e + 1))[0])
        a = softcapped_swiglu(a, gate_cap, up_cap)
        y = F.linear(a, w2.dequant(slice(e, e + 1))[0])
        out.index_add_(0, tok, y.to(COMPUTE) * wts[tok, slot, None])
    return out.to(torch.bfloat16)


def sdpa_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
