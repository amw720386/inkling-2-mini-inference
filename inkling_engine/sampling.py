"""Sampling and speculative acceptance (pure torch, no kernels)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None


def probs(logits: torch.Tensor, sp: Sampling) -> torch.Tensor:
    """Sampling distribution over the last dim after temperature, top-k and top-p (rows independent)."""
    logits = logits.float() / max(sp.temperature, 1e-5)
    if sp.top_k > 0:
        kth = torch.topk(logits, min(sp.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    p = torch.softmax(logits, -1)
    if sp.top_p < 1.0:
        sp_, order = p.sort(dim=-1, descending=True)
        cut = torch.cumsum(sp_, -1) - sp_ > sp.top_p
        p = torch.zeros_like(p).scatter_(-1, order, sp_.masked_fill(cut, 0.0))
        p = p / p.sum(-1, keepdim=True)
    return p


def sample(logits: torch.Tensor, sp: Sampling, gen=None) -> int:
    if sp.temperature <= 0:
        return int(logits.argmax())
    return int(torch.multinomial(probs(logits, sp), 1, generator=gen))


def accept_sampled(drafts: torch.Tensor, target_logits: torch.Tensor, cand: torch.Tensor, q: torch.Tensor,
                   sp: Sampling, gen=None) -> tuple[int, int]:
    """Speculative rejection sampling, exact for the target's sampling distribution p.

    drafts [n] were sampled from q (q [n, K] over candidate ids cand [n, K]). Accept draft r while
    u_r q_r(d_r) < p_r(d_r); after a rejection at a, sample from max(p_a - q_a, 0); after n acceptances,
    sample from p_n. Returns (accepted count, next token) with one host sync.
    """
    n = drafts.shape[0]
    p = probs(target_logits, sp)  # [n + 1, V]
    qd = (q * (cand == drafts[:, None])).sum(-1)
    pd = p[:n].gather(1, drafts[:, None])[:, 0]
    u = torch.rand(n, device=p.device, generator=gen)
    acc = (u * qd < pd).int().cumprod(0).sum()
    q_full = torch.zeros(n + 1, p.shape[1], device=p.device)
    q_full[:n].scatter_(1, cand, q)
    resid = (p - q_full).clamp(min=0)
    resid[n] = p[n]
    row = resid.index_select(0, acc.view(1))[0]
    row = torch.where(row.sum() > 0, row, p.index_select(0, acc.view(1))[0])
    nxt = torch.multinomial(row / row.sum(), 1, generator=gen)
    acc, nxt = torch.cat([acc.view(1), nxt]).tolist()
    return acc, nxt
