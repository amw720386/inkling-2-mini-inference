"""Speculative rejection sampling must emit tokens distributed exactly as the target p (CPU, toy sizes)."""

import torch

from inkling_engine.sampling import Sampling, accept_sampled, probs

V, K, N = 12, 6, 3


def make(seed):
    g = torch.Generator().manual_seed(seed)
    target_logits = torch.randn(N + 1, V, generator=g) * 2
    cand = torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(N)])
    draft_logits = torch.randn(N, K, generator=g) * 2
    return target_logits, cand, draft_logits


def draw_drafts(cand, q, gen):
    return torch.stack([cand[r][torch.multinomial(q[r], 1, generator=gen)][0] for r in range(cand.shape[0])])


def test_first_token_marginal_matches_target():
    target_logits, cand, draft_logits = make(0)
    sp = Sampling(temperature=0.8)
    q = torch.softmax(draft_logits / sp.temperature, -1)
    p0 = probs(target_logits, sp)[0]
    gen = torch.Generator().manual_seed(1)
    counts = torch.zeros(V)
    trials = 30000
    for _ in range(trials):
        d = draw_drafts(cand, q, gen)
        acc, nxt = accept_sampled(d, target_logits, cand, q, sp, gen)
        counts[int(d[0]) if acc >= 1 else nxt] += 1
    assert (counts / trials - p0).abs().max() < 0.012


def test_second_token_given_first_accepted_matches_target():
    target_logits, cand, draft_logits = make(5)
    sp = Sampling(temperature=1.0)
    q = torch.softmax(draft_logits / sp.temperature, -1)
    p1 = probs(target_logits, sp)[1]
    gen = torch.Generator().manual_seed(4)
    counts, kept = torch.zeros(V), 0
    for _ in range(40000):
        d = draw_drafts(cand, q, gen)
        acc, nxt = accept_sampled(d, target_logits, cand, q, sp, gen)
        if acc >= 1:  # token 2 is d[1] if accepted, else the correction
            counts[int(d[1]) if acc >= 2 else nxt] += 1
            kept += 1
    assert kept > 2000 and (counts / kept - p1).abs().max() < 0.02


def test_top_p_target_is_respected():
    target_logits, cand, draft_logits = make(3)
    sp = Sampling(temperature=1.0, top_p=0.5)
    q = torch.softmax(draft_logits, -1)
    allowed = probs(target_logits, sp)[0] > 0
    gen = torch.Generator().manual_seed(2)
    for _ in range(2000):
        d = draw_drafts(cand, q, gen)
        acc, nxt = accept_sampled(d, target_logits, cand, q, sp, gen)
        assert allowed[int(d[0]) if acc >= 1 else nxt]
