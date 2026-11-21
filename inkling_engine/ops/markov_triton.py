"""DSpark Markov head, one step = two Triton launches (CUDA-graph capturable).

cell  : z = [state, W1[prev]] (prev = anchor at step 0, else the previous step's pick);
        raw = bf16(z . Wstep^T + rest[i] + b); gate = bf16(sigmoid(raw_g)), cell = bf16(tanh(raw_c));
        state' = bf16(bf16(gate * state) + bf16(bf16(1 - gate) * cell)); latent = bf16(tanh(raw_l)).
        Program j owns rows {j, R + j, 2R + j}, so no cross-program dependency inside a step.
logits: bias_c = bf16(fold_b[c] + bf16(fold_w[c] . latent)), logit_c = bf16(30 tanh(bf16(base_c + bias_c) / 30))
        over the step's 1024 candidates; argmax and max-softmax confidence are finished by the last
        program (atomic counter) which writes the pick and the drafted token.
Rounding points follow dspark.py's torch loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _raw_rows(Wstep, Rest, Bias, step, rows, k, st, w1, R: tl.constexpr):
    """bf16-rounded raw[rows] = [state, W1[prev]] . Wstep[rows]^T + rest[step, rows] + b[rows]."""
    wa = tl.load(Wstep + rows[:, None] * (2 * R) + k[None, :]).to(tl.float32)
    wb = tl.load(Wstep + rows[:, None] * (2 * R) + R + k[None, :]).to(tl.float32)
    v = tl.sum(wa * st[None, :] + wb * w1[None, :], 1) + tl.load(Rest + step * 3 * R + rows)
    v = v + tl.load(Bias + rows).to(tl.float32)
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _cell_kernel(step, Wstep, Rest, Bias, W1, Anchor, Tokens, State, Latent,
                 R: tl.constexpr, BI: tl.constexpr):
    pid = tl.program_id(0)
    j = pid * BI + tl.arange(0, BI)
    k = tl.arange(0, R)
    if step == 0:
        prev = tl.load(Anchor).to(tl.int64)
    else:
        prev = tl.load(Tokens + step - 1).to(tl.int64)
    st = tl.load(State + step * R + k).to(tl.float32)  # state row `step`
    w1 = tl.load(W1 + prev * R + k).to(tl.float32)
    raw_g = _raw_rows(Wstep, Rest, Bias, step, 0 * R + j, k, st, w1, R)
    raw_c = _raw_rows(Wstep, Rest, Bias, step, 1 * R + j, k, st, w1, R)
    raw_l = _raw_rows(Wstep, Rest, Bias, step, 2 * R + j, k, st, w1, R)
    gate = tl.sigmoid(raw_g).to(tl.bfloat16).to(tl.float32)
    cell = _tanh(raw_c).to(tl.bfloat16).to(tl.float32)
    sj = tl.load(State + step * R + j).to(tl.float32)
    a = (gate * sj).to(tl.bfloat16).to(tl.float32)
    b = ((1.0 - gate).to(tl.bfloat16).to(tl.float32) * cell).to(tl.bfloat16).to(tl.float32)
    tl.store(State + (step + 1) * R + j, (a + b).to(tl.bfloat16))
    tl.store(Latent + j, _tanh(raw_l).to(tl.bfloat16))


@triton.jit
def _logits_kernel(step, CandBase, CandIdx, FoldW, FoldB, Latent, Logits, Part, Counter, Tokens, Conf,
                   K: tl.constexpr, R: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    c = pid * BC + tl.arange(0, BC)
    k = tl.arange(0, R)
    idx = tl.load(CandIdx + step * K + c).to(tl.int64)
    lat = tl.load(Latent + k).to(tl.float32)
    fw = tl.load(FoldW + idx[:, None] * R + k[None, :]).to(tl.float32)
    dot = tl.sum(fw * lat[None, :], 1).to(tl.bfloat16).to(tl.float32)
    bias = (tl.load(FoldB + idx).to(tl.float32) + dot).to(tl.bfloat16).to(tl.float32)
    z = (tl.load(CandBase + step * K + c) + bias).to(tl.bfloat16).to(tl.float32)
    lg = (30.0 * _tanh(z / 30.0)).to(tl.bfloat16).to(tl.float32)
    tl.store(Logits + step * K + c, lg)
    m = tl.max(lg, 0)
    am = tl.min(tl.where(lg == m, c, K), 0)
    se = tl.sum(tl.exp(lg - m), 0)
    tl.store(Part + pid * 3, m)
    tl.store(Part + pid * 3 + 1, am.to(tl.float32))
    tl.store(Part + pid * 3 + 2, se)
    done = tl.atomic_add(Counter, 1, sem="acq_rel")
    if done == NPROG - 1:
        p = tl.arange(0, NPROG)
        pm = tl.load(Part + p * 3, volatile=True)
        pa = tl.load(Part + p * 3 + 1, volatile=True)
        ps = tl.load(Part + p * 3 + 2, volatile=True)
        gm = tl.max(pm, 0)
        best = tl.min(tl.where(pm == gm, pa, K * 1.0), 0).to(tl.int32)  # lowest index among ties
        tot = tl.sum(ps * tl.exp(pm - gm), 0)
        tl.store(Tokens + step, tl.load(CandIdx + step * K + best))
        tl.store(Conf + step, 1.0 / tot)
        tl.atomic_xchg(Counter, 0)


class MarkovHead:
    """Device buffers for one drafter; run() executes the GAMMA steps (14 launches)."""

    def __init__(self, gamma: int, rank: int, k: int, device="cuda"):
        self.gamma, self.rank, self.k = gamma, rank, k
        self.state = torch.zeros(gamma + 1, rank, dtype=torch.bfloat16, device=device)
        self.latent = torch.zeros(rank, dtype=torch.bfloat16, device=device)
        self.logits = torch.zeros(gamma, k, dtype=torch.float32, device=device)
        self.bc = 64
        self.nprog = k // self.bc
        self.part = torch.zeros(self.nprog * 3, dtype=torch.float32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)

    def run(self, rest, cand_base, cand_idx, w_step, bias, w1, anchor, fold_w, fold_b, tokens, conf):
        """rest fp32 [G, 3R]; cand_base fp32 [G, K]; cand_idx int32/int64 [G, K]; tokens int64 [G]; conf fp32 [G]."""
        r, kk = self.rank, self.k
        self.state[0].zero_()
        for i in range(self.gamma):
            _cell_kernel[(r // 32,)](i, w_step, rest, bias, w1, anchor, tokens, self.state, self.latent,
                                     R=r, BI=32, num_warps=4)
            _logits_kernel[(self.nprog,)](i, cand_base, cand_idx, fold_w, fold_b, self.latent, self.logits, self.part,
                                          self.counter, tokens, conf, K=kk, R=r, BC=self.bc, NPROG=self.nprog,
                                          num_warps=4)
