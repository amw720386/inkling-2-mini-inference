"""Triton short convs (static + dynamic) vs the torch reference, including state carry."""

import pytest
import torch

from inkling_engine.batch import SeqChunk
from inkling_engine.ops import ref
from inkling_engine.ops.attn_triton import AttnMeta
from inkling_engine.ops.conv_triton import short_conv

CASES = {
    "decode": [SeqChunk(0, 10, 1), SeqChunk(2, 1, 1), SeqChunk(1, 0, 1)],
    "prefill": [SeqChunk(0, 0, 37)],
    "mixed": [SeqChunk(0, 5, 2), SeqChunk(1, 0, 20), SeqChunk(2, 99, 1)],
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("c,dyn", [(2304, False), (1024, False), (1024, True)])
def test_conv(case, c, dyn):
    seqs = CASES[case]
    torch.manual_seed(0)
    t = sum(s.n for s in seqs)
    x = torch.randn(t, c, device="cuda").to(torch.bfloat16)
    state = torch.randn(3, 3, c, device="cuda").to(torch.bfloat16)
    state0 = state.clone()
    offs, o = [], 0
    for s in seqs:
        offs.append(o)
        o += s.n
    meta = AttnMeta.build(seqs, offs, 4, 1)
    tok_seq = torch.repeat_interleave(torch.arange(len(seqs), dtype=torch.int32),
                                      torch.tensor([s.n for s in seqs])).cuda()
    if dyn:
        static = torch.randn(4, c, device="cuda").to(torch.bfloat16)
        dw = torch.randn(t, 8 * 4, device="cuda").to(torch.bfloat16)
        out = short_conv(x, state, meta, tok_seq, static=static, dyn=dw, ch_per_head=128, scale=1 / 3)
    else:
        w = torch.randn(c, 4, device="cuda").to(torch.bfloat16)
        out = short_conv(x, state, meta, tok_seq, weight=w, scale=5 ** -0.5)
    for s, off in zip(seqs, offs):
        sl = slice(off, off + s.n)
        if dyn:
            exp, new = ref.dyn_conv(x[sl], state0[s.slot], static, dw[sl], 8, 1 / 3)
        else:
            exp, new = ref.short_conv(x[sl], state0[s.slot], w, 5 ** -0.5)
        assert (out[sl].float() - exp.float()).abs().max().item() < 2e-2
        assert torch.equal(state[s.slot], new)
