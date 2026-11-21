"""Fused Triton Markov head vs the torch loop on real drafter weights (Modal: parity --args "markov_check.py")."""

import sys
import time

import torch

sys.path.insert(0, "/root")
from inkling_engine.config import InklingConfig  # noqa: E402
from inkling_engine.dspark import Drafter  # noqa: E402
from inkling_engine.weights import load_weights  # noqa: E402

path = sys.argv[1]
cfg = InklingConfig.from_pretrained(path)
w = load_weights(cfg, path)
d = Drafter(path, w, 8192)
torch.manual_seed(0)
feats = (torch.randn(600, 4, cfg.hidden_size, device="cuda") * 2).to(torch.bfloat16)  # stand-in context
d.commit(feats, 0)
same_tok = total = 0
conf_err = 0.0
for anchor in torch.randint(1000, 60000, (20,)).tolist():
    d.anchor.fill_(anchor)
    d.P.fill_(600)
    d.fused_head = False
    t_ref, c_ref = (x.clone() for x in d._draft())
    d.fused_head = True
    t_new, c_new = (x.clone() for x in d._draft())
    # compare step by step until the first difference (later steps depend on earlier picks)
    for a, b in zip(t_ref.tolist(), t_new.tolist()):
        total += 1
        if a != b:
            break
        same_tok += 1
    conf_err = max(conf_err, (c_ref - c_new).abs().max().item())
print(f"fused vs torch: matching leading draft tokens {same_tok}/{total} (of 20 rounds x 7), max conf diff {conf_err:.2e}")

for fused in (False, True):
    d.fused_head = fused
    d.graphs.clear()
    d.capture()
    d.commit(feats, 0)
    d.anchor.fill_(1234)
    d.P.fill_(600)
    g = d.graphs[next(b for b in d.graphs if b >= 608)]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    print(f"draft graph ({'fused' if fused else 'torch'} head): {(time.time() - t0) / 50 * 1000:.2f} ms")
