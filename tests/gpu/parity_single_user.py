"""Token / KL parity of the one-request CUDA decode path (INKLING_SINGLE_USER).

Reference: SGLang's greedy outputs for 8 prompts (tests/reference/sglang_ref.json, original weights). Every
configuration prefills each prompt, then decodes teacher-forced on SGLang's output tokens (up to N steps), so all
configurations see the same token history. Reported per configuration:
  top1     share of steps whose argmax equals SGLang's next token
  lp       mean log-probability of SGLang's token
  KL       KL(baseline || config) over the full vocabulary, mean and max over steps (baseline = the Triton path)
  agree    share of steps with the same argmax as the baseline
A configuration passes when its top-1 agreement with SGLang is within 1.5 points of the baseline's and its mean KL
against the baseline is below 2e-2 (the baseline's own run-to-run noise from split counts is of that order, see
--noise).

Modal: modal run modal/engine_app.py::parity --args "parity_single_user.py [N] [configs]"
configs: comma list of baseline, single, noise. PARITY_VERIFY_M=2/4/8 checks the verify graphs; PARITY_SLOT=s
runs the request in cache slot s (the engine has 4).
"""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine, VerifyGraph  # noqa: E402
from inkling_engine.fused_model import FusedDecodeGraph  # noqa: E402

N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
CONFIGS = (sys.argv[3] if len(sys.argv) > 3 else "baseline,all").split(",")
if "baseline" not in CONFIGS:
    CONFIGS = ["baseline"] + CONFIGS
FLAGS = ("single_user",)
PRESETS = {
    "baseline": dict(),
    "noise": dict(splits=(8, 16)),
    "single": dict(single_user=True),
}
SLOT = int(__import__("os").environ.get("PARITY_SLOT", "0"))

items = json.load(open("/root/tests/reference/sglang_ref.json"))["items"]
eng = Engine(sys.argv[1], max_ctx=8192, spec=True, dense_fp8=True, slots=4)
m = eng.model


VERIFY_M = int(__import__("os").environ.get("PARITY_VERIFY_M", "0"))  # > 0: check verify graphs of m tokens


def prefill(ids):
    return eng.prefill(ids)[0] if SLOT == 0 else eng.prefill_slot(SLOT, ids)


def run(cfg: dict):
    for f in FLAGS:
        setattr(m, f, bool(cfg.get(f, False)))
    nl, ng = cfg.get("splits", (4, 16))
    if VERIFY_M:
        vg = VerifyGraph(m, eng.cache, 1, VERIFY_M, pool=eng.pool)
        out = []
        for it in items:
            ids, ref = it["input_ids"], it["output_ids"][:N]
            rows = []
            for off in range(0, len(ref) - VERIFY_M + 1, VERIFY_M):
                prefill(ids + ref[:off])
                vg.run(torch.tensor(ref[off:off + VERIFY_M], device="cuda"), [SLOT], [len(ids) + off])
                rows.append(torch.log_softmax(vg.ws.logits[:VERIFY_M].float(), -1))
            out.append(torch.cat(rows).half())
        for f in FLAGS:
            setattr(m, f, False)
        return out
    g = FusedDecodeGraph(m, eng.cache, 1, n_split_local=nl, n_split_global=ng, pool=eng.pool)
    out = []
    for it in items:
        ids, ref = it["input_ids"], it["output_ids"][:N]
        lg = prefill(ids).reshape(-1, eng.cfg.unpadded_vocab_size)[-1]
        rows = [torch.log_softmax(lg.float(), -1)]
        P = len(ids)
        for tok in ref[:-1]:
            lg = g.step([SLOT], [P], [tok])[0]
            rows.append(torch.log_softmax(lg.float(), -1))
            P += 1
        out.append(torch.stack(rows).half())
    for f in FLAGS:
        setattr(m, f, False)
    return out


def stats(name, res, base):
    top1 = agree = n = 0
    lp, kls = 0.0, []
    for it, r, b in zip(items, res, base):
        ref = torch.tensor(it["output_ids"][:N], device=r.device)
        k = r.shape[0]
        if VERIFY_M:  # row i of a verify predicts the token after it
            ref = torch.cat([ref[1:], ref[-1:]])
        top1 += (r.argmax(-1) == ref[:k]).sum().item()
        agree += (r.argmax(-1) == b.argmax(-1)).sum().item()
        lp += r.float().gather(1, ref[:k, None]).sum().item()
        bf, rf = b.float(), r.float()
        kls.append((bf.exp() * (bf - rf)).sum(-1))
        n += k
    kl = torch.cat(kls)
    print(f"[{name:9s}] steps {n}  top1 {top1 / n:.4f}  lp {lp / n:.4f}  KL mean {kl.mean().item():.2e} "
          f"max {kl.max().item():.2e}  agree {agree / n:.4f}", flush=True)
    return top1 / n, kl.mean().item()


base = run(PRESETS["baseline"])
b_top1, _ = stats("baseline", base, base)
ok = True
for name in CONFIGS:
    if name == "baseline":
        continue
    t1, kl = stats(name, run(PRESETS[name]), base)
    if name != "noise":
        good = t1 >= b_top1 - 0.015 and kl < 2e-2
        ok &= good
        print(f"  -> {'PASS' if good else 'FAIL'}", flush=True)
print("PARITY", "PASS" if ok else "FAIL")
