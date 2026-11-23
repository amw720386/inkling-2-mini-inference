"""Kernel breakdown of one drafter commit graph (4 rows) (Modal: parity --args "prof_commit.py")."""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
eng = Engine(sys.argv[1], max_ctx=32768, spec=True)
d = eng.drafter
eng.prefill(list(range(1000, 1600)))
feats = torch.randn(4, 4, 2304, device="cuda").to(torch.bfloat16)

# fused commit vs the eager reference math
from inkling_engine import dspark as ds  # noqa: E402

d.commit(feats, 600, slot=0)
torch.cuda.synchronize()
pos = torch.arange(600, 604, device="cuda")
err = 0.0
for l, L in enumerate(d.layers):
    c = ds.mix_rmsnorm(feats, d.alpha[l], L["ctx_norm"])
    kv = ds.bf(c @ L["kv_ctx"].T)
    k = ds.rope(ds.bf(ds.rms(kv[:, :1024].reshape(-1, 8, 128), L["k_norm_ctx"])), pos)
    v = kv[:, 1024:].reshape(-1, 8, 128)
    size = ds.WINDOW if l in ds.LOCAL else d.ctx
    sl = pos % size
    err = max(err, (d.k_ctx[l][sl].float() - k.float()).abs().max().item(), (d.v_ctx[l][sl].float() - v.float()).abs().max().item())
    if l in ds.LOCAL:
        assert torch.equal(d.ctx_pos[l][sl], pos)
f_ref = ds.mix_rmsnorm(feats[-1:], d.alpha[3], d.f_norm)[0]
err = max(err, (d.f_rnn.float() - f_ref.float()).abs().max().item())
print(f"fused commit vs eager reference: max |diff| {err:.4f}")
for _ in range(5):
    d.commit(feats, 600, slot=0)
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(50):
    d.commit(feats, 600, slot=0)
e1.record()
torch.cuda.synchronize()
print(f"commit(4): {e0.elapsed_time(e1) / 50:.3f} ms")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        d.commit(feats, 600, slot=0)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))
