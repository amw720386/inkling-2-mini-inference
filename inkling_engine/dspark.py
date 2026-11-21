"""DSpark drafter for Inkling-2-Mini (mtp.safetensors).

Ported from ingine's PyTorch reference (github.com/satellitedown/inkling-2-mini-abliterated-ingine,
engine/ingine/dspark.py @ 44bc891, Apache-2.0), which follows vLLM's and SGLang's DSpark implementations.

* Target taps per processed row: the partial block entering visits 5, 23, 41 and the final normed hidden,
  bf16 `feats [rows, 4, H]` (FusedModel.write_taps -> Workspace.taps).
* Committed rows become draft context per draft layer l: `c = rms(sum_j alpha[l, j] feats_j) * ctx_norm`,
  `k = rope(rms_head(c Wk_ctx) * k_norm_ctx)`, `v = c Wv_ctx`. Layers 0 and 2 keep the last 1024 positions,
  layer 1 a ring of `ctx` positions.
* A round drafts the block P..P+7 (row 0 = the anchor's normed embedding, rows 1-7 = mask_embed) through 3
  bidirectional layers, then a sequential Markov head picks 7 tokens among each row's top-1024 base logits.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .ops.gemv_triton import mm as dense_mm
from .weights import ModelWeights

HEADS, KV_HEADS, HEAD_DIM = 24, 8, 128
BLOCK = 8
GAMMA = BLOCK - 1
RANK = 256
LOCAL = (0, 2)
WINDOW = 1024
DRAFT_CTX = 65536
ROPE_THETA = 1e6
EPS = 1e-6
CANDIDATES = 1024


def bf(x):
    return x.to(torch.bfloat16)


def rms(x, w=None):
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return y * w.float() if w is not None else y


def mix_rmsnorm(feats, alpha_row, w):
    m = (feats.float() * alpha_row.float()[None, :, None]).sum(1)
    return bf(rms(m, w))


def rope(x, positions):
    """Neox rotary over the full head dim. x [n, heads, 128] bf16, positions [n]."""
    half = HEAD_DIM // 2
    inv = ROPE_THETA ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) * 2 / HEAD_DIM)
    ang = positions.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return bf(torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1))


class Drafter:
    def __init__(self, model_dir: str | Path, target: ModelWeights, max_ctx: int, device="cuda"):
        with safe_open(str(Path(model_dir) / "mtp.safetensors"), framework="pt", device=device) as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        self.alpha = bf(torch.softmax(t["fuse_weight"].float(), -1))  # [4 layers/rnn, 4 taps]
        self.embed, self.embed_norm, self.head = target.embed, target.embed_norm, target.unembed
        hw, hb = t["markov_head.hidden_out.weight"].double(), t["markov_head.hidden_out.bias"].double()
        fw, fb = [], []
        for s in range(0, self.head.shape[0], 16384):  # FP64 fold of the unembedding into the Markov head
            u = self.head[s:s + 16384].double()
            fw.append(bf(u @ hw))
            fb.append(bf(u @ hb))
        self.fold_w, self.fold_b = torch.cat(fw), torch.cat(fb)  # [V, 256], [V]
        del fw, fb
        self.layers = []
        for l in range(3):
            p, a = f"layers.{l}.", f"layers.{l}.self_attn."
            self.layers.append({
                "in_norm": t[p + "input_layernorm.weight"], "post_norm": t[p + "post_attention_layernorm.weight"],
                "ctx_norm": t[p + "dflare_norm.weight"],
                "qkv": torch.cat([t[a + "q_proj.weight"], t[a + "k_proj.weight"], t[a + "v_proj.weight"]]),
                "wo": t[a + "o_proj.weight"], "q_norm": t[a + "q_norm.weight"], "k_norm": t[a + "k_norm.weight"],
                "kv_ctx": torch.cat([t[a + "k_proj_ctx.weight"], t[a + "v_proj_ctx.weight"]]),
                "k_norm_ctx": t[a + "k_norm_ctx.weight"],
                "gate_up": torch.cat([t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]]),
                "w_down": t[p + "mlp.down_proj.weight"],
            })
        self.final_norm, self.mask_embed = t["norm.weight"], t["mask_embed"]
        mh = "markov_head."
        self.w1 = t[mh + "markov_w1.weight"]
        self.rnn_w, self.rnn_b = t[mh + "rnn_proj.weight"], t[mh + "rnn_proj.bias"]
        self.f_norm = t[mh + "dflare_norm.weight"]
        D, V = self.embed.shape[1], self.head.shape[0]
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.qkv_out = z(BLOCK, self.layers[0]["qkv"].shape[0], dt=torch.bfloat16)
        self.gu_out = z(BLOCK, self.layers[0]["gate_up"].shape[0], dt=torch.bfloat16)
        self.base_out = z(GAMMA, V, dt=torch.bfloat16)
        self.draft_logits = torch.full((GAMMA, V), float("-inf"), device=device)
        self.ctx = min(max_ctx, DRAFT_CTX)
        kv = dict(dtype=torch.bfloat16, device=device)
        self.k_ctx = {l: torch.zeros((WINDOW + BLOCK) if l in LOCAL else self.ctx, KV_HEADS, HEAD_DIM, **kv)
                      for l in range(3)}
        self.v_ctx = {l: torch.zeros_like(self.k_ctx[l]) for l in range(3)}
        self.ctx_pos = {l: torch.full((WINDOW,), -1, dtype=torch.long, device=device) for l in LOCAL}
        self.f_rnn = torch.zeros(D, **kv)
        self.length = 0
        self.anchor = torch.zeros(1, dtype=torch.long, device=device)
        self.P = torch.zeros(1, dtype=torch.long, device=device)
        self.draft_tokens = torch.zeros(GAMMA, dtype=torch.long, device=device)
        self.conf = torch.zeros(GAMMA, device=device)
        self.bucket = self.ctx
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.commit_feats = torch.zeros(BLOCK, 4, D, **kv)
        self.commit_start = torch.zeros(1, dtype=torch.long, device=device)
        self.commit_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def reset(self):
        for l in LOCAL:
            self.ctx_pos[l].fill_(-1)
        self.length = 0

    # ------------------------------------------------------------------ commit
    @torch.inference_mode()
    def commit(self, feats: torch.Tensor, start: int):
        """Target rows (feats [n, 4, H] at positions start..start+n-1) become draft context."""
        n = feats.shape[0]
        if n in self.commit_graphs:
            self.commit_feats[:n].copy_(feats)
            self.commit_start.fill_(start)
            self.commit_graphs[n].replay()
        else:
            self._commit(feats, torch.arange(start, start + n, device=feats.device))
        self.length = start + n

    def _commit(self, feats, positions):
        n = feats.shape[0]
        for l, L in enumerate(self.layers):
            c = mix_rmsnorm(feats, self.alpha[l], L["ctx_norm"])
            kv = bf(c @ L["kv_ctx"].T)
            k = kv[:, :KV_HEADS * HEAD_DIM].reshape(-1, KV_HEADS, HEAD_DIM)
            k = rope(bf(rms(k, L["k_norm_ctx"])), positions)
            v = kv[:, KV_HEADS * HEAD_DIM:].reshape(-1, KV_HEADS, HEAD_DIM)
            size = WINDOW if l in LOCAL else self.ctx
            keep = min(n, size)
            slots = positions[-keep:] % size
            self.k_ctx[l][slots] = k[-keep:]
            self.v_ctx[l][slots] = v[-keep:]
            if l in LOCAL:
                self.ctx_pos[l][slots] = positions[-keep:]
        self.f_rnn.copy_(mix_rmsnorm(feats[-1:], self.alpha[3], self.f_norm)[0])

    # ------------------------------------------------------------------- draft
    def _attention(self, l, L, xn, positions):
        n = xn.shape[0]
        P = positions[:1]
        qkv = dense_mm(xn, L["qkv"], self.qkv_out)
        q, k, v = qkv.split([HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
        q = rope(bf(rms(q.reshape(n, HEADS, HEAD_DIM), L["q_norm"])), positions)
        k = rope(bf(rms(k.reshape(n, KV_HEADS, HEAD_DIM), L["k_norm"])), positions)
        v = v.reshape(n, KV_HEADS, HEAD_DIM)
        keys, vals = self.k_ctx[l], self.v_ctx[l]
        if l in LOCAL:
            keys[WINDOW:] = k
            vals[WINDOW:] = v
            cpos = self.ctx_pos[l]
            ring_ok = (cpos >= 0) & (cpos < P) & (cpos >= P - WINDOW)
            valid = torch.cat([ring_ok, torch.ones(n, dtype=torch.bool, device=xn.device)])
        else:
            slots = positions % self.ctx
            keys.index_copy_(0, slots, k)
            vals.index_copy_(0, slots, v)
            keys, vals = keys[:self.bucket], vals[:self.bucket]
            valid = torch.arange(self.bucket, device=xn.device) < P + n
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], keys.transpose(0, 1)[None],
                                           vals.transpose(0, 1)[None], attn_mask=valid[None, None, None, :],
                                           enable_gqa=True)
        o = o[0].transpose(0, 1).reshape(n, HEADS * HEAD_DIM)
        return bf(bf(o) @ L["wo"].T)

    @torch.inference_mode()
    def _draft(self):
        """Greedy drafts for P+1..P+7 from the anchor at P -> draft_tokens [7], conf [7], draft_logits."""
        dev = self.mask_embed.device
        x0 = torch.cat([bf(rms(self.embed[self.anchor].float(), self.embed_norm)),
                        self.mask_embed[None].expand(GAMMA, -1)])
        positions = self.P + torch.arange(BLOCK, device=dev)
        res = x0
        xn = bf(rms(res, self.layers[0]["in_norm"]))
        for l, L in enumerate(self.layers):
            res = bf(res.float() + self._attention(l, L, xn, positions).float())
            xn = bf(rms(res, L["post_norm"]))
            gu = dense_mm(xn, L["gate_up"], self.gu_out)
            gate, up = gu.chunk(2, dim=-1)
            m = bf(bf(F.silu(gate.float()) * up.float()) @ L["w_down"].T)
            res = bf(res.float() + m.float())
            nxt = self.layers[l + 1]["in_norm"] if l < 2 else self.final_norm
            xn = bf(rms(res, nxt))
        h = xn[1:].contiguous()  # [7, D]
        base = dense_mm(h, self.head, self.base_out).float()
        cand_base, cand = base.topk(CANDIDATES, dim=-1)
        cand_w, cand_b = self.fold_w[cand], self.fold_b[cand]
        state = torch.zeros(RANK, dtype=torch.bfloat16, device=dev)
        prev = self.anchor
        tokens, all_logits = [], []
        for i in range(GAMMA):
            z = torch.cat([state, self.w1[prev][0], h[i], self.f_rnn])
            raw = bf(z @ self.rnn_w.T + self.rnn_b).float()
            gate = bf(torch.sigmoid(raw[:RANK]))
            cell = bf(torch.tanh(raw[RANK:2 * RANK]))
            state = bf(bf(gate.float() * state.float()).float() + bf(bf(1 - gate.float()).float() * cell.float()).float())
            latent = bf(torch.tanh(raw[2 * RANK:]))
            bias = bf(cand_b[i].float() + (cand_w[i] @ latent).float())
            logits = bf(30 * torch.tanh(bf(cand_base[i] + bias.float()).float() / 30)).float()
            pick = logits.argmax()
            prev = cand[i].gather(0, pick.view(1))
            tokens.append(prev)
            all_logits.append(logits)
        lg = torch.stack(all_logits)  # [7, K]
        self.conf.copy_(torch.softmax(lg, -1).amax(-1))
        self.draft_tokens.copy_(torch.cat(tokens))
        return self.draft_tokens, self.conf

    def capture(self):
        dev = self.mask_embed.device
        buckets, b = [], 1024
        while b < self.ctx:
            buckets.append(b)
            b *= 2
        buckets.append(self.ctx)
        commit = {n: (lambda n=n: self._commit(self.commit_feats[:n], self.commit_start + torch.arange(n, device=dev)))
                  for n in range(1, BLOCK + 1)}
        pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for B in buckets:
                self.bucket = B
                self._draft()
            for fn in commit.values():
                fn()
        torch.cuda.current_stream().wait_stream(s)
        for B in buckets:
            self.bucket = B
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._draft()
            self.graphs[B] = g
        self.bucket = self.ctx
        for n, fn in commit.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                fn()
            self.commit_graphs[n] = g
        self.reset()

    def draft(self, anchor: int, P: int):
        self.anchor.fill_(anchor)
        self.P.fill_(P)
        if not self.graphs:
            return self._draft()
        self.graphs[next((b for b in self.graphs if b >= P + BLOCK), self.ctx)].replay()
        return self.draft_tokens, self.conf
