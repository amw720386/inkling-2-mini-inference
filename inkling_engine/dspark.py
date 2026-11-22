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

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .ops.draft_triton import add_rms, qk_norm_rope, silu_mul
from .ops.gemv_triton import Fp8Weight
from .ops.markov_triton import MarkovHead
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
# The base logits that pick each row's candidates are scored over a shortlist of the most frequent tokens (plus the
# special tokens) instead of the whole vocabulary: data/draft_shortlist.bin ranks token ids by frequency in code and
# English text (tests/gpu/build_shortlist.py; the idea is ingine's). 0 = the whole vocabulary.
SHORTLIST = int(os.environ.get("INKLING_DRAFT_SHORTLIST", "32768"))
SPECIAL_FROM = 199998  # Inkling's special tokens (chat markers, eos, pad) sit at the top of the vocabulary


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
    def __init__(self, model_dir: str | Path, target: ModelWeights, max_ctx: int, device="cuda", slots: int = 1):
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
        # FP8 copies for the drafter's big GEMVs (base logits over the vocab, layer projections)
        V0 = self.head.shape[0]
        rank_file = Path(__file__).parent / "data" / "draft_shortlist.bin"
        if SHORTLIST and rank_file.exists():
            ranked = torch.frombuffer(bytearray(rank_file.read_bytes()), dtype=torch.int32).long()
            ids = torch.cat([ranked[:SHORTLIST], torch.arange(SPECIAL_FROM, V0)])
            ids = torch.unique(ids[ids < V0])  # sorted, no duplicates
            self.short_ids = ids.to(device)
        else:
            self.short_ids = torch.arange(V0, device=device)
        self.head8 = Fp8Weight(self.head[self.short_ids])  # FP8 rows of the shortlist
        for L in self.layers:
            for k in ("qkv", "gate_up", "wo", "w_down"):
                L[k] = Fp8Weight(L[k])
        mh = "markov_head."
        self.w1 = t[mh + "markov_w1.weight"]
        self.rnn_w, self.rnn_b = t[mh + "rnn_proj.weight"], t[mh + "rnn_proj.bias"]
        # step-dependent columns [state, W1[prev]] vs step-independent [h_i, f_rnn] (precomputed per round)
        self.rnn_w_step = self.rnn_w[:, :2 * RANK].contiguous()
        self.rnn_w_rest = self.rnn_w[:, 2 * RANK:].t().contiguous()  # bf16 [4608, 768] (tensor-core GEMM)
        self.f_norm = t[mh + "dflare_norm.weight"]
        D, V = self.embed.shape[1], self.head.shape[0]
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.qkv_out = z(BLOCK, self.layers[0]["qkv"].shape[0], dt=torch.bfloat16)
        self.gu_out = z(BLOCK, self.layers[0]["gate_up"].shape[0], dt=torch.bfloat16)
        self.base_out = z(GAMMA, self.short_ids.numel(), dt=torch.bfloat16)
        self.h_out = z(BLOCK, D, dt=torch.bfloat16)
        self.draft_logits = torch.full((GAMMA, V), float("-inf"), device=device)
        self.ctx = min(max_ctx, DRAFT_CTX)
        kv = dict(dtype=torch.bfloat16, device=device)
        # one draft context per engine slot (a request decoding alone in any slot can speculate)
        self.slots = slots
        self.ctxs = []
        for _ in range(slots):
            k_ctx = {l: torch.zeros((WINDOW + BLOCK) if l in LOCAL else self.ctx, KV_HEADS, HEAD_DIM, **kv)
                     for l in range(3)}
            self.ctxs.append({"k": k_ctx, "v": {l: torch.zeros_like(k_ctx[l]) for l in range(3)},
                              "pos": {l: torch.full((WINDOW,), -1, dtype=torch.long, device=device) for l in LOCAL},
                              "f": torch.zeros(D, **kv)})
        self.lengths = [0] * slots
        self._use(0)
        self.anchor = torch.zeros(1, dtype=torch.long, device=device)
        self.P = torch.zeros(1, dtype=torch.long, device=device)
        self.draft_tokens = torch.zeros(GAMMA, dtype=torch.long, device=device)
        self.conf = torch.zeros(GAMMA, device=device)
        self.bucket = self.ctx
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.commit_feats = torch.zeros(BLOCK, 4, D, **kv)
        self.commit_start = torch.zeros(1, dtype=torch.long, device=device)
        self.commit_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.temperature = torch.zeros(1, device=device)  # <= 0: greedy drafts, else Gumbel-max at T
        self.seed = torch.zeros(1, dtype=torch.int32, device=device)  # advanced every draft (in-graph)
        self.cand_buf = torch.zeros(GAMMA, CANDIDATES, dtype=torch.int64, device=device)
        self.fused_head = True  # Triton Markov head (2 launches per step); False: the torch reference loop
        self.head_kernels = MarkovHead(GAMMA, RANK, CANDIDATES, device)

    def _use(self, slot: int):
        """Point the commit/draft code (and graphs captured now) at `slot`'s context buffers."""
        c = self.ctxs[slot]
        self.cur = slot
        self.k_ctx, self.v_ctx, self.ctx_pos, self.f_rnn = c["k"], c["v"], c["pos"], c["f"]

    @property
    def length(self) -> int:  # slot 0 (prefix reuse lives there)
        return self.lengths[0]

    @length.setter
    def length(self, n: int):
        self.lengths[0] = n

    def reset(self, slot: int | None = None):
        for s_ in (range(self.slots) if slot is None else (slot,)):
            for l in LOCAL:
                self.ctxs[s_]["pos"][l].fill_(-1)
            self.lengths[s_] = 0

    # ------------------------------------------------------------------ commit
    @torch.inference_mode()
    def commit(self, feats: torch.Tensor, start: int, slot: int = 0):
        """Target rows (feats [n, 4, H] at positions start..start+n-1) become `slot`'s draft context."""
        n = feats.shape[0]
        self._use(slot)
        if (n, slot) in self.commit_graphs:
            self.commit_feats[:n].copy_(feats)
            self.commit_start.fill_(start)
            self.commit_graphs[(n, slot)].replay()
        else:
            self._commit(feats, torch.arange(start, start + n, device=feats.device))
        self.lengths[slot] = start + n

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
    def _project(self, L, xn, positions, out):
        """q/k/v of block rows: q, k = rope(rms_head * norm), v as is."""
        n = xn.shape[0]
        qkv = dense_mm(xn, L["qkv"], out)
        q, k = qk_norm_rope(qkv, positions, L["q_norm"], L["k_norm"], HEADS, KV_HEADS, HEAD_DIM, ROPE_THETA, EPS)
        v = qkv[:, (HEADS + KV_HEADS) * HEAD_DIM:].reshape(n, KV_HEADS, HEAD_DIM)
        return q, k, v

    def _attn_core(self, l, q, k, v, positions):
        """Bidirectional block attention of one slot's 8 rows over its context (the current _use slot)."""
        n = q.shape[0]
        P = positions[:1]
        keys, vals = self.k_ctx[l], self.v_ctx[l]
        if l in LOCAL:
            keys[WINDOW:] = k
            vals[WINDOW:] = v
            cpos = self.ctx_pos[l]
            ring_ok = (cpos >= 0) & (cpos < P) & (cpos >= P - WINDOW)
            valid = torch.cat([ring_ok, torch.ones(n, dtype=torch.bool, device=q.device)])
        else:
            slots = positions % self.ctx
            keys.index_copy_(0, slots, k)
            vals.index_copy_(0, slots, v)
            keys, vals = keys[:self.bucket], vals[:self.bucket]
            valid = torch.arange(self.bucket, device=q.device) < P + n
        # grouped attention per KV head (SDPA's enable_gqa would copy K/V to all 24 query heads: ~0.8 GB at the
        # 64k bucket): rows (head in group, token) of KV head j attend to its keys
        grp = HEADS // KV_HEADS
        qh = q.view(n, KV_HEADS, grp, HEAD_DIM).permute(1, 2, 0, 3).reshape(KV_HEADS, grp * n, HEAD_DIM)
        s = torch.matmul(qh, keys.permute(1, 2, 0)).float() * HEAD_DIM ** -0.5  # [KVH, grp*n, L]
        p = torch.softmax(s.masked_fill(~valid[None, None, :], float("-inf")), -1).to(torch.bfloat16)
        o = torch.matmul(p, vals.transpose(0, 1))  # [KVH, grp*n, D]
        return o.view(KV_HEADS, grp, n, HEAD_DIM).permute(2, 0, 1, 3).reshape(n, HEADS * HEAD_DIM)

    def _attention(self, l, L, xn, positions):
        q, k, v = self._project(L, xn, positions, self.qkv_out)
        return dense_mm(self._attn_core(l, q, k, v, positions).contiguous(), L["wo"], self.h_out)

    @torch.inference_mode()
    def _draft(self):
        """Greedy drafts for P+1..P+7 from the anchor at P -> draft_tokens [7], conf [7], draft_logits."""
        dev = self.mask_embed.device
        x0 = torch.cat([bf(rms(self.embed[self.anchor].float(), self.embed_norm)),
                        self.mask_embed[None].expand(GAMMA, -1)])
        positions = self.P + torch.arange(BLOCK, device=dev)
        res, xn = add_rms(x0.contiguous(), None, self.layers[0]["in_norm"], EPS)
        for l, L in enumerate(self.layers):
            res, xn = add_rms(res, self._attention(l, L, xn, positions), L["post_norm"], EPS)
            gu = dense_mm(xn, L["gate_up"], self.gu_out)
            m = dense_mm(silu_mul(gu), L["w_down"], self.h_out)
            nxt = self.layers[l + 1]["in_norm"] if l < 2 else self.final_norm
            res, xn = add_rms(res, m, nxt, EPS)
        h = xn[1:].contiguous()  # [7, D]
        base = dense_mm(h, self.head8, self.base_out).float()
        cand_base, ci = base.topk(CANDIDATES, dim=-1, sorted=False)  # order is irrelevant to the head
        cand = self.short_ids[ci]  # shortlist positions -> token ids
        rest = (torch.cat([h, self.f_rnn[None].expand(GAMMA, -1)], dim=1) @ self.rnn_w_rest).float()  # [7, 768]
        if self.fused_head:
            self.cand_buf.copy_(cand)
            self.head_kernels.run(rest.contiguous(), cand_base.contiguous(), self.cand_buf, self.rnn_w_step,
                                  self.rnn_b, self.w1, self.anchor, self.fold_w, self.fold_b, self.draft_tokens,
                                  self.conf, self.temperature, self.seed)
            self.seed.add_(1)
            return self.draft_tokens, self.conf
        cand_w, cand_b = self.fold_w[cand], self.fold_b[cand]
        state = torch.zeros(RANK, dtype=torch.bfloat16, device=dev)
        prev = self.anchor
        tokens, all_logits = [], []
        for i in range(GAMMA):
            z = torch.cat([state, self.w1[prev][0]])
            raw = bf(z.float() @ self.rnn_w_step.float().T + rest[i] + self.rnn_b.float()).float()
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

    # -------------------------------------------------------- multi-slot draft
    def _init_multi(self, device="cuda"):
        D, V = self.embed.shape[1], self.head.shape[0]
        n = self.slots
        z = lambda *s_, dt=torch.float32: torch.zeros(*s_, dtype=dt, device=device)  # noqa: E731
        self.m_anchor = z(n, dt=torch.long)
        self.m_P = z(n, dt=torch.long)
        self.m_temp = z(n)
        self.m_tok = z(n, GAMMA, dt=torch.long)
        self.m_conf = z(n, GAMMA)
        self.m_cand = z(n, GAMMA, CANDIDATES, dt=torch.long)
        self.m_qkv = z(BLOCK * n, self.qkv_out.shape[1], dt=torch.bfloat16)
        self.m_gu = z(BLOCK * n, self.gu_out.shape[1], dt=torch.bfloat16)
        self.m_h = z(BLOCK * n, D, dt=torch.bfloat16)
        self.m_o = z(BLOCK * n, HEADS * HEAD_DIM, dt=torch.bfloat16)
        self.m_base = z(GAMMA * n, self.short_ids.numel(), dt=torch.bfloat16)
        self.m_heads = [MarkovHead(GAMMA, RANK, CANDIDATES, device) for _ in range(n)]
        self.multi_graphs: dict = {}

    @torch.inference_mode()
    def _draft_multi(self, slots: tuple):
        """Drafts of several slots at once: the layers' and the head's GEMVs run once over all blocks
        (weights read once); attention and the Markov head run per slot."""
        k = len(slots)
        dev = self.mask_embed.device
        anchors = self.m_anchor[:k]
        rows = BLOCK * k
        emb = bf(rms(self.embed[anchors].float(), self.embed_norm))  # [k, D]
        x0 = torch.cat([emb[:, None], self.mask_embed[None, None].expand(k, GAMMA, -1)], 1).reshape(rows, -1)
        positions = (self.m_P[:k, None] + torch.arange(BLOCK, device=dev)[None]).reshape(-1)
        res, xn = add_rms(x0.contiguous(), None, self.layers[0]["in_norm"], EPS)
        for l, L in enumerate(self.layers):
            q, kk, v = self._project(L, xn, positions, self.m_qkv[:rows])
            for i, s_ in enumerate(slots):
                self._use(s_)
                r = slice(BLOCK * i, BLOCK * (i + 1))
                self.m_o[r] = self._attn_core(l, q[r], kk[r], v[r], positions[r])
            attn = dense_mm(self.m_o[:rows], L["wo"], self.m_h[:rows])
            res, xn = add_rms(res, attn, L["post_norm"], EPS)
            gu = dense_mm(xn, L["gate_up"], self.m_gu[:rows])
            m = dense_mm(silu_mul(gu), L["w_down"], self.m_h[:rows])
            nxt = self.layers[l + 1]["in_norm"] if l < 2 else self.final_norm
            res, xn = add_rms(res, m, nxt, EPS)
        h = xn.reshape(k, BLOCK, -1)[:, 1:].reshape(GAMMA * k, -1).contiguous()
        base = dense_mm(h, self.head8, self.m_base[:GAMMA * k]).float()
        for i, s_ in enumerate(slots):
            r = slice(GAMMA * i, GAMMA * (i + 1))
            cand_base, ci = base[r].topk(CANDIDATES, dim=-1, sorted=False)
            cand = self.short_ids[ci]
            f = self.ctxs[s_]["f"]
            rest = (torch.cat([h[r], f[None].expand(GAMMA, -1)], dim=1) @ self.rnn_w_rest).float()
            self.m_cand[i].copy_(cand)
            self.m_heads[i].run(rest.contiguous(), cand_base.contiguous(), self.m_cand[i], self.rnn_w_step,
                                self.rnn_b, self.w1, self.m_anchor[i:i + 1], self.fold_w, self.fold_b, self.m_tok[i],
                                self.m_conf[i], self.m_temp[i:i + 1], self.seed)
        self.seed.add_(1)

    def draft_multi(self, anchors: list[int], Ps: list[int], temps: list[float], slots: list[int]):
        """Drafts for several slots in one pass. Returns per slot (tokens [7], conf [7], candidate ids
        [7, K], candidate logits [7, K]) views, valid until the next draft."""
        if not hasattr(self, "multi_graphs"):
            self._init_multi()
        k = len(slots)
        self.m_anchor[:k].copy_(torch.tensor(anchors, dtype=torch.long), non_blocking=True)
        self.m_P[:k].copy_(torch.tensor(Ps, dtype=torch.long), non_blocking=True)
        self.m_temp[:k].copy_(torch.tensor(temps, dtype=torch.float32), non_blocking=True)
        # context bucket of the longest sequence (as single drafts), so the global layer reads only that much
        bucket = next((b for b in getattr(self, "buckets", []) if b >= max(Ps) + BLOCK), self.ctx)
        key = (tuple(slots), bucket)
        if self.graphs:  # graphs mode: capture this slot set / bucket on first use
            g = self.multi_graphs.get(key)
            if g is None:
                self.bucket = bucket
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    self._draft_multi(key[0])
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._draft_multi(key[0])
                self.multi_graphs[key] = g
            g.replay()
        else:
            self.bucket = bucket
            self._draft_multi(key[0])
        return [(self.m_tok[i], self.m_conf[i], self.m_cand[i], self.m_heads[i].logits) for i in range(k)]

    def capture(self, pool=None):
        """Draft graphs per (context bucket, slot) and commit graphs per (rows, slot); one memory pool."""
        dev = self.mask_embed.device
        buckets, b = [], 1024
        while b < self.ctx:
            buckets.append(b)
            b *= 2
        buckets.append(self.ctx)
        commit = {n: (lambda n=n: self._commit(self.commit_feats[:n], self.commit_start + torch.arange(n, device=dev)))
                  for n in range(1, BLOCK + 1)}
        pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self.pool = pool
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for slot in range(self.slots):
                self._use(slot)
                for B in buckets:
                    self.bucket = B
                    self._draft()
                for fn in commit.values():
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        for slot in range(self.slots):
            self._use(slot)
            for B in buckets:
                self.bucket = B
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    self._draft()
                self.graphs[(B, slot)] = g
            for n, fn in commit.items():
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    fn()
                self.commit_graphs[(n, slot)] = g
        self.bucket = self.ctx
        self.buckets = buckets
        # multi-slot drafts for every slot set of 2+ slots (batched speculation), captured now, not mid-serving
        if self.slots > 1:
            from itertools import combinations

            self._init_multi()
            for k in range(2, self.slots + 1):
                for combo in combinations(range(self.slots), k):
                    for B in buckets:  # P = B - BLOCK selects bucket B
                        self.draft_multi([0] * k, [B - BLOCK] * k, [0.0] * k, list(combo))
        self._use(0)
        self.reset()

    def draft_distribution(self, n: int, temperature: float):
        """(candidate ids [n, K], q over them [n, K]) of the last draft's first n steps at `temperature`."""
        return self.cand_buf[:n], torch.softmax(self.head_kernels.logits[:n] / temperature, -1)

    def draft(self, anchor: int, P: int, temperature: float = 0.0, slot: int = 0):
        self.anchor.fill_(anchor)
        self.P.fill_(P)
        self.temperature.fill_(temperature)
        self._use(slot)
        if not self.graphs:
            return self._draft()
        bucket = next((b for b in self.buckets if b >= P + BLOCK), self.ctx)
        self.graphs[(bucket, slot)].replay()
        return self.draft_tokens, self.conf
