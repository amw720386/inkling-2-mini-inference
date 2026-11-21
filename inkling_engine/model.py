"""Inkling-2-Mini forward pass over a static cache.

The math lives in a backend object so each piece can be swapped for a fused kernel.
`RefBackend` is pure torch and loops over sequences for the stateful ops.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .batch import Batch
from .cache import InklingCache
from .config import InklingConfig, Visit
from .ops import ref
from .weights import LayerWeights, ModelWeights


class RefBackend:
    """Pure torch. Stateful ops loop over the sequences of the batch."""

    def __init__(self, cfg: InklingConfig, exact: bool = False):
        self.cfg = cfg
        # exact: fp32 GEMMs rounded to bf16, so results do not depend on batch shape
        self.exact = exact

    # stateless ---------------------------------------------------------------
    def rms_norm(self, x, w):
        return ref.rms_norm(x, w, self.cfg.rms_eps)

    def attn_res(self, sources, queries):
        return ref.attn_res(sources, queries)

    def message_gate(self, h, m, lw: LayerWeights, which: int):
        return ref.message_gate(h, m, lw.ctl_down[which], lw.ctl_gain[which], lw.ctl_forget[which], self.cfg.rms_eps)

    def linear(self, x, w):
        if self.exact:
            return F.linear(x.to(ref.COMPUTE), w.to(ref.COMPUTE)).to(x.dtype)
        return F.linear(x, w)

    def moe(self, x, lw: LayerWeights):
        if self.exact:
            c = self.cfg
            wts, idx = ref.route(x, lw.router_w, lw.router_bias, c.top_k, c.route_scale)
            out = torch.zeros(x.shape, dtype=ref.COMPUTE, device=x.device)
            for e in torch.unique(idx).tolist():
                tok, slot = torch.where(idx == e)
                a = F.linear(x[tok].to(ref.COMPUTE), lw.w13.dequant(slice(e, e + 1), ref.COMPUTE)[0]).to(torch.bfloat16)
                a = ref.softcapped_swiglu(a, c.mlp_gate_softcap, c.mlp_up_softcap)
                y = F.linear(a.to(ref.COMPUTE), lw.w2.dequant(slice(e, e + 1), ref.COMPUTE)[0]).to(torch.bfloat16)
                out.index_add_(0, tok, y.to(ref.COMPUTE) * wts[tok, slot, None])
            return out.to(torch.bfloat16)
        c = self.cfg
        wts, idx = ref.route(x, lw.router_w, lw.router_bias, c.top_k, c.route_scale)
        return ref.moe(x, wts, idx, lw.w13, lw.w2, c.mlp_gate_softcap, c.mlp_up_softcap)

    # stateful ----------------------------------------------------------------
    def conv(self, state: torch.Tensor, x, weight, batch: Batch, out_scale: float):
        """state: [slots, K-1, C] view of this visit's conv cache."""
        out = torch.empty_like(x)
        for i, s in enumerate(batch.seqs):
            sl = batch.seq_slice(i)
            out[sl], new = ref.short_conv(x[sl], state[s.slot], weight, out_scale)
            state[s.slot] = new
        return out

    def dyn_conv(self, state, x, static_w, dyn_w, batch: Batch, out_scale: float):
        out = torch.empty_like(x)
        for i, s in enumerate(batch.seqs):
            sl = batch.seq_slice(i)
            out[sl], new = ref.dyn_conv(x[sl], state[s.slot], static_w, dyn_w[sl], self.cfg.num_kv_heads, out_scale)
            state[s.slot] = new
        return out

    def attention(self, v: Visit, q, k, vv, rel_logits, sinks, batch: Batch, cache: InklingCache):
        c = self.cfg
        out = torch.empty(q.shape[0], c.q_dim, dtype=torch.bfloat16, device=q.device)
        scale = ref.sdpa_scale(c.head_dim)
        for i, s in enumerate(batch.seqs):
            sl = batch.seq_slice(i)
            qp = batch.positions[sl]
            if v.is_local:
                ring = cache.spec.ring
                kc, vc = cache.local_k[v.local_index, s.slot], cache.local_v[v.local_index, s.slot]
                lo = max(0, s.start - (c.sliding_window - 1))
                prev = torch.arange(lo, s.start, device=q.device)
                kk = torch.cat([kc[prev % ring], k[sl]], 0)
                vvv = torch.cat([vc[prev % ring], vv[sl]], 0)
                kp = torch.cat([prev, qp], 0)
                out[sl] = ref.attention(q[sl], kk, vvv, qp, kp, rel_logits[sl], sinks[sl], c.sliding_window, scale)
                keep = min(s.n, ring)
                wpos = qp[-keep:]
                kc[wpos % ring] = k[sl][-keep:]
                vc[wpos % ring] = vv[sl][-keep:]
            else:
                kc, vc = cache.global_k[v.global_index, s.slot], cache.global_v[v.global_index, s.slot]
                end = s.start + s.n
                if end > cache.spec.max_ctx:
                    raise ValueError("sequence exceeds cache max_ctx")
                kc[s.start:end] = k[sl]
                vc[s.start:end] = vv[sl]
                kp = torch.arange(0, end, device=q.device)
                out[sl] = ref.attention(q[sl], kc[:end], vc[:end], qp, kp, rel_logits[sl], sinks[sl], None, scale)
        return out


class InklingModel:
    def __init__(self, cfg: InklingConfig, weights: ModelWeights, backend=None):
        self.cfg, self.w = cfg, weights
        self.be = backend or RefBackend(cfg)
        self.visits = cfg.visits
        self.capture_visits: list[int] = []  # DSpark feature taps (visit index; num_visits = final)
        self.debug_visit: int | None = None
        self.debug: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ one visit
    def _visit(self, v: Visit, partial, blocks, msg_bank, batch: Batch, cache: InklingCache):
        c, be = self.cfg, self.be
        lw = self.w.layers[v.layer]
        layer = v.layer
        h, d = c.num_heads, c.head_dim
        t = partial.shape[0]
        nq = 1 if v.is_local else 3

        # attention inputs ------------------------------------------------------
        if layer == 0:
            ar = partial.to(ref.COMPUTE)[:, None].expand(t, nq, -1)
        else:
            ar = be.attn_res([*blocks, partial], self.w.attn_queries[layer])  # [T, nq, H]
        msgs = None
        if msg_bank is not None:
            msgs = be.attn_res(msg_bank, self.w.loop_queries[layer - c.core_start])  # [T, 4, H]
            src = [ar[:, 0 if v.is_local else j] for j in range(3)]
            src = [be.message_gate(src[j], msgs[:, j], lw, j) for j in range(3)]
        else:
            src = [ar[:, 0 if v.is_local else j] for j in range(3)]
        if v.is_local:
            if msgs is None:
                xq = xk = xv = be.rms_norm(src[0], lw.attn_norm)
            else:
                xq, xk, xv = (be.rms_norm(s, lw.attn_norm) for s in src)
        else:
            xq = be.rms_norm(src[0], lw.q_norm)
            xk = be.rms_norm(src[1], lw.k_norm)
            xv = be.rms_norm(src[2], lw.v_norm)

        # projections -----------------------------------------------------------
        nqg = c.q_dim * 2 + c.rel_dim + h
        if xq is xk and xk is xv:
            proj = be.linear(xq, lw.w_all)
            qgrs, k_lin, v_lin = proj[:, :nqg], proj[:, nqg:nqg + c.kv_dim], proj[:, nqg + c.kv_dim:]
        else:
            qgrs = be.linear(xq, lw.w_qgrs)
            k_lin = be.linear(xk, lw.wk)
            v_lin = be.linear(xv, lw.wv)
        q = qgrs[:, : c.q_dim].reshape(t, h, d)
        gate = qgrs[:, c.q_dim: 2 * c.q_dim]
        r = qgrs[:, 2 * c.q_dim: 2 * c.q_dim + c.rel_dim].reshape(t, h, c.d_rel)
        sinks = qgrs[:, 2 * c.q_dim + c.rel_dim:]

        q = ref.l2_clip(q, c.qk_l2_clip, lw.q_scale)
        r = ref.l2_clip(r, c.rel_l2_clip, lw.rel_scale)
        if not v.is_local:
            q = ref.log_scale(q, batch.positions, c.log_scaling_alpha, c.log_scaling_n_floor)
            r = ref.log_scale(r, batch.positions, c.log_scaling_alpha, c.log_scaling_n_floor)
        if getattr(be, "exact", False):
            rel_logits = torch.matmul(r.to(ref.COMPUTE), lw.rel_proj.to(ref.COMPUTE)).to(torch.bfloat16)
        else:
            rel_logits = torch.matmul(r, lw.rel_proj)  # [T, H, R] bf16

        vi = v.index
        if v.is_local:
            k_c = be.conv(cache.k_conv[vi], k_lin, lw.k_conv_w, batch, c.conv_out_scale)
            v_c = be.conv(cache.v_conv[vi], v_lin, lw.v_conv_w, batch, c.conv_out_scale)
        else:
            nk = c.num_kv_heads * c.sconv_kernel
            dk = be.linear(xk, lw.kv_dyn_proj[:nk])
            dv = be.linear(xv, lw.kv_dyn_proj[nk:])
            k_c = be.dyn_conv(cache.k_conv[vi], k_lin, lw.k_static, dk, batch, c.dyn_conv_out_scale)
            v_c = be.dyn_conv(cache.v_conv[vi], v_lin, lw.v_static, dv, batch, c.dyn_conv_out_scale)
        kk = ref.l2_clip(k_c.reshape(t, c.num_kv_heads, d), c.qk_l2_clip)
        vv = ref.unweighted_rms(v_c.reshape(t, c.num_kv_heads, d), c.rms_eps).to(torch.bfloat16)

        if getattr(be, "fused_gate", False):
            attn = be.attention(v, q, kk, vv, rel_logits, sinks, batch, cache, gate=gate)
        else:
            attn = be.attention(v, q, kk, vv, rel_logits, sinks, batch, cache)
            attn = (attn.to(ref.COMPUTE) * torch.sigmoid(gate.to(ref.COMPUTE))).to(torch.bfloat16)
        if v.index == self.debug_visit:
            self.debug.update(xq=xq, xk=xk, xv=xv, k_lin=k_lin, k_c=k_c, v_c=v_c, q=q, attn=attn)
        o = be.linear(attn, lw.wo)
        a = be.conv(cache.attn_conv[vi], o, lw.attn_conv_w, batch, c.conv_out_scale)

        # MLP ---------------------------------------------------------------------
        boundary = c.is_block_boundary(layer)
        if boundary:
            mlp_blocks = [*blocks, partial]
            resid = a.to(ref.COMPUTE)
        else:
            mlp_blocks = blocks
            resid = partial.to(ref.COMPUTE) + a.to(ref.COMPUTE)
        mi = be.attn_res([*mlp_blocks, resid], self.w.mlp_queries[layer:layer + 1])[:, 0]
        if msgs is not None:
            mi = be.message_gate(mi, msgs[:, 3], lw, 3)
        x = be.rms_norm(mi, lw.mlp_norm)
        m = be.moe(x, lw)
        m = be.conv(cache.mlp_conv[vi], m, lw.mlp_conv_w, batch, c.conv_out_scale)
        new_partial = (resid + m.to(ref.COMPUTE)).to(torch.bfloat16)
        if v.index == self.debug_visit:
            self.debug.update(a=a, mi=mi, m=m, out=new_partial)
        if boundary:
            blocks.append(partial)
        return new_partial

    # --------------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, batch: Batch, cache: InklingCache):
        """Returns (fp32 logits for batch.logit_rows, captured hidden states)."""
        c, be = self.cfg, self.be
        if hasattr(be, "begin"):
            be.begin(batch)
        partial = be.rms_norm(F.embedding(batch.input_ids, self.w.embed), self.w.embed_norm)
        blocks: list[torch.Tensor] = []
        captures: dict[int, torch.Tensor] = {}
        msg_bank = None
        pre_partial = pre_blocks = None
        for v in self.visits:
            if v.index in self.capture_visits:
                captures[v.index] = partial
            if v.is_core and v.layer == c.core_start:
                if v.pass_idx == 0:
                    pre_partial, pre_blocks = partial, list(blocks)
                else:
                    msg_bank = [*blocks[len(pre_blocks) + 1:], partial]
                    partial, blocks = pre_partial, list(pre_blocks)
            if not v.is_core:
                msg_bank = None
            partial = self._visit(v, partial, blocks, msg_bank if v.pass_idx > 0 else None, batch, cache)
        out = be.attn_res([*blocks, partial], self.w.output_query)[:, 0]
        hidden = be.rms_norm(out, self.w.final_norm)
        if c.num_visits in self.capture_visits:
            captures[c.num_visits] = hidden
        rows = hidden[batch.logit_rows]
        logits = ref.softcap(be.linear(rows, self.w.unembed).to(ref.COMPUTE), c.final_logit_softcap)
        return logits, captures
