"""Fused fast forward: ~12 kernel launches per visit over preallocated workspaces.

Same math as model.InklingModel (the reference); tests compare the two.
The residual stream lives in `ws.blk` (see ops/fused_triton.py); the Python side only
tracks how many blocks exist, which is identical every step, so the whole step can be
captured in one CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .batch import Batch
from .cache import InklingCache
from .config import InklingConfig
from .fast import StepMeta, build_step_meta, pick_splits
from .ops import fused_triton as fk
from .ops import ref
from .ops.attn_triton import inkling_attention, store_kv
from .ops.conv_triton import _conv_state_kernel, conv_candidates
from .ops.gemv_triton import Fp8Weight
from .ops.gemv_triton import mm as dense_mm
from .ops.moe_triton import GroupedScratch, moe_decode, moe_grouped_small, moe_prefill
from .weights import ModelWeights

DIRECT_CACHE_MAX_NEW = 64  # ring slack: chunks this small may write the cache before attention
DECODE_MOE_MAX_TOKENS = 16
GROUPED_MIN_TOKENS = 4  # below this the per-(token, expert) GEMV is faster


@dataclass
class LayerExtras:
    qkv_norm: torch.Tensor  # [3, H] bf16 (local: attn_norm repeated)
    router_bf16: torch.Tensor  # [E, H]


class Workspace:
    def __init__(self, cfg: InklingConfig, max_tokens: int, max_out: int, n_split: int, device="cuda"):
        t, h = max_tokens, cfg.hidden_size
        bf, f32 = torch.bfloat16, torch.float32
        z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=device)
        self.max_tokens = t
        self.blk = z(5, t, h)
        self.bank = z(2, t, h)
        self.resid = z(t, h, dt=f32)
        self.ar = z(3, t, h, dt=f32)
        self.xn = z(3, t, h)
        self.msgs = z(4, t, h, dt=f32)
        self.tmp = z(3, t, h, dt=f32)
        self.gbuf = z(t, 4, cfg.gate_rank, dt=f32)
        nqg = 2 * cfg.q_dim + cfg.rel_dim + cfg.num_heads
        self.proj = z(t, nqg + 2 * cfg.kv_dim)
        self.qgrs = z(t, nqg)
        self.kv_lin = z(2, t, cfg.kv_dim)
        self.dyn = z(2, t, cfg.num_kv_heads * cfg.sconv_kernel)
        self.q = z(t, cfg.num_heads, cfg.head_dim)
        self.rel = z(t, cfg.num_heads, cfg.rel_extent)
        self.k = z(t, cfg.num_kv_heads, cfg.head_dim)
        self.v = z(t, cfg.num_kv_heads, cfg.head_dim)
        self.attn = z(t, cfg.q_dim)
        self.o = z(t, h)
        self.mi = z(t, h, dt=f32)
        self.x = z(t, h)
        self.rlog = z(t, cfg.n_experts, dt=f32)
        self.ids = z(t, cfg.top_k, dt=torch.int32)
        self.wts = z(t, cfg.top_k, dt=f32)
        self.m = z(t, h)
        small = t <= DECODE_MOE_MAX_TOKENS
        self.act = z(t * cfg.top_k, cfg.moe_inter) if small else None
        self.rows = z(t * cfg.top_k, h, dt=f32) if small else None
        self.grouped = GroupedScratch(t, cfg.top_k, cfg.n_experts, h, cfg.moe_inter, device) if small and t >= GROUPED_MIN_TOKENS else None
        self.hidden = z(t, h)
        self.logits = z(max_out, cfg.unpadded_vocab_size, dt=f32)
        self.logits_bf16 = z(max_out, cfg.unpadded_vocab_size)
        self.taps = z(t, 4, h)  # DSpark features of each row (see FusedModel.tap_visits)
        self.part_acc = z(t, cfg.num_heads, n_split, cfg.head_dim, dt=f32) if n_split > 1 else None
        self.part_ml = z(t, cfg.num_heads, n_split, 2, dt=f32) if n_split > 1 else None


class FusedModel:
    def __init__(self, cfg: InklingConfig, weights: ModelWeights, dense_fp8: bool = False):
        """dense_fp8: row-scaled E4M3 copies of the attention projections and the LM head for decode-sized
        GEMVs (halves their bytes; lossy w.r.t. the bf16 checkpoint, off by default). Prefill keeps bf16."""
        self.cfg, self.w = cfg, weights
        self.dense_fp8 = dense_fp8
        self.fp8 = {}
        if dense_fp8:
            for lw in weights.layers:
                for name in ("w_all", "w_qgrs", "wk", "wv", "wo"):
                    self.fp8[(lw.layer, name)] = Fp8Weight(getattr(lw, name))
            self.fp8["unembed"] = Fp8Weight(weights.unembed)
        self.visits = cfg.visits
        self.extras = []
        for lw in weights.layers:
            if lw.is_local:
                qn = lw.attn_norm[None].expand(3, -1).contiguous()
            else:
                qn = torch.stack([lw.q_norm, lw.k_norm, lw.v_norm])
            self.extras.append(LayerExtras(qn, lw.router_w.to(torch.bfloat16).contiguous()))
        self.nqg = 2 * cfg.q_dim + cfg.rel_dim + cfg.num_heads
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.capture_visits: list[int] = []
        self.captures: dict[int, torch.Tensor] = {}
        # DSpark taps: the partial block entering these visits, then the final normed hidden -> ws.taps
        tl_ids = list(cfg.mtp.get("target_layer_ids", [5, 23, 41, 48]))
        self.tap_visits = {v: j for j, v in enumerate(tl_ids[:-1])}
        self.write_taps = False

    # ---------------------------------------------------------------- metadata
    def step_meta(self, batch: Batch) -> StepMeta:
        max_new = max(s.n for s in batch.seqs)
        tok = 16 if max_new > 8 else 4
        n_tiles = sum((s.n + tok - 1) // tok for s in batch.seqs)
        max_keys = max(s.start + s.n for s in batch.seqs)
        nl, ng = pick_splits(max_keys, n_tiles, self.cfg.num_kv_heads, self.sms, self.cfg.sliding_window)
        return build_step_meta(batch, nl, ng)

    def _w(self, layer: int, name: str, T: int):
        """The projection weight for a T-token step (FP8 copy for decode-sized steps when enabled)."""
        if self.dense_fp8 and T <= 16:
            return self.fp8[(layer, name)]
        return getattr(self.w.layers[layer], name)

    def workspace(self, max_tokens: int, max_out: int, n_split: int = 32) -> Workspace:
        return Workspace(self.cfg, max_tokens, max_out, n_split)

    # ----------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, batch: Batch, cache: InklingCache, ws: Workspace, meta: StepMeta | None = None,
                decode_state: bool | None = None, n_out: int | None = None, verify: bool = False):
        """verify: one sequence whose conv states must stay rollback-able; candidates go to cache.cand."""
        c, w = self.cfg, self.w
        meta = meta or self.step_meta(batch)
        T = batch.num_tokens
        assert T <= ws.max_tokens
        if decode_state is None:
            decode_state = all(s.n == 1 for s in batch.seqs) and not verify
        if verify:
            assert len(batch.seqs) == 1 and not decode_state
            vslot = batch.seqs[0].slot
        direct_cache = meta.max_new <= DIRECT_CACHE_MAX_NEW
        pos = batch.positions
        blk = ws.blk[:, :T]
        ws.blk[0, :T] = ref.rms_norm(F.embedding(batch.input_ids, w.embed), w.embed_norm, c.rms_eps)
        nb = 0
        pre_nb = None
        for v in self.visits:
            layer = v.layer
            lw, ex = w.layers[layer], self.extras[layer]
            if v.index in self.capture_visits:
                self.captures[v.index] = blk[nb].clone()
            if self.write_taps and v.index in self.tap_visits:
                ws.taps[:T, self.tap_visits[v.index]].copy_(blk[nb])
            if v.is_core and layer == c.core_start:
                if v.pass_idx == 0:
                    pre_nb = nb
                else:
                    ws.bank[:, :T].copy_(blk[pre_nb + 1:nb + 1])
                    nb = pre_nb
            msgs = v.pass_idx > 0
            local = v.is_local
            nq = 1 if local else 3
            vi = v.index

            # ---- attention input
            xn = ws.xn[:, :T]
            if layer == 0:
                xn[0] = ref.rms_norm(blk[0], lw.attn_norm, c.rms_eps)
                shared = True
            elif msgs:
                ar = ws.ar[:nq, :T]
                fk.depth_attn(blk, nb + 1, w.attn_queries[layer], ar, out_f32=True)
                fk.depth_attn(ws.bank[:, :T], 2, w.loop_queries[layer - c.core_start], ws.msgs[:, :T], out_f32=True)
                hs = ar.expand(3, -1, -1) if local else ar
                fk.msg_gates(hs, ws.msgs[:3, :T], lw, 0, 3, ws.gbuf[:T], xn, c, norm_w=ex.qkv_norm,
                             tmp=ws.tmp[:, :T])
                shared = False
            else:
                fk.depth_attn(blk, nb + 1, w.attn_queries[layer], xn[:nq], norm_w=ex.qkv_norm[:nq])
                shared = local

            # ---- projections
            if shared:
                proj = ws.proj[:T]
                dense_mm(xn[0], self._w(layer, 'w_all', T), proj)
                qp, k_lin, v_lin = proj, proj[:, self.nqg:self.nqg + c.kv_dim], proj[:, self.nqg + c.kv_dim:]
            else:
                qp = ws.qgrs[:T]
                dense_mm(xn[0], self._w(layer, 'w_qgrs', T), qp)
                dense_mm(xn[1], self._w(layer, 'wk', T), ws.kv_lin[0, :T])
                dense_mm(xn[2], self._w(layer, 'wv', T), ws.kv_lin[1, :T])
                k_lin, v_lin = ws.kv_lin[0, :T], ws.kv_lin[1, :T]
            dyn = None
            if not local:
                nk = c.num_kv_heads * c.sconv_kernel
                dense_mm(xn[1] if not shared else xn[0], lw.kv_dyn_proj[:nk], ws.dyn[0, :T])
                dense_mm(xn[2] if not shared else xn[0], lw.kv_dyn_proj[nk:], ws.dyn[1, :T])
                dyn = (ws.dyn[0, :T], ws.dyn[1, :T])

            if local:
                kc, vc, am, ring = cache.local_k[v.local_index], cache.local_v[v.local_index], meta.local, cache.spec.ring
                window = c.sliding_window
            else:
                kc, vc, am, ring = cache.global_k[v.global_index], cache.global_v[v.global_index], meta.glob, 0
                window = 0
            q, rel, kk, vv = ws.q[:T], ws.rel[:T], ws.k[:T], ws.v[:T]
            fk.qkv_post(qp, 2 * c.q_dim, k_lin, v_lin, dyn, lw, cache.k_conv[vi], cache.v_conv[vi], q, rel, kk, vv,
                        meta.local, meta.tok_seq, pos, c, is_global=not local,
                        cache_kv=(kc, vc) if direct_cache else None, ring=ring, decode_state=decode_state)
            if not decode_state:
                self._conv_state(k_lin, cache.k_conv[vi], meta) if not verify else conv_candidates(k_lin, cache.k_conv[vi, vslot], cache.cand['k_conv'][vi])
                self._conv_state(v_lin, cache.v_conv[vi], meta) if not verify else conv_candidates(v_lin, cache.v_conv[vi, vslot], cache.cand['v_conv'][vi])

            # ---- attention
            sinks = qp[:, 2 * c.q_dim + c.rel_dim:2 * c.q_dim + c.rel_dim + c.num_heads]
            gate = qp[:, c.q_dim:2 * c.q_dim]
            attn = inkling_attention(q, kk, vv, kc, vc, rel, sinks, gate, am, window=window, ring=ring,
                                     scale=c.head_dim ** -0.5, out=ws.attn[:T],
                                     part_acc=None if ws.part_acc is None else ws.part_acc[:T, :, :am.n_split],
                                     part_ml=None if ws.part_ml is None else ws.part_ml[:T, :, :am.n_split])
            if not direct_cache:
                store_kv(kk, vv, kc, vc, am, ring, meta.max_new)
            o = ws.o[:T]
            dense_mm(attn, self._w(layer, 'wo', T), o)

            # ---- residual + MLP input
            boundary = c.is_block_boundary(layer)
            ns = nb + 1 if boundary else nb
            resid = ws.resid[:T]
            mi_out = ws.mi[:T] if msgs else ws.x[:T]
            fk.post_attn(o, lw.attn_conv_w, cache.attn_conv[vi], blk, ns, nb, resid, w.mlp_queries[layer],
                         None if msgs else lw.mlp_norm, mi_out, meta.local, meta.tok_seq, c,
                         boundary=boundary, decode_state=decode_state)
            if not decode_state:
                self._conv_state(o, cache.attn_conv[vi], meta) if not verify else conv_candidates(o, cache.attn_conv[vi, vslot], cache.cand['attn_conv'][vi])
            if boundary:
                nb += 1
            x = ws.x[:T]
            if msgs:
                fk.msg_gates(ws.mi[:T][None], ws.msgs[3:4, :T], lw, 3, 1, ws.gbuf[:T], x[None], c,
                             norm_w=lw.mlp_norm[None], tmp=ws.tmp[:1, :T])

            # ---- MoE
            fk.route(x, ex.router_bf16, lw.router_bias, ws.rlog, ws.ids[:T], ws.wts[:T], c.top_k, c.route_scale)
            dst = blk[nb]
            if T <= DECODE_MOE_MAX_TOKENS and ws.rows is not None:
                if T >= GROUPED_MIN_TOKENS and ws.grouped is not None:
                    rows = moe_grouped_small(x, ws.ids[:T], ws.wts[:T], lw.w13, lw.w2, c.mlp_gate_softcap,
                                             c.mlp_up_softcap, ws.grouped, ws.rows)
                else:
                    rows = moe_decode(x, ws.ids[:T], ws.wts[:T], lw.w13, lw.w2, c.mlp_gate_softcap, c.mlp_up_softcap,
                                      ws.act[:T * c.top_k], ws.rows[:T * c.top_k])
                fk.post_mlp(rows, ws.m[:T], lw.mlp_conv_w, cache.mlp_conv[vi], resid, dst, meta.local, meta.tok_seq,
                            c, decode_state=decode_state)
                m = ws.m[:T]
            else:
                m = moe_prefill(x, ws.ids[:T].long(), ws.wts[:T], lw.w13, lw.w2, c.mlp_gate_softcap, c.mlp_up_softcap)
                fk.post_mlp(None, m, lw.mlp_conv_w, cache.mlp_conv[vi], resid, dst, meta.local, meta.tok_seq, c,
                            decode_state=False)
            if not decode_state:
                self._conv_state(m, cache.mlp_conv[vi], meta) if not verify else conv_candidates(m, cache.mlp_conv[vi, vslot], cache.cand['mlp_conv'][vi])

        hidden = ws.hidden[:T]
        fk.depth_attn(blk, nb + 1, w.output_query, hidden[None], norm_w=w.final_norm[None])
        if c.num_visits in self.capture_visits:
            self.captures[c.num_visits] = hidden.clone()
        if self.write_taps:
            ws.taps[:T, 3].copy_(hidden)
        rows = hidden[batch.logit_rows]
        n = rows.shape[0]
        if n > ws.logits.shape[0]:  # eager only; graphs size their workspace up front
            ws.logits = torch.empty(n, c.unpadded_vocab_size, dtype=torch.float32, device=rows.device)
            ws.logits_bf16 = torch.empty(n, c.unpadded_vocab_size, dtype=torch.bfloat16, device=rows.device)
        dense_mm(rows, self.fp8['unembed'] if (self.dense_fp8 and n <= 16) else w.unembed, ws.logits_bf16[:n])
        logits = ws.logits[:n]
        torch.div(ws.logits_bf16[:n], c.final_logit_softcap, out=logits)
        logits.tanh_().mul_(c.final_logit_softcap)
        return logits

    def _conv_state(self, x, state, meta: StepMeta):
        cdim = x.shape[1]
        _conv_state_kernel[(meta.local.seq_slot.shape[0], (cdim + 511) // 512)](
            x, x.stride(0), state, state.stride(0), state.stride(1), meta.local.seq_slot, meta.local.seq_off,
            meta.local.seq_n, cdim, 4, BC=512, num_warps=4,
        )


class FusedDecodeGraph:
    """Captured fused decode step for `bs` sequences (one token each, any slots)."""

    def __init__(self, model: FusedModel, cache: InklingCache, bs: int, n_split_local: int = 4,
                 n_split_global: int = 16, capture: bool = True):
        from .batch import SeqChunk

        self.model, self.cache, self.bs = model, cache, bs
        seqs = [SeqChunk(slot=i, start=0, n=1) for i in range(bs)]
        self.batch = Batch(seqs=seqs, input_ids=torch.zeros(bs, dtype=torch.int64, device="cuda"),
                           positions=torch.zeros(bs, dtype=torch.int64, device="cuda"), offsets=list(range(bs)),
                           logit_rows=torch.arange(bs, device="cuda"))
        self.meta = build_step_meta(self.batch, n_split_local, n_split_global)
        self.ws = model.workspace(bs, bs, max(n_split_local, n_split_global))
        self.graph = None
        self.logits = None
        if capture:
            self._capture()

    def _run(self):
        return self.model.forward(self.batch, self.cache, self.ws, self.meta, decode_state=True)

    def _capture(self):
        c = self.cache
        sl = slice(0, self.bs)
        snap = [(t, t[:, sl].clone()) for t in (c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._run()
        for t, v in snap:
            t[:, sl].copy_(v)

    def step(self, slots, positions, tokens):
        self.batch.input_ids.copy_(torch.tensor(tokens, dtype=torch.int64), non_blocking=True)
        self.batch.positions.copy_(torch.tensor(positions, dtype=torch.int64), non_blocking=True)
        self.meta.local.seq_slot.copy_(torch.tensor(slots, dtype=torch.int32), non_blocking=True)
        self.meta.local.seq_start.copy_(torch.tensor(positions, dtype=torch.int32), non_blocking=True)
        if self.graph is None:
            return self._run()
        self.graph.replay()
        return self.logits
