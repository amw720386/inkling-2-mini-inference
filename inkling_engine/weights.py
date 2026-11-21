"""Checkpoint loading into flat per-layer weight records.

Names follow the original Inkling-2-Mini checkpoint (model.llm.*). Small fp32 query
vectors are pre-scaled by hidden**-0.5. Projections that share an input are also
stored concatenated so one GEMM serves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import InklingConfig
from .quant import ExpertWeight, expert_weight_from_checkpoint

P = "model.llm."


@dataclass
class LayerWeights:
    layer: int
    is_local: bool
    # attention input norms: local -> attn_norm; global -> q/k/v norms
    attn_norm: torch.Tensor | None
    q_norm: torch.Tensor | None
    k_norm: torch.Tensor | None
    v_norm: torch.Tensor | None
    # projections (out, in), bf16
    w_qgrs: torch.Tensor  # cat(wq, wgate, wr, wsink): [q_dim + q_dim + rel_dim + H, hidden]
    wk: torch.Tensor
    wv: torch.Tensor
    w_all: torch.Tensor  # cat(w_qgrs, wk, wv) for the shared-input case
    wo: torch.Tensor
    q_scale: torch.Tensor  # [H, D]
    rel_scale: torch.Tensor  # [H, d_rel]
    rel_proj: torch.Tensor  # [d_rel, R]
    # K/V convs: local -> weight [C, K]; global -> static [K, C] + weight_proj [H_kv*K, hidden]
    k_conv_w: torch.Tensor | None
    v_conv_w: torch.Tensor | None
    k_static: torch.Tensor | None
    v_static: torch.Tensor | None
    kv_dyn_proj: torch.Tensor | None  # cat(k weight_proj, v weight_proj) [2*H_kv*K, hidden]
    attn_conv_w: torch.Tensor  # [hidden, K]
    mlp_conv_w: torch.Tensor
    mlp_norm: torch.Tensor
    router_w: torch.Tensor  # fp32 [E, hidden]
    router_bias: torch.Tensor  # fp32 [E]
    w13: ExpertWeight
    w2: ExpertWeight
    # loop controllers, order (attn/q, k, v, mlp)
    ctl_down: torch.Tensor | None  # [4, rank, 2*hidden]
    ctl_gain: torch.Tensor | None  # [4, hidden, rank]
    ctl_forget: torch.Tensor | None


@dataclass
class ModelWeights:
    embed: torch.Tensor
    embed_norm: torch.Tensor
    final_norm: torch.Tensor
    unembed: torch.Tensor  # [unpadded_vocab, hidden]
    attn_queries: list[torch.Tensor | None]  # per layer, scaled fp32 [1 or 3, hidden]; None for layer 0
    mlp_queries: torch.Tensor  # scaled fp32 [L, hidden]
    output_query: torch.Tensor  # scaled fp32 [1, hidden]
    loop_queries: torch.Tensor | None  # scaled fp32 [n_core, 4, hidden] in (q, k, v, mlp) order
    layers: list[LayerWeights]


def _open_tensors(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    files = sorted(p for p in path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {path}")
    for f in files:
        with safe_open(str(f), framework="pt", device=device) as fh:
            for name in fh.keys():
                if name.startswith(P):
                    out[name] = fh.get_tensor(name)
    return out


def load_weights(cfg: InklingConfig, path: str | Path, device: str = "cuda") -> ModelWeights:
    t = _open_tensors(Path(path), device)
    h = cfg.hidden_size
    sc = h**-0.5

    def take(name: str) -> torch.Tensor:
        return t.pop(P + name)

    layers = []
    for i in range(cfg.num_layers):
        pre = f"layers.{i}."
        local = cfg.is_local(i)
        wq, wg, wr, ws = (take(pre + f"attn.{n}.weight") for n in ("wq_du", "gate_attn", "wr_du", "attn_sink_proj"))
        wk, wv = take(pre + "attn.wk_dv.weight"), take(pre + "attn.wv_dv.weight")
        w_qgrs = torch.cat([wq, wg, wr, ws], 0).contiguous()
        if local:
            kc = take(pre + "attn.k_sconv.weight").reshape(cfg.kv_dim, -1).contiguous()
            vc = take(pre + "attn.v_sconv.weight").reshape(cfg.kv_dim, -1).contiguous()
            ks = vs = dyn = None
        else:
            kc = vc = None
            ks = take(pre + "attn.k_sconv.static_w").contiguous()
            vs = take(pre + "attn.v_sconv.static_w").contiguous()
            dyn = torch.cat([take(pre + "attn.k_sconv.weight_proj"), take(pre + "attn.v_sconv.weight_proj")], 0)
        ex = {k: v for k, v in t.items() if k.startswith(P + pre + "mlp.experts.")}
        for k in ex:
            del t[k]
        core = cfg.is_core(i)
        ctl = {}
        if core:
            for part in ("down", "gain", "forget"):
                ctl[part] = torch.stack([take(pre + f"controllers.{m}.{part}_weight") for m in ("attn", "k", "v", "mlp")])
        layers.append(
            LayerWeights(
                layer=i,
                is_local=local,
                attn_norm=take(pre + "attn_norm.weight") if local else None,
                q_norm=None if local else take(pre + "q_input_norm.weight"),
                k_norm=None if local else take(pre + "k_input_norm.weight"),
                v_norm=None if local else take(pre + "v_input_norm.weight"),
                w_qgrs=w_qgrs,
                wk=wk,
                wv=wv,
                w_all=torch.cat([w_qgrs, wk, wv], 0).contiguous(),
                wo=take(pre + "attn.wo_ud.weight"),
                q_scale=take(pre + "attn.q_l2_clip.scale"),
                rel_scale=take(pre + "attn.rel_l2_clip.scale"),
                rel_proj=take(pre + "attn.rel_logits_proj.proj"),
                k_conv_w=kc,
                v_conv_w=vc,
                k_static=ks,
                v_static=vs,
                kv_dyn_proj=dyn,
                attn_conv_w=take(pre + "attn_sconv.weight").reshape(h, -1).contiguous(),
                mlp_conv_w=take(pre + "mlp_sconv.weight").reshape(h, -1).contiguous(),
                mlp_norm=take(pre + "mlp_norm.weight"),
                router_w=take(pre + "mlp.gate.weight").float().contiguous(),
                router_bias=take(pre + "mlp.gate.bias").float(),
                w13=expert_weight_from_checkpoint(ex, P + pre + "mlp.experts.w13"),
                w2=expert_weight_from_checkpoint(ex, P + pre + "mlp.experts.w2"),
                ctl_down=ctl.get("down"),
                ctl_gain=ctl.get("gain"),
                ctl_forget=ctl.get("forget"),
            )
        )

    attn_q: list[torch.Tensor | None] = [None]
    for i in range(1, cfg.num_layers):
        q = take(f"attn_res.attn_queries.{i - 1}").float()
        attn_q.append((q.reshape(-1, h) * sc).contiguous())
    mlp_q = torch.stack([take(f"attn_res.mlp_queries.{i}").float() for i in range(cfg.num_layers)]) * sc
    loop_q = None
    if cfg.num_passes > 1:
        qq = take("middle_cycle_cross_pass_queries").float()
        kv = take("middle_cycle_cross_pass_kv_queries").float()
        loop_q = torch.stack((qq[0::2], kv[0::2], kv[1::2], qq[1::2]), dim=1) * sc
    unembed = take("unembed.weight")[: cfg.unpadded_vocab_size].contiguous()
    w = ModelWeights(
        embed=take("embed.weight"),
        embed_norm=take("embed_norm.weight"),
        final_norm=take("norm.weight"),
        unembed=unembed,
        attn_queries=attn_q,
        mlp_queries=mlp_q.contiguous(),
        output_query=(take("attn_res.output_query").float() * sc).reshape(1, h),
        loop_queries=None if loop_q is None else loop_q.contiguous(),
        layers=layers,
    )
    leftover = [k for k in t if not k.startswith(P + "visual")]
    if leftover:
        raise ValueError(f"unused checkpoint tensors: {leftover[:8]} (+{len(leftover) - 8})")
    return w
