"""Inkling-2-Mini text config, parsed from the checkpoint's config.json.

Visits: the looped checkpoint runs layers [core_start, core_end) num_passes times.
Every (layer, pass) pair is a *visit* with its own KV cache and conv states.
Visit order (prelude, core x passes, coda) is the order layers execute.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Visit:
    index: int  # execution order, 0..num_visits-1
    layer: int  # physical layer id
    pass_idx: int  # 0 for prelude/coda and the first core pass
    is_local: bool  # sliding-window layer
    is_core: bool  # layer belongs to the loop core
    local_index: int  # index among local visits (or -1)
    global_index: int  # index among global visits (or -1)


@dataclass
class InklingConfig:
    hidden_size: int = 2304
    num_layers: int = 24
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    sliding_window: int = 1024
    local_layer_ids: tuple[int, ...] = ()
    sconv_kernel: int = 4
    rms_eps: float = 1e-6
    qk_l2_clip: float = 22.627416997969522
    rel_l2_clip: float = 8.0
    log_scaling_n_floor: float = 8192.0
    log_scaling_alpha: float = 0.1
    final_logit_softcap: float = 30.0
    attn_res_block: int = 6
    n_experts: int = 128
    top_k: int = 8
    moe_inter: int = 1152
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    core_start: int = 6
    core_end: int = 18
    num_passes: int = 3
    gate_rank: int = 64
    eos_token_id: int = 200006
    max_position: int = 1 << 20
    mtp: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def rel_dim(self) -> int:
        return self.num_heads * self.d_rel

    def is_local(self, layer: int) -> bool:
        return layer in self.local_layer_ids

    def is_core(self, layer: int) -> bool:
        return self.core_start <= layer < self.core_end

    def is_block_boundary(self, layer: int) -> bool:
        return layer % self.attn_res_block == 0

    @property
    def visits(self) -> list[Visit]:
        order: list[tuple[int, int]] = [(l, 0) for l in range(self.core_start)]
        for p in range(self.num_passes):
            order += [(l, p) for l in range(self.core_start, self.core_end)]
        order += [(l, 0) for l in range(self.core_end, self.num_layers)]
        out, nl, ng = [], 0, 0
        for i, (layer, p) in enumerate(order):
            local = self.is_local(layer)
            out.append(
                Visit(
                    index=i,
                    layer=layer,
                    pass_idx=p,
                    is_local=local,
                    is_core=self.is_core(layer),
                    local_index=nl if local else -1,
                    global_index=-1 if local else ng,
                )
            )
            nl += local
            ng += not local
        return out

    @property
    def num_visits(self) -> int:
        return self.num_layers + (self.num_passes - 1) * (self.core_end - self.core_start)

    @property
    def num_local_visits(self) -> int:
        return sum(v.is_local for v in self.visits)

    @property
    def num_global_visits(self) -> int:
        return self.num_visits - self.num_local_visits

    @property
    def conv_out_scale(self) -> float:
        return 1.0 / math.sqrt(self.sconv_kernel + 1)

    @property
    def dyn_conv_out_scale(self) -> float:
        return 1.0 / math.sqrt(2 * self.sconv_kernel + 1)

    # -------------------------------------------------------------------- load
    @classmethod
    def from_dict(cls, cfg: dict) -> "InklingConfig":
        t = cfg.get("text_config", cfg)
        loop = t.get("loop_config") or {}
        if t.get("swa_num_attention_heads", t["num_attention_heads"]) != t["num_attention_heads"]:
            raise ValueError("engine assumes identical SWA and global head counts")
        if t.get("n_shared_experts", 0):
            raise ValueError("engine assumes no shared experts")
        return cls(
            hidden_size=t["hidden_size"],
            num_layers=t["num_hidden_layers"],
            vocab_size=t["vocab_size"],
            unpadded_vocab_size=t.get("unpadded_vocab_size", t["vocab_size"]),
            num_heads=t["num_attention_heads"],
            num_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            d_rel=t["d_rel"],
            rel_extent=t["rel_extent"],
            sliding_window=t["sliding_window_size"],
            local_layer_ids=tuple(t["local_layer_ids"]),
            sconv_kernel=t["sconv_kernel_size"],
            rms_eps=t["rms_norm_eps"],
            qk_l2_clip=t["qk_l2_clip_threshold"],
            rel_l2_clip=t["rel_l2_clip_threshold"],
            log_scaling_n_floor=float(t["log_scaling_n_floor"]),
            log_scaling_alpha=t["log_scaling_alpha"],
            final_logit_softcap=t["final_logit_softcapping"],
            attn_res_block=t["n_layers_per_attn_res_block"],
            n_experts=t["n_routed_experts"],
            top_k=t["num_experts_per_tok"],
            moe_inter=t["intermediate_size"],
            mlp_gate_softcap=t["mlp_gate_softcap"],
            mlp_up_softcap=t["mlp_up_softcap"],
            route_scale=t["route_scale"],
            core_start=loop.get("core_start", t["num_hidden_layers"]),
            core_end=loop.get("core_end", t["num_hidden_layers"]),
            num_passes=loop.get("num_passes", 1),
            gate_rank=loop.get("gate_rank", 0),
            eos_token_id=cfg.get("eos_token_id", 200006),
            max_position=t.get("model_max_length", 1 << 20),
            mtp=dict(cfg.get("mtp_config") or {}),
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "InklingConfig":
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text()))
