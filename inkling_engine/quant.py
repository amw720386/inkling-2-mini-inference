"""Expert weight formats of the Inkling-2-Mini checkpoint and torch dequant references.

NVFP4 (layers 0-19): weight u8 [E, N, K/2], two E2M1 codes per byte (even element in
the low nibble), float8_e4m3 scale per 16 values [E, N, K/16], and a per-expert factor
amax[E] / outer_scale_denominator.

MXFP8 (layers 20-23): weight float8_e4m3 [E, N, K], u8 exponent scale per 32 values
[E, N, K/32] meaning 2 ** (s - 127).

w13 rows interleave gate and up: row 2*i is gate_i, row 2*i+1 is up_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@dataclass
class ExpertWeight:
    """One quantized expert matrix stack: logical shape [E, N, K]."""

    fmt: str  # "nvfp4" | "mxfp8" | "bf16"
    weight: torch.Tensor
    scale: torch.Tensor | None = None
    expert_factor: torch.Tensor | None = None  # fp32 [E], nvfp4 only

    @property
    def shape(self) -> tuple[int, int, int]:
        e, n, k = self.weight.shape
        return (e, n, k * 2 if self.fmt == "nvfp4" else k)

    def to(self, device) -> "ExpertWeight":
        mv = lambda t: None if t is None else t.to(device, non_blocking=True)
        return ExpertWeight(self.fmt, mv(self.weight), mv(self.scale), mv(self.expert_factor))

    def dequant(self, experts: torch.Tensor | slice | None = None, dtype=torch.bfloat16) -> torch.Tensor:
        sel = (lambda t: t) if experts is None else (lambda t: t[experts])
        if self.fmt == "bf16":
            return sel(self.weight).to(dtype)
        if self.fmt == "nvfp4":
            return dequant_nvfp4(sel(self.weight), sel(self.scale), sel(self.expert_factor), dtype)
        return dequant_mxfp8(sel(self.weight), sel(self.scale), dtype)


def _byte_lut(device) -> torch.Tensor:
    vals = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    b = torch.arange(256, device=device)
    return torch.stack((vals[b & 0xF], vals[b >> 4]), dim=-1)  # [256, 2]


def dequant_nvfp4(w: torch.Tensor, scale: torch.Tensor, factor: torch.Tensor, dtype=torch.bfloat16):
    lut = _byte_lut(w.device)
    *lead, n, kb = w.shape
    vals = lut[w.long()].reshape(*lead, n, kb * 2 // 16, 16)
    vals = vals * scale.float()[..., None] * factor.float().reshape(*factor.shape, *([1] * 3))
    return vals.reshape(*lead, n, kb * 2).to(dtype)


def dequant_mxfp8(w: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16):
    *lead, n, k = w.shape
    vals = w.float().reshape(*lead, n, k // 32, 32) * torch.exp2(scale.float() - 127.0)[..., None]
    return vals.reshape(*lead, n, k).to(dtype)


def expert_weight_from_checkpoint(tensors: dict[str, torch.Tensor], prefix: str) -> ExpertWeight:
    """prefix like 'model.llm.layers.3.mlp.experts.w13'."""
    w = tensors[f"{prefix}_weight"]
    if f"{prefix}_weight_amax" in tensors:
        amax = tensors[f"{prefix}_weight_amax"].float()
        denom = tensors[f"{prefix}_weight_outer_scale_denominator"].float()
        return ExpertWeight("nvfp4", w, tensors[f"{prefix}_weight_scale"], (amax / denom).contiguous())
    if w.dtype == torch.float8_e4m3fn:
        return ExpertWeight("mxfp8", w, tensors[f"{prefix}_weight_scale"])
    return ExpertWeight("bf16", w)
