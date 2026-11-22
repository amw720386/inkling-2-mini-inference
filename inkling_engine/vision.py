"""Image input: transformers' Inkling 2 vision tower + processor, spliced into the engine's embeddings.

Image encoding runs once per image at prefill, so the reference module is used as-is. Its outputs replace
the (post-embed-norm) embeddings at the image placeholder tokens, exactly as Inkling2LoopModel does.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

IMAGE_TOKEN_ID = 200054

# checkpoint name -> transformers vision tower name (transformers conversion_mapping, "inkling2")
_RENAMES = [
    (r"^model\.visual\.backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^model\.visual\.backbone\.embeddings\.", "embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "layers."),
    (r"^model\.visual\.proj_in\.", "merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "merger.linear_2."),
    (r"(layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(layers\.\d+)\.ls_1\.gamma$", r"\1.lambda_1"),
    (r"(layers\.\d+)\.ls_2\.gamma$", r"\1.lambda_2"),
]


def _rename(name: str) -> str:
    for pat, rep in _RENAMES:
        name = re.sub(pat, rep, name)
    return name


class VisionEncoder:
    def __init__(self, model_dir: str | Path, device="cuda"):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModel, AutoProcessor

        cfg = AutoConfig.from_pretrained(model_dir)
        self.tower = AutoModel.from_config(cfg.vision_config, torch_dtype=torch.bfloat16).to(device).eval()
        state = {}
        for f in sorted(Path(model_dir).glob("model*.safetensors")):
            with safe_open(str(f), framework="pt", device=device) as fh:
                for k in fh.keys():
                    if k.startswith("model.visual."):
                        state[_rename(k)] = fh.get_tensor(k)
        missing, unexpected = self.tower.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"vision weights mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.image_token_id = getattr(cfg, "image_token_id", IMAGE_TOKEN_ID)

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """[n_image_tokens, hidden] bf16 features in placeholder order."""
        out = self.tower(pixel_values=pixel_values.to(self.tower.device, torch.bfloat16),
                         image_grid_thw=image_grid_thw.cpu(), return_dict=True)
        return out.pooler_output.to(torch.bfloat16)

    def chat_inputs(self, messages, tools=None, **template_kwargs):
        """Tokenized chat with expanded image placeholders: (input_ids list, image features or None)."""
        enc = self.processor.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors="pt", **template_kwargs)
        ids = enc["input_ids"][0].tolist()
        feats = None
        if enc.get("pixel_values") is not None:
            feats = self.encode(enc["pixel_values"], enc["image_grid_thw"])
        return ids, feats
