from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from diffusers.pipelines import FluxPipeline
from safetensors import safe_open
from safetensors.torch import save_file


LAYOUT_PREFIX = "layout::"
LORA_PREFIX = "lora::"
BUNDLE_FORMAT = "occlusionformer_bundle"


def _unwrap_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _to_tensor_dict(state_dict) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
    return out


def load_checkpoint_bundle(
    ckpt_path: str | Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float], Dict[str, str]]:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "bundle_weights.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing bundle checkpoint: {ckpt_path}")

    layout_state_dict: Dict[str, torch.Tensor] = {}
    lora_state_dict: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, str] = {}

    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.startswith(LAYOUT_PREFIX):
                layout_state_dict[key[len(LAYOUT_PREFIX):]] = tensor
            elif key.startswith(LORA_PREFIX):
                lora_state_dict[key[len(LORA_PREFIX):]] = tensor

    network_alphas_raw = metadata.get("network_alphas_json", "{}")
    try:
        network_alphas = json.loads(network_alphas_raw)
    except json.JSONDecodeError:
        network_alphas = {}

    return layout_state_dict, lora_state_dict, network_alphas, metadata
