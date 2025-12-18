#!/usr/bin/env python3
# lora_hoptimus1_loader.py

import os
from importlib import import_module
from typing import Optional, Tuple, Callable, Any

import torch
import torch.nn as nn


def _pick_precision(device: str) -> torch.dtype:
    return torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32


def _maybe_hf_login(token: Optional[str]) -> None:
    if not token:
        return
    try:
        from huggingface_hub import whoami
        _ = whoami()
    except Exception:
        from huggingface_hub import login
        login(token=token)


class _VisionWrapper(nn.Module):
    """
    Wrapper so model(x) works consistently, but we still preserve .core for attention access.
    """
    def __init__(self, core: nn.Module, out_dtype: torch.dtype, expected_img_size: int):
        super().__init__()
        self.core = core
        self.out_dtype = out_dtype
        self.expected_img_size = expected_img_size  # used by attention_from_h5.py

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.out_dtype)
        out = self.core(x)

        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            for v in out:
                if torch.is_tensor(v):
                    return v
        if isinstance(out, dict):
            for k in ("features", "vision_cls", "global", "pooled_output", "last_hidden_state"):
                v = out.get(k, None)
                if torch.is_tensor(v):
                    # if transformer hidden states, return CLS
                    return v[:, 0] if (k == "last_hidden_state" and v.dim() == 3) else v

        raise RuntimeError(f"Unexpected forward output type: {type(out)}")


def _find_trident_builder() -> Optional[Callable[..., Any]]:
    """
    TRIDENT includes builders that return (core, tfm) for H-optimus.
    We keep this as first choice because it avoids timm-model-name issues.
    """
    candidates = (
        "trident.patch_encoder_models.model_zoo.hoptimus1.hoptimus1",
        "trident.patch_encoder_models.model_zoo.hoptimus1",
        "trident.patch_encoder_models.model_zoo.hoptimus.hoptimus",
    )
    for modpath in candidates:
        try:
            mod = import_module(modpath)
            if hasattr(mod, "create_model_from_pretrained"):
                return getattr(mod, "create_model_from_pretrained")
        except Exception:
            pass
    return None


def load_hoptimus1(
    device: str = "cuda",
    prefer_hf: bool = True,
    hf_token: Optional[str] = None,
    img_size: int = 224,
    checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, callable, torch.dtype]:
    """
    Returns: (model_wrapper, tfm, precision)

    Notes:
      - H-optimus-1 expects 224x224 by default.
      - TRIDENT builder is preferred (more stable on pinned timm installs).
    """
    prec = _pick_precision(device)
    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN", None)

    # ---- TRIDENT builder path (preferred) ----
    builder = _find_trident_builder()

    if checkpoint_path is None:
        checkpoint_path = "hf_hub:bioptimus/H-optimus-1"

    if prefer_hf and builder is not None:
        kwargs = {"checkpoint_path": checkpoint_path, "img_size": int(img_size)}
        if token:
            kwargs["hf_auth_token"] = token
        core, tfm = builder(**kwargs)
        core = core.to(device=device, dtype=prec).eval()
        model = _VisionWrapper(core, prec, expected_img_size=int(img_size))
        return model, tfm, prec

    # ---- timm fallback path ----
    _maybe_hf_login(token)

    from torchvision import transforms as T
    from timm import create_model

    # Try the most likely hub id first; other ids can be added if needed.
    model_id = "hf-hub:bioptimus/H-optimus-1"
    core = create_model(model_id, pretrained=True, num_classes=0)

    core = core.to(device=device, dtype=prec).eval()
    model = _VisionWrapper(core, prec, expected_img_size=int(img_size))

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
        ),
    ])

    return model, tfm, prec
