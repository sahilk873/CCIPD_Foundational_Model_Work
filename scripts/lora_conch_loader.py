#!/usr/bin/env python3
# lora_conch_loader.py

import os
from importlib import import_module
from typing import Optional, Tuple

import torch


def _pick_precision(device: str) -> torch.dtype:
    return torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32


def _find_trident_builder() -> Optional[callable]:
    """
    Try to find TRIDENT's Conch v1.5 builder.
    """
    candidates = (
        "trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5",
        "trident.patch_encoder_models.model_zoo.conchv1_5",
        "trident.patch_encoder_models.model_zoo.conch.conch",
        "trident.patch_encoder_models.model_zoo.conch",
    )
    for modpath in candidates:
        try:
            mod = import_module(modpath)
            if hasattr(mod, "create_model_from_pretrained"):
                return getattr(mod, "create_model_from_pretrained")
        except Exception:
            pass
    return None


def load_conch_v15(
    device: str = "cuda",
    prefer_hf: bool = True,
    hf_token: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    img_size: Optional[int] = None,
) -> Tuple[torch.nn.Module, callable, torch.dtype]:
    """
    Returns: (model, tfm, precision)

    Notes:
      - This expects TRIDENT to be installed (Conch builder lives in TRIDENT).
      - Conch often supports variable image sizes, so img_size is optional.
      - If you want HF, pass checkpoint_path like: "hf_hub:<org>/<repo>".
    """
    prec = _pick_precision(device)

    builder = _find_trident_builder()
    if builder is None:
        raise RuntimeError(
            "Could not find TRIDENT Conch builder (create_model_from_pretrained). "
            "Make sure TRIDENT is installed and includes conchv1_5."
        )

    # Prefer HF hub checkpoint unless user provides something else
    if checkpoint_path is None:
        # You can override this by passing checkpoint_path explicitly.
        # Keep it flexible because different TRIDENT installs name this differently.
        checkpoint_path = os.environ.get("CONCH_CKPT", "hf_hub:mahmoodlab/CONCHv1_5")

    kwargs = {"checkpoint_path": checkpoint_path}
    if img_size is not None:
        kwargs["img_size"] = int(img_size)
    if hf_token:
        kwargs["hf_auth_token"] = hf_token

    core, tfm = builder(**kwargs)
    core = core.to(device=device, dtype=prec).eval()

    # Expose expected size if known
    if img_size is not None:
        setattr(core, "expected_img_size", int(img_size))

    return core, tfm, prec
