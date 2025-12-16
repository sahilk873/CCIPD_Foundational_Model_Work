#!/usr/bin/env python3
# lora_musk_loader.py

import os
from importlib import import_module
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _pick_precision(device: str) -> torch.dtype:
    return torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32


def _maybe_hf_login(token: Optional[str]) -> None:
    """
    Only attempts HF login if a token is provided.
    Useful on clusters where you may need auth to pull checkpoints.
    """
    if not token:
        return
    try:
        from huggingface_hub import whoami
        _ = whoami()
    except Exception:
        from huggingface_hub import login
        login(token=token)


class _MuskVisionWrapper(nn.Module):
    """
    Make MUSK behave like a plain vision encoder: model(x) -> tensor embedding.

    Keeps .core so attention_from_h5.py can still reach internals if needed.
    Provides .expected_img_size for resizing enforcement upstream.
    """
    def __init__(self, core: nn.Module, out_dtype: torch.dtype, expected_img_size: int):
        super().__init__()
        self.core = core
        self.out_dtype = out_dtype
        self.expected_img_size = expected_img_size

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.out_dtype)

        # MUSK commonly uses a keyword-based API: model(image=..., ...)
        try:
            out = self.core(
                image=x,
                with_head=False,
                out_norm=False,
                ms_aug=True,
                return_global=True,
            )
        except TypeError:
            # If this particular MUSK build supports plain forward(x), try that.
            out = self.core(x)

        # Normalize return formats into a single tensor embedding
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            for v in out:
                if torch.is_tensor(v):
                    return v
        if isinstance(out, dict):
            for k in ("vision_cls", "features", "global", "pooled_output", "last_hidden_state"):
                v = out.get(k, None)
                if torch.is_tensor(v):
                    return v[:, 0] if (k == "last_hidden_state" and v.dim() == 3) else v

        raise RuntimeError(f"Unexpected MUSK forward output type: {type(out)}")


def _find_trident_builder() -> Optional[callable]:
    """
    Tries a few likely TRIDENT module paths that may expose create_model_from_pretrained().
    """
    candidates = (
        "trident.patch_encoder_models.model_zoo.musk.musk",
        "trident.patch_encoder_models.model_zoo.musk",
        "trident.patch_encoder_models.model_zoo.beit3.beit3",
        "trident.patch_encoder_models.model_zoo.beit3",
    )
    for modpath in candidates:
        try:
            mod = import_module(modpath)
            if hasattr(mod, "create_model_from_pretrained"):
                return getattr(mod, "create_model_from_pretrained")
        except Exception:
            pass
    return None


def load_musk(
    device: str = "cuda",
    prefer_trident: bool = True,
    hf_token: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    img_size: int = 384,
) -> Tuple[nn.Module, callable, torch.dtype]:
    """
    Returns: (model, tfm, precision)

    Strategy:
      1) If TRIDENT MUSK/BEiT-3 builder exists and prefer_trident=True, use it.
      2) Else fallback to timm + musk package loader.

    Notes:
      - The timm fallback requires the MUSK repo python package installed (importable as `musk`).
      - Crucially: importing `musk.modeling` registers "musk_large_patch16_384" into timm.
    """
    prec = _pick_precision(device)
    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN", None)

    # --------------------------
    # TRIDENT path (preferred)
    # --------------------------
    builder = _find_trident_builder()
    if prefer_trident and builder is not None:
        if checkpoint_path is None:
            checkpoint_path = os.environ.get("MUSK_CKPT", "hf_hub:mahmoodlab/MUSK")

        kwargs = {"checkpoint_path": checkpoint_path}
        # Some TRIDENT builders accept img_size
        if img_size is not None:
            kwargs["img_size"] = int(img_size)
        if token:
            kwargs["hf_auth_token"] = token

        model, tfm = builder(**kwargs)
        model = model.to(device=device, dtype=prec).eval()
        # for attention_from_h5.py compatibility
        setattr(model, "expected_img_size", int(img_size))
        return model, tfm, prec

    # --------------------------
    # timm + MUSK fallback
    # --------------------------
    _maybe_hf_login(token)

    try:
        # IMPORTANT: modeling import registers MUSK models into timm.
        from musk import utils as musk_utils, modeling  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "TRIDENT MUSK builder not found, and timm fallback requires the MUSK python package.\n"
            "Install the MUSK repo package so `import musk` and `import musk.modeling` work.\n"
            f"Original error: {e}"
        )

    from timm.models import create_model
    from torchvision import transforms as T
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

    # This now works after importing musk.modeling (registers entrypoints)
    core = create_model("musk_large_patch16_384")

    # Load pretrained weights
    ckpt = checkpoint_path or os.environ.get("MUSK_CKPT_FALLBACK", "hf_hub:xiangjx/musk")
    musk_utils.load_model_and_may_interpolate(ckpt, core, "model|module", "")

    core = core.to(device=device, dtype=prec).eval()
    model = _MuskVisionWrapper(core, prec, expected_img_size=int(img_size))

    # Default preprocessing used in MUSK examples (resize/crop + inception norm).
    tfm = T.Compose([
        T.Resize(int(img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop((int(img_size), int(img_size))),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    return model, tfm, prec
