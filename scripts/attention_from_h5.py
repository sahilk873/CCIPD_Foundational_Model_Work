#!/usr/bin/env python3
# attention_from_h5.py

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


# -----------------------------
# Loaders
# -----------------------------
from lora_conch_loader import load_conch_v15
from lora_hoptimus1_loader import load_hoptimus1
from lora_musk_loader import load_musk


# -----------------------------
# Utilities
# -----------------------------
def load_tiles_from_h5(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        for key in ("images", "imgs"):
            if key in f:
                return f[key][:]
    raise KeyError("H5 must contain dataset key 'images' or 'imgs'.")


def tile_to_pil(tile: np.ndarray) -> Image.Image:
    if tile.ndim == 3 and tile.shape[-1] in (1, 3, 4):  # (H,W,C)
        arr = tile[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return Image.fromarray(arr.astype(np.uint8))
    if tile.ndim == 3 and tile.shape[0] in (1, 3, 4):   # (C,H,W)
        arr = tile[:3]
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr.astype(np.uint8))
    if tile.ndim == 2:
        return Image.fromarray(tile.astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported tile shape: {tile.shape}")


def resize_pil_if_needed(img: Image.Image, img_size: Optional[int]) -> Image.Image:
    if img_size is None:
        return img
    if img.size == (img_size, img_size):
        return img
    return img.resize((img_size, img_size), resample=Image.BILINEAR)


def _collect_attn_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    timm ViT attention: has qkv/num_heads/proj
    TorchScale-style attention: has q_proj/k_proj/v_proj
    """
    attn_mods = []
    for m in model.modules():
        if all(hasattr(m, a) for a in ("qkv", "num_heads", "proj")):
            attn_mods.append(m)
            continue
        if all(hasattr(m, a) for a in ("q_proj", "k_proj", "v_proj")):
            attn_mods.append(m)
    return attn_mods


def _infer_patch_layout_from_tokens(N: int) -> Tuple[int, int]:
    """
    Robust token layout inference.

    Many models have:
      N = 1 + R + P, where P = s*s patches, R = register/distill tokens (often 4)
    We find largest square P <= (N-1) and set special = N - P.

    If N itself is a square, we allow no-CLS case: special=0, P=N.

    Returns: (grid_side s, special_tokens_count)
    """
    # no CLS case
    s0 = int(np.floor(np.sqrt(N)))
    if s0 * s0 == N:
        return s0, 0

    # CLS plus possibly extra tokens
    maxP = N - 1
    s = int(np.floor(np.sqrt(maxP)))
    P = s * s
    special = N - P  # includes CLS + any registers
    if P <= 0:
        raise ValueError(f"Cannot infer patch grid from token count N={N}")
    return s, special


def _normalize_attn_to_BHNN(attn: torch.Tensor) -> torch.Tensor:
    """
    Normalize common attention formats to (B, H, N, N).
    """
    if attn.ndim == 4:
        return attn
    if attn.ndim == 3:
        return attn.unsqueeze(1)
    if attn.ndim == 2:
        return attn.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")


def get_attention_map(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: str,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    musk_forward: bool = False,
) -> np.ndarray:
    """
    Monkey-patch one attention module and capture its softmax weights.
    Works for:
      - timm ViT Attention (qkv)
      - TorchScale-style MHA (q_proj/k_proj/v_proj)
    """

    search_root = model.core if hasattr(model, "core") else model
    attn_modules = _collect_attn_modules(search_root)
    if not attn_modules:
        raise RuntimeError("No attention modules found to hook (timm or TorchScale-style).")

    target = attn_modules[layer_idx]
    attn_store = {}

    is_timm_attn = all(hasattr(target, a) for a in ("qkv", "num_heads", "proj"))

    if is_timm_attn:
        def patched_forward_timm(self, x_in):
            B, N, C = x_in.shape
            qkv = (
                self.qkv(x_in)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * getattr(self, "scale", 1.0)
            attn = attn.softmax(dim=-1)
            attn_store["attn"] = attn.detach().cpu()
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            if hasattr(self, "proj_drop"):
                out = self.proj_drop(out)
            return out

        orig_forward = target.forward
        target.forward = patched_forward_timm.__get__(target, type(target))

        with torch.no_grad():
            if musk_forward and hasattr(model, "core"):
                _ = model.core(
                    image=x.to(device, dtype=next(model.parameters()).dtype),
                    with_head=False,
                    out_norm=False,
                    ms_aug=False,
                    return_global=False,
                )
            else:
                _ = model(x.to(device, dtype=next(model.parameters()).dtype))

        target.forward = orig_forward

    else:
        # TorchScale / fairseq-style: force returning weights if supported
        import inspect
        sig = inspect.signature(target.forward)
        orig_forward = target.forward

        def patched_forward_ts(*args, **kwargs):
            if "need_weights" in sig.parameters:
                kwargs.setdefault("need_weights", True)
            if "need_head_weights" in sig.parameters:
                kwargs.setdefault("need_head_weights", True)
            if "average_attn_weights" in sig.parameters:
                kwargs.setdefault("average_attn_weights", False)

            out = orig_forward(*args, **kwargs)

            if isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[1]):
                attn_store["attn"] = out[1].detach().cpu()
            elif isinstance(out, dict):
                for k in ("attn_weights", "attn", "attn_probs"):
                    v = out.get(k, None)
                    if torch.is_tensor(v):
                        attn_store["attn"] = v.detach().cpu()
                        break
            return out

        target.forward = patched_forward_ts

        with torch.no_grad():
            if musk_forward and hasattr(model, "core"):
                _ = model.core(
                    image=x.to(device, dtype=next(model.parameters()).dtype),
                    with_head=False,
                    out_norm=False,
                    ms_aug=False,
                    return_global=False,
                )
            else:
                _ = model(x.to(device, dtype=next(model.parameters()).dtype))

        target.forward = orig_forward

    if "attn" not in attn_store:
        raise RuntimeError("Failed to capture attention weights.")

    attn = _normalize_attn_to_BHNN(attn_store["attn"])  # (B,H,N,N)
    attn = attn[0]  # (H,N,N)

    if head_idx is not None:
        attn_2d = attn[head_idx]         # (N,N)
    else:
        attn_2d = attn.mean(0)           # (N,N)

    N = attn_2d.shape[-1]
    grid_side, special = _infer_patch_layout_from_tokens(N)
    P = grid_side * grid_side

    if special == 0:
        # No CLS token: use mean attention each token gives to others, then reshape
        patch_vec = attn_2d.mean(0)[:P]
    else:
        # CLS is assumed at index 0; patch tokens assumed at the end
        patch_start = special
        patch_vec = attn_2d[0, patch_start:patch_start + P]

    grid = patch_vec.reshape(grid_side, grid_side)

    attn_map = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0),
        size=(x.shape[2], x.shape[3]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    return attn_map


def plot_attention_overlay(tile_tensor: torch.Tensor, attn_map: np.ndarray, save_path: Path):
    img = tile_tensor[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)
    vmin, vmax = float(img.min()), float(img.max())
    img = (img - vmin) / (vmax - vmin + 1e-8)

    attn = np.asarray(attn_map, dtype=np.float32)
    amin, amax = float(attn.min()), float(attn.max())
    attn = (attn - amin) / (amax - amin + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, interpolation="nearest")
    plt.imshow(attn, cmap="jet", alpha=0.5, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=["conch", "hoptimus", "musk"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-tiles", type=int, default=100)
    ap.add_argument("--layer-idx", type=int, default=-1)
    ap.add_argument("--head-idx", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None, help="Force a square resize before tfm (e.g., 224).")
    ap.add_argument("--hf-token-env", type=str, default="HF_TOKEN", help="Env var name for HF token.")
    return ap.parse_args()


def main():
    args = parse_args()
    h5_path = Path(args.h5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    hf_token = os.environ.get(args.hf_token_env, None)

    # Load model + tfm
    musk_forward = False

    if args.model == "conch":
        model, tfm, _ = load_conch_v15(device=device, prefer_hf=True, hf_token=hf_token)
        expected = getattr(model, "expected_img_size", None)
    elif args.model == "hoptimus":
        model, tfm, _ = load_hoptimus1(device=device, prefer_hf=True, hf_token=hf_token, img_size=224)
        expected = getattr(model, "expected_img_size", 224)
    else:
        model, tfm, _ = load_musk(device=device, hf_token=hf_token)
        expected = getattr(model, "expected_img_size", None)
        musk_forward = True  # safe default for MUSK wrappers

    model.eval()
    param_dtype = next(model.parameters()).dtype

    # Choose resize size: CLI overrides loader
    resize_size = args.img_size if args.img_size is not None else expected

    tiles = load_tiles_from_h5(h5_path)
    N = min(args.max_tiles, len(tiles))

    for idx in range(N):
        img = tile_to_pil(tiles[idx])
        img = resize_pil_if_needed(img, resize_size)

        x = tfm(img).unsqueeze(0).to(device=device, dtype=param_dtype)

        attn_map = get_attention_map(
            model=model,
            x=x,
            device=device,
            layer_idx=args.layer_idx,
            head_idx=args.head_idx,
            musk_forward=musk_forward,
        )

        head_tag = f"H{args.head_idx}" if args.head_idx is not None else "Havg"
        save_path = out_dir / f"tile{idx}_L{args.layer_idx}_{head_tag}.png"
        plot_attention_overlay(x, attn_map, save_path)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{N}")

    print(f"✅ Done. Saved {N} attention overlays to: {out_dir}")


if __name__ == "__main__":
    main()
