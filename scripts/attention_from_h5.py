#!/usr/bin/env python3
# attention_from_h5_folder.py
#
# What this script does
# 1) Accepts either a single H5 file OR a folder (or glob) of H5/HDF5 files
# 2) For each H5, loads tiles from dataset key "images" or "imgs"
# 3) Runs a chosen foundation model (conch, hoptimus, musk)
# 4) Hooks a chosen attention layer and captures attention weights
# 5) Produces attention overlay PNGs for up to --max-tiles tiles per H5
#
# Key fixes included
# - Works on folders (not just one H5)
# - timm attention hook now accepts attn_mask and other kwargs
# - More defensive attention module discovery across timm and TorchScale-style MHA
# - Robust patch-grid inference even with register/distill tokens

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# -----------------------------
# Loaders (your local loaders)
# -----------------------------
from lora_conch_loader import load_conch_v15
from lora_hoptimus1_loader import load_hoptimus1
from lora_musk_loader import load_musk


# -----------------------------
# H5 -> tiles
# -----------------------------
def load_tiles_from_h5(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        for key in ("images", "imgs"):
            if key in f:
                return f[key][:]
    raise KeyError(f"{h5_path.name}: H5 must contain dataset key 'images' or 'imgs'.")


# -----------------------------
# Image helpers
# -----------------------------
def tile_to_pil(tile: np.ndarray) -> Image.Image:
    """
    Supports tiles stored as:
      - (H, W, C) with C in {1,3,4}
      - (C, H, W) with C in {1,3,4}
      - (H, W) grayscale
    """
    if tile.ndim == 3 and tile.shape[-1] in (1, 3, 4):  # (H,W,C)
        arr = tile[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return Image.fromarray(arr.astype(np.uint8))
    if tile.ndim == 3 and tile.shape[0] in (1, 3, 4):  # (C,H,W)
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


# -----------------------------
# Attention module discovery
# -----------------------------
def _collect_attn_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Collect possible attention modules across common implementations.

    timm ViT Attention modules usually have:
      - qkv, num_heads, proj
    TorchScale / fairseq-style MHA often have:
      - q_proj, k_proj, v_proj
    """
    attn_mods: List[torch.nn.Module] = []
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

    Many ViTs:
      N = special + P, where P = s*s patches, special includes CLS and possible register/distill tokens.

    Strategy:
      - If N is a perfect square, allow special=0, P=N.
      - Else assume at least CLS, search for largest square P <= (N-1), then special = N - P.
    """
    s0 = int(np.floor(np.sqrt(N)))
    if s0 * s0 == N:
        return s0, 0

    maxP = N - 1
    s = int(np.floor(np.sqrt(maxP)))
    P = s * s
    special = N - P
    if P <= 0:
        raise ValueError(f"Cannot infer patch grid from token count N={N}")
    return s, special


def _normalize_attn_to_BHNN(attn: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention tensors to shape (B, H, N, N)
    """
    if attn.ndim == 4:
        return attn
    if attn.ndim == 3:
        return attn.unsqueeze(1)
    if attn.ndim == 2:
        return attn.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")


# -----------------------------
# Attention capture
# -----------------------------
def get_attention_map(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: str,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    musk_forward: bool = False,
) -> np.ndarray:
    """
    Hook one attention module and capture softmax attention weights.

    Supports:
      - timm ViT Attention (qkv/num_heads/proj)
      - TorchScale-style MHA (q_proj/k_proj/v_proj) when it can return weights

    Returns:
      attn_map as numpy array sized (H, W) matching x spatial size
    """

    search_root = model.core if hasattr(model, "core") else model
    attn_modules = _collect_attn_modules(search_root)
    if not attn_modules:
        raise RuntimeError("No attention modules found to hook (timm or TorchScale-style).")

    # normalize negative indices
    if layer_idx < 0:
        layer_idx = len(attn_modules) + layer_idx
    if layer_idx < 0 or layer_idx >= len(attn_modules):
        raise IndexError(f"layer_idx out of range. Got {layer_idx}, num_layers={len(attn_modules)}")

    target = attn_modules[layer_idx]
    attn_store = {}

    is_timm_attn = all(hasattr(target, a) for a in ("qkv", "num_heads", "proj"))

    if is_timm_attn:
        # timm newer versions can pass attn_mask=... into attention forward
        def patched_forward_timm(self, x_in, **kwargs):
            B, N, C = x_in.shape
            qkv = (
                self.qkv(x_in)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * getattr(self, "scale", 1.0)

            # Optional support for masks (best-effort)
            attn_mask = kwargs.get("attn_mask", None)
            if attn_mask is not None and torch.is_tensor(attn_mask):
                # Common shapes: (N,N), (B,N,N), (B,1,N,N)
                if attn_mask.ndim == 2:
                    attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.ndim == 3:
                    attn = attn + attn_mask.unsqueeze(1)
                elif attn_mask.ndim == 4:
                    attn = attn + attn_mask

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
        # TorchScale / fairseq-style MHA: attempt to force returning weights if signature supports it
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
        attn_2d = attn[head_idx]  # (N,N)
    else:
        attn_2d = attn.mean(0)  # (N,N)

    Ntok = attn_2d.shape[-1]
    grid_side, special = _infer_patch_layout_from_tokens(Ntok)
    P = grid_side * grid_side

    if special == 0:
        # No CLS token case
        patch_vec = attn_2d.mean(0)[:P]
    else:
        # CLS assumed at index 0, patch tokens assumed after special tokens
        patch_start = special
        patch_vec = attn_2d[0, patch_start : patch_start + P]

    grid = patch_vec.reshape(grid_side, grid_side)

    attn_map = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0),
        size=(x.shape[2], x.shape[3]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()

    return attn_map


def plot_attention_overlay(
    tile_tensor: torch.Tensor,
    attn_map: np.ndarray,
    save_path: Path,
    attn_thresh: float = 0.15,
    attn_alpha: float = 0.5,
    attn_display_gamma: Optional[float] = 0.4,
):
    img = tile_tensor[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)
    vmin, vmax = float(img.min()), float(img.max())
    img = (img - vmin) / (vmax - vmin + 1e-8)

    attn = np.asarray(attn_map, dtype=np.float32)
    amin, amax = float(attn.min()), float(attn.max())
    attn = (attn - amin) / (amax - amin + 1e-8)  # now in [0,1]

    # Log-style display transform so colormap spreads smoothly (gamma < 1 stretches low values)
    if attn_display_gamma is not None:
        attn_display = np.power(attn, attn_display_gamma)
    else:
        attn_display = attn

    # Threshold in linear space; colormap/alpha use display values
    attn_thresh = float(np.clip(attn_thresh, 0.0, 1.0))
    mask = attn >= attn_thresh
    alpha = np.zeros_like(attn, dtype=np.float32)
    alpha[mask] = (attn_display[mask] * np.clip(attn_alpha, 0.0, 1.0)).astype(np.float32)

    attn_vis = attn_display.copy()
    attn_vis[~mask] = np.nan

    plt.figure(figsize=(5, 5))
    plt.imshow(img, interpolation="nearest")
    plt.imshow(attn_vis, cmap="jet", alpha=alpha, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------
# Input listing (single, folder, glob)
# -----------------------------
def list_h5_files(h5: Optional[str], h5_dir: Optional[str], glob_pat: Optional[str]) -> List[Path]:
    items: List[Path] = []

    if h5 is not None:
        items.append(Path(h5))

    if h5_dir is not None:
        d = Path(h5_dir)
        items.extend(sorted(d.glob("*.h5")))
        items.extend(sorted(d.glob("*.hdf5")))

    if glob_pat is not None:
        p = Path(glob_pat)
        if any(ch in glob_pat for ch in ["*", "?", "["]):
            parent = p.parent if str(p.parent) != "" else Path(".")
            items.extend(sorted(parent.glob(p.name)))
        else:
            if p.is_dir():
                items.extend(sorted(p.glob("*.h5")))
                items.extend(sorted(p.glob("*.hdf5")))
            else:
                items.append(p)

    uniq: List[Path] = []
    seen = set()
    for x in items:
        x = x.resolve()
        if x in seen:
            continue
        seen.add(x)
        if x.is_file():
            uniq.append(x)
    return uniq


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--h5", type=str, help="Single H5 file (backward compatible).")
    g.add_argument("--h5-dir", type=str, help="Directory containing H5/HDF5 files.")
    g.add_argument("--glob", type=str, help="Glob pattern like '/path/*.h5' or a directory path.")

    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=["conch", "hoptimus", "musk"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-tiles", type=int, default=100)
    ap.add_argument("--layer-idx", type=int, default=-1)
    ap.add_argument("--head-idx", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None, help="Force square resize before tfm (e.g., 224).")
    ap.add_argument("--hf-token-env", type=str, default="HF_TOKEN", help="Env var name containing HF token.")
    ap.add_argument("--flat-out", action="store_true", help="Dump PNGs directly into out-dir (no per-H5 subfolders).")
    ap.add_argument("--attn-thresh", type=float, default=0.15,
                help="Attention threshold in [0,1] after normalization. Below becomes transparent.")
    ap.add_argument("--attn-alpha", type=float, default=0.5,
                    help="Max overlay opacity for attention >= threshold.")
    ap.add_argument("--attn-display-gamma", type=float, default=0.4,
                    help="Power (gamma) for display transform; <1 spreads low attention (default 0.4). Use 1.0 for linear.")

    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    hf_token = os.environ.get(args.hf_token_env, None)

    # Load model and transform once
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
        musk_forward = True

    model.eval()
    param_dtype = next(model.parameters()).dtype

    resize_size = args.img_size if args.img_size is not None else expected

    h5_files = list_h5_files(args.h5, args.h5_dir, args.glob)
    if not h5_files:
        raise FileNotFoundError("No H5/HDF5 files found for the given input.")

    head_tag = f"H{args.head_idx}" if args.head_idx is not None else "Havg"

    total_tiles = 0
    for fi, h5_path in enumerate(h5_files, start=1):
        print(f"\n[{fi}/{len(h5_files)}] Processing: {h5_path}")

        try:
            tiles = load_tiles_from_h5(h5_path)
        except Exception as e:
            print(f"  [WARN] Skipping {h5_path.name}: {e}")
            continue

        n_tiles = min(args.max_tiles, len(tiles))

        # output folder per file
        if args.flat_out:
            cur_out = out_dir
        else:
            cur_out = out_dir / h5_path.stem
            cur_out.mkdir(parents=True, exist_ok=True)

        for idx in range(n_tiles):
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

            prefix = "" if not args.flat_out else f"{h5_path.stem}_"
            save_path = cur_out / f"{prefix}tile{idx}_L{args.layer_idx}_{head_tag}.png"
            plot_attention_overlay(x,
                                    attn_map,
                                    save_path,
                                    attn_thresh=args.attn_thresh,
                                    attn_alpha=args.attn_alpha,
                                    attn_display_gamma=args.attn_display_gamma,
                                )


            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{n_tiles} tiles")

        total_tiles += n_tiles
        print(f"  Saved {n_tiles} overlays -> {cur_out}")

    print(f"\nDone. Processed {len(h5_files)} files, {total_tiles} total tiles. Output: {out_dir}")


if __name__ == "__main__":
    main()
