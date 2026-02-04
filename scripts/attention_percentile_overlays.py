#!/usr/bin/env python3
"""
Create attention-overlay folders per tile with percentile cutoffs.

For each tile we capture a single attention map and then render overlays
inside a tile-specific folder. Each folder name encodes whether the tile
is tumorous so downstream review can separate tumor/benign tiles easily.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

import attention_from_h5 as base


DEFAULT_PERCENTILES = "10,30,50,70,90"


def parse_percentiles(value: str) -> List[float]:
    chunks: List[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            pct = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid percentile value: {token}") from exc
        if not 0.0 <= pct <= 100.0:
            raise argparse.ArgumentTypeError(f"Percentile must be 0-100, got {pct}")
        chunks.append(pct)
    if not chunks:
        raise argparse.ArgumentTypeError("At least one percentile must be provided.")
    return sorted(set(chunks))


def load_labels_from_h5(h5_path: Path, dataset_name: str) -> Optional[np.ndarray]:
    with h5py.File(h5_path, "r") as h5_file:
        if dataset_name in h5_file:
            return h5_file[dataset_name][:]
    return None


def make_tile_tag(label: Optional[int]) -> str:
    if label == 1:
        return "tumor"
    if label == 0:
        return "benign"
    return "unknown"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Render overlays at multiple percentile thresholds per tile."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--h5", type=str, help="Single H5 file.")
    group.add_argument("--h5-dir", type=str, help="Directory of H5/HDF5 files.")
    group.add_argument("--glob", type=str, help="Glob or directory path.")

    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=["conch", "hoptimus", "musk"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-tiles", type=int, default=100)
    ap.add_argument("--layer-idx", type=int, default=-1)
    ap.add_argument("--head-idx", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    ap.add_argument("--attn-alpha", type=float, default=0.5)
    ap.add_argument(
        "--attn-display-gamma",
        type=float,
        default=0.4,
        help="Power (gamma) for display transform; <1 spreads low attention (default 0.4). Use 1.0 for linear.",
    )
    ap.add_argument("--label-dataset", type=str, default="labels", help="Dataset name containing tumor labels.")
    ap.add_argument(
        "--percentiles",
        type=str,
        default=DEFAULT_PERCENTILES,
        help="Comma-separated percentiles (default: 10,30,50,70,90).",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    percentiles = parse_percentiles(args.percentiles)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    hf_token = os.environ.get(args.hf_token_env, None)

    musk_forward = False
    if args.model == "conch":
        model, tfm, _ = base.load_conch_v15(device=device, prefer_hf=True, hf_token=hf_token)
        expected = getattr(model, "expected_img_size", None)
    elif args.model == "hoptimus":
        model, tfm, _ = base.load_hoptimus1(
            device=device, prefer_hf=True, hf_token=hf_token, img_size=224
        )
        expected = getattr(model, "expected_img_size", 224)
    else:
        model, tfm, _ = base.load_musk(device=device, hf_token=hf_token)
        expected = getattr(model, "expected_img_size", None)
        musk_forward = True

    model.eval()
    param_dtype = next(model.parameters()).dtype
    resize_size = args.img_size if args.img_size is not None else expected

    h5_files = base.list_h5_files(args.h5, args.h5_dir, args.glob)
    if not h5_files:
        raise FileNotFoundError("No H5/HDF5 files found for the given input.")

    total_tiles = 0
    for fi, h5_path in enumerate(h5_files, start=1):
        print(f"\n[{fi}/{len(h5_files)}] {h5_path.name}")
        try:
            tiles = base.load_tiles_from_h5(h5_path)
        except Exception as exc:
            print(f"  [WARN] Skipping {h5_path.name}: {exc}")
            continue

        labels = load_labels_from_h5(h5_path, args.label_dataset)
        if labels is None:
            print(f"  [WARN] '{args.label_dataset}' not found; defaulting status to unknown.")

        n_tiles = min(args.max_tiles, len(tiles))
        slide_out = out_dir / h5_path.stem
        slide_out.mkdir(parents=True, exist_ok=True)

        for idx in range(n_tiles):
            tag = make_tile_tag(labels[idx] if labels is not None and idx < len(labels) else None)
            tile_folder = slide_out / f"tile{idx:04d}_{tag}"
            tile_folder.mkdir(parents=True, exist_ok=True)

            img = base.tile_to_pil(tiles[idx])
            img = base.resize_pil_if_needed(img, resize_size)

            x = tfm(img).unsqueeze(0).to(device=device, dtype=param_dtype)
            attn_map = base.get_attention_map(
                model=model,
                x=x,
                device=device,
                layer_idx=args.layer_idx,
                head_idx=args.head_idx,
                musk_forward=musk_forward,
            )

            arr = np.asarray(attn_map, dtype=np.float32)
            amin, amax = float(arr.min()), float(arr.max())
            norm = (arr - amin) / (amax - amin + 1e-8)

            for pct in percentiles:
                thresh = float(np.percentile(norm, pct))
                save_path = tile_folder / f"threshold_p{int(pct):02d}.png"
                base.plot_attention_overlay(
                    x,
                    attn_map,
                    save_path,
                    attn_thresh=thresh,
                    attn_alpha=args.attn_alpha,
                    attn_display_gamma=args.attn_display_gamma,
                )

            if (idx + 1) % 10 == 0 or idx == n_tiles - 1:
                print(f"  Tile {idx+1}/{n_tiles} -> {tile_folder}")

        total_tiles += n_tiles
        print(f"  Saved {n_tiles} tiles in {slide_out}")

    print(f"\nDone. Processed {len(h5_files)} files, {total_tiles} tiles. Output: {out_dir}")


if __name__ == "__main__":
    main()
