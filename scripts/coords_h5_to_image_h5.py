#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

import h5py
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def find_matching_slide(h5_name: str, slide_dir: Path) -> Path:
    """
    Match:
      <stem>_patches.h5  ->  <stem>.svs (or other WSI ext)
    Example:
      TCGA-...4d9c_patches.h5 -> TCGA-...4d9c.svs
    """
    slide_exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")

    stem = Path(h5_name).stem
    stem = re.sub(r"(_patches|_coords|_features|_images)$", "", stem)

    candidates = []
    # exact expected filename
    for ext in slide_exts:
        candidates.append(slide_dir / f"{stem}{ext}")

    for p in candidates:
        if p.is_file():
            return p

    # fallback: substring search (in case of extra suffixes in slide filenames)
    slides = [p for p in slide_dir.iterdir() if p.is_file() and p.suffix.lower() in slide_exts]
    matches = [p for p in slides if stem in p.name]
    if matches:
        matches = sorted(matches, key=lambda p: len(p.name))
        return matches[0]

    raise FileNotFoundError(
        f"No matching slide found for {h5_name}\n"
        f"Looking for stem: {stem}\n"
        f"In dir: {slide_dir}"
    )


def extract_tile(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    size: int,
    level: int
) -> np.ndarray:
    """
    Extract an RGB tile at (x, y) from slide.
    Returns (H, W, 3) uint8.
    """
    tile = slide.read_region(
        (int(x), int(y)),
        level,
        (size, size)
    ).convert("RGB")
    return np.asarray(tile, dtype=np.uint8)


# -----------------------------
# Main conversion
# -----------------------------
def convert_coords_h5_to_image_h5(
    coords_h5_dir: Path,
    slide_dir: Path,
    out_dir: Path,
    patch_size: int,
    level: int,
    limit: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(coords_h5_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError("No .h5 files found in coords_h5_dir")

    for h5_path in h5_files:
        print(f"\nProcessing {h5_path.name}")

        slide_path = find_matching_slide(h5_path.name, slide_dir)
        slide = openslide.OpenSlide(str(slide_path))

        with h5py.File(h5_path, "r") as f:
            if "coords" not in f:
                raise KeyError(f"{h5_path} does not contain 'coords'")
            coords = f["coords"][:]

        if limit > 0:
            coords = coords[:limit]

        images = np.zeros(
            (len(coords), patch_size, patch_size, 3),
            dtype=np.uint8
        )

        for i, (x, y) in enumerate(tqdm(coords, desc="Extracting tiles")):
            images[i] = extract_tile(
                slide,
                x=x,
                y=y,
                size=patch_size,
                level=level,
            )

        out_path = out_dir / h5_path.name.replace(".h5", "_images.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset(
                "images",
                data=images,
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset("coords", data=coords)

        slide.close()
        print(f"Saved image H5 → {out_path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Convert coords-only H5 + WSI into image-tile H5 for attention maps."
    )
    ap.add_argument("--coords-h5-dir", required=True, type=str)
    ap.add_argument("--slide-dir", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--level", type=int, default=0,
                    help="OpenSlide level to extract from (0 = highest resolution)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max tiles per slide (0 = all)")
    args = ap.parse_args()

    convert_coords_h5_to_image_h5(
        coords_h5_dir=Path(args.coords_h5_dir),
        slide_dir=Path(args.slide_dir),
        out_dir=Path(args.out_dir),
        patch_size=args.patch_size,
        level=args.level,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
