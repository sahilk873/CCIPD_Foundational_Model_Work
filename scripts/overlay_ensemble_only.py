#!/usr/bin/env python3
"""
overlay_ensemble_only.py

Generate patch-level TP/FN/FP overlays for an ENSEMBLE model only.

Inputs:
  - Three directories of .h5 files (same basenames per slide)
  - A directory of slide JPEGs (same stem as .h5, e.g., SLIDE_001.h5 -> SLIDE_001.jpg)
  - A trained ensemble model weights file (state_dict)
  - models.py must define DeepMLPEnsemble(input_dim=...)

Each .h5 must contain:
  - coords:   (N, 2)
  - features: (N, D)
  - labels:   (N,)   (required for TP/FN/FP; if missing, slide is skipped)

By default, this script REQUIRES coords arrays to be identical across the three dirs.
If your patch ordering differs, use --align_by_coords to match rows by coordinate values.
"""

import os
import glob
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.patches import Patch

from models import DeepMLPEnsemble  # must exist in your models.py


# ------------------------ IO helpers ------------------------

def load_h5_data(h5_path):
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        features = f["features"][:]
        labels = f["labels"][:] if "labels" in f else None
    return coords, features, labels


def predict_classes_and_probs(model, features, device):
    model.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds, probs


def find_common_files(dir1, dir2, dir3):
    s1 = {os.path.basename(p) for p in glob.glob(os.path.join(dir1, "*.h5"))}
    s2 = {os.path.basename(p) for p in glob.glob(os.path.join(dir2, "*.h5"))}
    s3 = {os.path.basename(p) for p in glob.glob(os.path.join(dir3, "*.h5"))}
    return sorted(s1 & s2 & s3)


# ------------------------ Coordinate mode detection ------------------------

def detect_coords_mode(coords, tile_size):
    """
    Heuristically detect whether coords are tile TOP-LEFT or CENTER.
    """
    c = np.asarray(coords, dtype=float)
    if len(c) == 0:
        return "topleft"

    rx = (c[:, 0] - c[:, 0].min()) % tile_size
    ry = (c[:, 1] - c[:, 1].min()) % tile_size

    def dist_to_half(x):
        d = np.abs(x - tile_size / 2.0)
        return np.minimum(d, tile_size - d)

    mean_dist_zero = 0.5 * (np.mean(np.minimum(rx, tile_size - rx)) +
                            np.mean(np.minimum(ry, tile_size - ry)))
    mean_dist_half = 0.5 * (np.mean(dist_to_half(rx)) + np.mean(dist_to_half(ry)))

    return "center" if mean_dist_half + 1e-6 < mean_dist_zero else "topleft"


# ------------------------ Optional robust alignment by coords ------------------------

def align_three_by_coords(coords1, feats1, labels1, coords2, feats2, labels2, coords3, feats3, labels3):
    """
    Align rows across three sources by exact (x,y) coordinate match.
    Keeps only intersection of coords across all three.

    Returns aligned coords (in source1's order), feats1/2/3, labels (from source1).
    """
    c1 = np.asarray(coords1)
    c2 = np.asarray(coords2)
    c3 = np.asarray(coords3)

    # Build maps: coord -> row index
    m1 = { (int(x), int(y)): i for i, (x, y) in enumerate(c1) }
    m2 = { (int(x), int(y)): i for i, (x, y) in enumerate(c2) }
    m3 = { (int(x), int(y)): i for i, (x, y) in enumerate(c3) }

    common = sorted(set(m1.keys()) & set(m2.keys()) & set(m3.keys()))
    if not common:
        return None

    idx1 = np.array([m1[k] for k in common], dtype=int)
    idx2 = np.array([m2[k] for k in common], dtype=int)
    idx3 = np.array([m3[k] for k in common], dtype=int)

    coords = c1[idx1]
    f1 = feats1[idx1]
    f2 = feats2[idx2]
    f3 = feats3[idx3]

    # Labels: prefer source1 if present, else source2, else source3
    lab = None
    if labels1 is not None:
        lab = np.asarray(labels1).ravel()[idx1]
    elif labels2 is not None:
        lab = np.asarray(labels2).ravel()[idx2]
    elif labels3 is not None:
        lab = np.asarray(labels3).ravel()[idx3]

    return coords, f1, f2, f3, lab


# ------------------------ Overlay ------------------------

def overlay_confusion_map(
    slide_img_rgb,            # np.uint8 (H, W, 3), RGB
    patch_preds,              # (N,) 0/1
    patch_labels,             # (N,) 0/1
    coords,                   # (N,2) coords in ORIGINAL slide space
    tile_size,                # original patch size (pixels at level-0)
    coords_mode="topleft",    # "topleft" or "center"
    alpha=0.85,
    pad_px=6,
    debug_print=False,
):
    """
    TN tiles are transparent (not drawn).
    Tiles are snapped to a discrete grid to guarantee no gaps between cells.
    """
    preds  = np.asarray(patch_preds,  dtype=np.uint8).ravel()
    gts    = np.asarray(patch_labels, dtype=np.uint8).ravel()
    coords = np.asarray(coords,       dtype=float)
    assert len(preds) == len(gts) == len(coords), "Mismatch among preds/labels/coords."

    H, W, _ = slide_img_rgb.shape
    out = slide_img_rgb.copy()

    # Convert coords to TOP-LEFT if they are centers
    tl = coords - tile_size / 2.0 if coords_mode == "center" else coords.copy()

    # Establish a grid by normalizing to the minimum and dividing by tile_size
    min_x, min_y = float(np.min(tl[:, 0])), float(np.min(tl[:, 1]))
    col_idx = np.rint((tl[:, 0] - min_x) / tile_size).astype(int)
    row_idx = np.rint((tl[:, 1] - min_y) / tile_size).astype(int)

    n_cols = int(col_idx.max()) + 1
    n_rows = int(row_idx.max()) + 1

    cell_w = W / max(n_cols, 1)
    cell_h = H / max(n_rows, 1)

    if debug_print:
        print(f"[DEBUG] grid: {n_cols}x{n_rows}, image: {W}x{H}, cell: {cell_w:.3f}x{cell_h:.3f}, mode={coords_mode}")

    # Confusion categories (TN not drawn)
    tp = (preds == 1) & (gts == 1)
    fn = (preds == 0) & (gts == 1)
    fp = (preds == 1) & (gts == 0)

    COLORS = {
        "TP": (0, 255, 0),
        "FN": (255, 0, 0),
        "FP": (0, 0, 255),
    }

    def draw_filled_alpha(img, x1, y1, x2, y2, color, a, pad=0):
        x1 = max(0, int(np.floor(x1)) - pad)
        y1 = max(0, int(np.floor(y1)) - pad)
        x2 = min(W, int(np.ceil(x2)) + pad)
        y2 = min(H, int(np.ceil(y2)) + pad)
        if x1 >= x2 or y1 >= y2:
            return
        region = img[y1:y2, x1:x2]
        color_patch = np.full_like(region, color, dtype=np.uint8)
        blended = cv2.addWeighted(region, 1 - a, color_patch, a, 0)
        img[y1:y2, x1:x2] = blended

    for i in range(len(col_idx)):
        x1 = col_idx[i] * cell_w
        y1 = row_idx[i] * cell_h
        x2 = (col_idx[i] + 1) * cell_w
        y2 = (row_idx[i] + 1) * cell_h

        if tp[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["TP"], alpha, pad=pad_px)
        elif fn[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["FN"], alpha, pad=pad_px)
        elif fp[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["FP"], alpha, pad=pad_px)
        # TN: draw nothing

    legend = [
        Patch(color=np.array(COLORS["TP"]) / 255.0, label="TP (pred=1, gt=1)"),
        Patch(color=np.array(COLORS["FN"]) / 255.0, label="FN (pred=0, gt=1)"),
        Patch(color=np.array(COLORS["FP"]) / 255.0, label="FP (pred=1, gt=0)"),
    ]
    return out, legend


# ------------------------ Main ------------------------

def main(args):
    plt.rcParams["font.size"] = 16

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    common_files = find_common_files(args.dir1, args.dir2, args.dir3)
    if not common_files:
        raise ValueError("No common .h5 files across the three directories.")
    if args.max_slides is not None:
        common_files = common_files[: min(args.max_slides, len(common_files))]

    os.makedirs(args.save_dir, exist_ok=True)

    # Infer input dims from a sample slide (and ensure concat order matches YOUR training)
    sample = common_files[0]
    _, f1, _ = load_h5_data(os.path.join(args.dir1, sample))
    _, f2, _ = load_h5_data(os.path.join(args.dir2, sample))
    _, f3, _ = load_h5_data(os.path.join(args.dir3, sample))
    d1, d2, d3 = f1.shape[1], f2.shape[1], f3.shape[1]
    ensemble_in = d1 + d2 + d3
    print(f"[INFO] Feature dims — dir1: {d1}, dir2: {d2}, dir3: {d3} (Ensemble input: {ensemble_in})")
    print("[INFO] CONCAT ORDER is [dir1, dir2, dir3] — this must match ensemble training order!")

    # Load ensemble model
    model = DeepMLPEnsemble(input_dim=ensemble_in).to(device)
    model.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    model.eval()

    # Process each slide
    for base in common_files:
        p1 = os.path.join(args.dir1, base)
        p2 = os.path.join(args.dir2, base)
        p3 = os.path.join(args.dir3, base)

        coords1, feat1, lab1 = load_h5_data(p1)
        coords2, feat2, lab2 = load_h5_data(p2)
        coords3, feat3, lab3 = load_h5_data(p3)

        # Align / validate coords
        labels = lab1 if lab1 is not None else (lab2 if lab2 is not None else lab3)
        if labels is None:
            print(f"[WARN] {base}: missing labels in all three .h5 files — skipping.")
            continue

        if args.align_by_coords:
            aligned = align_three_by_coords(coords1, feat1, lab1, coords2, feat2, lab2, coords3, feat3, lab3)
            if aligned is None:
                print(f"[WARN] {base}: no overlapping coords across dirs — skipping.")
                continue
            coords, feat1a, feat2a, feat3a, labels = aligned
            coords_for_overlay = coords
            feat1, feat2, feat3 = feat1a, feat2a, feat3a
        else:
            if not (np.array_equal(coords1, coords2) and np.array_equal(coords1, coords3)):
                print(f"[WARN] {base}: coords mismatch across dirs — skipping (use --align_by_coords to fix).")
                continue
            coords_for_overlay = coords1

        # Coords mode
        coords_mode_final = args.coords_mode
        if coords_mode_final == "auto":
            coords_mode_final = detect_coords_mode(coords_for_overlay, args.tile_size)

        # Load slide JPEG
        slide_jpg = os.path.join(args.slides_dir, os.path.splitext(base)[0] + ".jpg")
        if not os.path.exists(slide_jpg):
            print(f"[WARN] {base}: slide image missing at {slide_jpg} — skipping.")
            continue
        slide_img = np.array(Image.open(slide_jpg).convert("RGB"))

        # Ensemble inference (concat order: [dir1, dir2, dir3])
        concat_feats = np.concatenate([feat1, feat2, feat3], axis=1)
        preds, _ = predict_classes_and_probs(model, concat_feats, device)

        overlay_img, legend = overlay_confusion_map(
            slide_img_rgb=slide_img,
            patch_preds=preds,
            patch_labels=labels,
            coords=coords_for_overlay,
            tile_size=args.tile_size,
            coords_mode=coords_mode_final,
            alpha=args.alpha,
            pad_px=args.pad_px,
            debug_print=args.debug,
        )

        # Optional orientation fix
        if args.rotate_k != 0:
            overlay_img = np.rot90(overlay_img, k=args.rotate_k)

        # Plot + legend and save
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        ax.imshow(overlay_img)
        ax.set_title(f"Ensemble vs Ground Truth — {os.path.splitext(base)[0]}", fontsize=18)
        ax.axis("off")
        ax.legend(handles=legend, loc="lower center", ncol=3, frameon=True, fontsize=12)

        out_path = os.path.join(args.save_dir, f"{os.path.splitext(base)[0]}_ensemble_overlay.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved overlay → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate ensemble-only TP/FN/FP overlays from three feature dirs.")
    p.add_argument("--dir1", required=True, help="Feature dir #1 (concat block 1).")
    p.add_argument("--dir2", required=True, help="Feature dir #2 (concat block 2).")
    p.add_argument("--dir3", required=True, help="Feature dir #3 (concat block 3).")
    p.add_argument("--slides_dir", required=True, help="Directory of slide JPGs with same stem as .h5 files.")
    p.add_argument("--ensemble_model", required=True, help="Path to ensemble .pth state_dict.")
    p.add_argument("--tile_size", type=int, default=512, help="Patch size in ORIGINAL slide pixels.")
    p.add_argument("--coords_mode", choices=["auto", "topleft", "center"], default="auto",
                   help="Coordinate semantics in H5. 'auto' tries to detect.")
    p.add_argument("--alpha", type=float, default=0.85, help="Overlay opacity for TP/FN/FP (0..1).")
    p.add_argument("--pad_px", type=int, default=6, help="Pad each tile by this many pixels in image space.")
    p.add_argument("--save_dir", required=True, help="Output directory for overlay PNGs.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_slides", type=int, default=None, help="If set, process only first N common slides.")
    p.add_argument("--align_by_coords", action="store_true",
                   help="Align patches by coords if ordering differs across dirs (keeps coord intersection).")
    p.add_argument("--rotate_k", type=int, default=0,
                   help="Rotate output image by 90° k times (e.g., 3 matches your earlier k=3 fix).")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    main(args)


