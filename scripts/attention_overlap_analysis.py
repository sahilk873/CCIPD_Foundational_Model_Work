#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv


# -------------------------
# Helpers: filenames -> tile_id
# -------------------------
def extract_tile_id(name: str) -> Optional[int]:
    """Extract numeric tile ID from filename, e.g. tile17_L-1.png -> 17."""
    m = re.search(r"tile(\d+)", name)
    return int(m.group(1)) if m else None


def index_dir_by_tile_id(d: Path) -> Dict[int, Path]:
    """
    Map tile_id -> image path.
    If multiple files share the same tile_id, choose the shortest filename (usually the "base" one).
    """
    out: Dict[int, Path] = {}
    for p in d.iterdir():
        if not p.is_file():
            continue
        tid = extract_tile_id(p.name)
        if tid is None:
            continue
        if tid not in out:
            out[tid] = p
        else:
            # deterministic preference: shorter name tends to be "cleaner"
            if len(p.name) < len(out[tid].name):
                out[tid] = p
    return out


# -------------------------
# Image loading / normalization
# -------------------------
def load_gray01(path: Path) -> np.ndarray:
    """Load image as grayscale float32 in [0,1]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # normalize to [0,1] per-image (helps if colormaps differ)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def resize_to(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize float32 [0,1] to (H,W) bilinear."""
    H, W = size_hw
    pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((W, H), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return out


# -------------------------
# Foreground mask + percentile thresholding
# -------------------------
def foreground_mask(arr: np.ndarray, fg_percentile: float) -> np.ndarray:
    """
    Foreground mask for an attention map:
    keep pixels >= percentile(arr).
    Example: fg_percentile=10 keeps top 90% intensities, removing lowest 10% background-ish pixels.
    """
    thr = float(np.percentile(arr, fg_percentile))
    return arr >= thr


def threshold_by_percentile(arr: np.ndarray, pct: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Threshold arr into a binary mask using percentile `pct`.
    If mask is provided, percentile is computed only on masked pixels.
    """
    if mask is None:
        vals = arr.ravel()
    else:
        vals = arr[mask]
        if vals.size == 0:
            # fallback: no foreground pixels
            vals = arr.ravel()

    thr = float(np.percentile(vals, pct))
    return arr >= thr


def dice_bool(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Hard Dice between two boolean masks."""
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter) / (denom + eps))


# -------------------------
# Plotting
# -------------------------
def save_examples_figure(
    out_path: Path,
    tile_tumor: int,
    tile_benign: int,
    conch: Dict[int, Path],
    hop: Dict[int, Path],
    musk: Dict[int, Path],
):
    """
    2 rows (Tumor, Benign) × 3 columns (Conch v1.5, H-optimus-1, MUSK)
    Row labels are horizontal and placed like y-axis labels.
    """

    models = [
        ("Conch v1.5", conch),
        ("H-Optimus-1", hop),
        ("MUSK", musk),
    ]

    tiles = [tile_tumor, tile_benign]
    row_names = ["Tumor", "Benign"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(10.5, 6.5),
        constrained_layout=False
    )

    for r, tile_id in enumerate(tiles):
        for c, (model_name, idx) in enumerate(models):
            ax = axes[r, c]

            if tile_id not in idx:
                ax.text(
                    0.5, 0.5, f"tile{tile_id} missing",
                    ha="center", va="center", fontsize=11
                )
                ax.axis("off")
                continue

            img = Image.open(idx[tile_id]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")

            # Column titles only on top row
            if r == 0:
                ax.set_title(
                    model_name,
                    fontsize=13,
                    fontweight="bold",
                    pad=10
                )

    # ---- Horizontal row labels (true y-axis style) ----
    fig.text(
        0.02, 0.67, "Tumor",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center"
    )
    fig.text(
        0.02, 0.28, "Benign",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center"
    )

    # Optional super-title
    fig.suptitle(
        "Attention Maps (Tumor vs Benign)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.90,
        bottom=0.05,
        wspace=0,
        hspace=0.0
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def save_overlap_plot(
    out_path: Path,
    percentiles: List[int],
    agg: Dict[str, List[float]],
    ylabel: str = "Mean Hard Dice",
):
    """
    Line plot: Dice vs percentile for each pair.
    agg: dict(pair_name -> list of mean dice in same order as percentiles)
    """
    plt.figure(figsize=(7.5, 4.5))
    for pair_name, ys in agg.items():
        plt.plot(percentiles, ys, marker="o", label=pair_name)
    plt.xlabel("Attention threshold percentile")
    plt.ylabel(ylabel)
    plt.xticks(percentiles)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Main computation
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Attention overlap analysis: example tumor/benign tiles + Dice overlap vs percentile thresholds."
    )
    ap.add_argument("--conch-dir", type=str, required=True, help="Directory of Conch attention map PNGs.")
    ap.add_argument("--hoptimus-dir", type=str, required=True, help="Directory of H-optimus attention map PNGs.")
    ap.add_argument("--musk-dir", type=str, required=True, help="Directory of MUSK attention map PNGs.")

    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for figures + CSV.")
    ap.add_argument("--tile-tumor", type=int, required=True, help="Tile id to display as tumor example (e.g., 22).")
    ap.add_argument("--tile-benign", type=int, required=True, help="Tile id to display as benign example.")

    ap.add_argument("--percentiles", type=str, default="10,30,50,70,90",
                    help="Comma-separated percentiles for thresholding (default: 10,30,50,70,90).")

    ap.add_argument("--fg-percentile", type=float, default=10.0,
                    help="Foreground mask percentile (default 10 = drop lowest 10%% intensity pixels).")

    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of matched tiles to process (0 = all).")

    args = ap.parse_args()

    conch_dir = Path(args.conch_dir)
    hop_dir = Path(args.hoptimus_dir)
    musk_dir = Path(args.musk_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    percentiles = [int(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    fg_pct = float(args.fg_percentile)

    # Index directories by tile_id
    conch_idx = index_dir_by_tile_id(conch_dir)
    hop_idx = index_dir_by_tile_id(hop_dir)
    musk_idx = index_dir_by_tile_id(musk_dir)

    # Matched tile IDs across all three
    common_ids = sorted(set(conch_idx.keys()) & set(hop_idx.keys()) & set(musk_idx.keys()))
    if not common_ids:
        raise SystemExit("No common tile IDs found across conch/hoptimus/musk directories (need filenames containing tile###).")

    if args.limit > 0:
        common_ids = common_ids[: args.limit]

    # Save example figure
    examples_path = out_dir / "attn_examples_tumor_vs_benign.png"
    save_examples_figure(
        out_path=examples_path,
        tile_tumor=args.tile_tumor,
        tile_benign=args.tile_benign,
        conch=conch_idx,
        hop=hop_idx,
        musk=musk_idx,
    )

    # Compute dice curves for each pair vs percentiles
    pairs = [
        ("Conch vs H-optimus", conch_idx, hop_idx),
        ("Conch vs MUSK", conch_idx, musk_idx),
        ("H-optimus vs MUSK", hop_idx, musk_idx),
    ]

    # Per-tile CSV rows
    csv_rows: List[dict] = []

    # Aggregate: pair -> percentile -> list of dice
    dice_store: Dict[str, Dict[int, List[float]]] = {pn: {p: [] for p in percentiles} for pn, _, _ in pairs}

    for tid in common_ids:
        # Load and normalize maps
        A_conch = load_gray01(conch_idx[tid])
        A_hop = resize_to(load_gray01(hop_idx[tid]), A_conch.shape)
        A_musk = resize_to(load_gray01(musk_idx[tid]), A_conch.shape)

        # Foreground mask: union across models so we ignore low-intensity background-y pixels consistently
        fg_c = foreground_mask(A_conch, fg_pct)
        fg_h = foreground_mask(A_hop, fg_pct)
        fg_m = foreground_mask(A_musk, fg_pct)
        fg_union = np.logical_or(np.logical_or(fg_c, fg_h), fg_m)

        for pair_name, idx1, idx2 in pairs:
            # Select the two arrays in a consistent way
            if pair_name == "Conch vs H-optimus":
                X, Y = A_conch, A_hop
            elif pair_name == "Conch vs MUSK":
                X, Y = A_conch, A_musk
            else:  # H-optimus vs MUSK
                X, Y = A_hop, A_musk

            for p in percentiles:
                X_bin = threshold_by_percentile(X, p, mask=fg_union)
                Y_bin = threshold_by_percentile(Y, p, mask=fg_union)
                d = dice_bool(X_bin, Y_bin)
                dice_store[pair_name][p].append(d)

                csv_rows.append({
                    "tile_id": tid,
                    "pair": pair_name,
                    "percentile": p,
                    "hard_dice": d,
                    "fg_percentile": fg_pct,
                })

    # Write CSV
    csv_path = out_dir / "dice_vs_percentile.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_id", "pair", "percentile", "hard_dice", "fg_percentile"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    # Aggregate means for plot
    agg_means: Dict[str, List[float]] = {}
    for pair_name in dice_store:
        ys = []
        for p in percentiles:
            arr = np.array(dice_store[pair_name][p], dtype=np.float64)
            ys.append(float(arr.mean()) if arr.size else float("nan"))
        agg_means[pair_name] = ys

    plot_path = out_dir / "dice_overlap_vs_percentile.png"
    save_overlap_plot(plot_path, percentiles, agg_means, ylabel="Mean Hard Dice (foreground-masked)")

    # Print a short summary
    print(f"Processed tiles: {len(common_ids)}")
    print(f"Foreground masking: drop lowest {fg_pct:.1f}% intensities per-map; use union mask across models")
    print(f"Saved examples: {examples_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print("Mean Dice by pair (percentiles):")
    for pair_name, ys in agg_means.items():
        msg = ", ".join([f"p{p}={y:.4f}" for p, y in zip(percentiles, ys)])
        print(f"  {pair_name}: {msg}")


if __name__ == "__main__":
    main()


