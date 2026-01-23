#!/usr/bin/env python3
# make_6panel_tumor_benign_examples.py
#
# Extracted "image generation" only.
# Takes exactly 6 images:
#   - Tumor row:   Conch, H-Optimus, MUSK (one each)
#   - Benign row:  Conch, H-Optimus, MUSK (one each)
#
# Produces a 2x3 panel figure matching your prior style,
# plus a blue→red attention legend bar on the right.

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl


def open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGB")


def save_6panel_figure(
    out_path: Path,
    tumor_conch: Path,
    tumor_hoptimus: Path,
    tumor_musk: Path,
    benign_conch: Path,
    benign_hoptimus: Path,
    benign_musk: Path,
    title: str = "Attention Maps (Tumor vs Benign)",
    col_titles: Tuple[str, str, str] = ("Conch v1.5", "H-Optimus-1", "MUSK"),
):
    """
    2 rows (Tumor, Benign) × 3 columns (Conch v1.5, H-Optimus-1, MUSK)
    Row labels are horizontal and placed like y-axis labels.
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(10.5, 6.5),
        constrained_layout=False,
    )

    # Order: rows then columns
    img_paths = [
        [tumor_conch, tumor_hoptimus, tumor_musk],
        [benign_conch, benign_hoptimus, benign_musk],
    ]

    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            img = open_rgb(img_paths[r][c])
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=10)

    # Horizontal row labels
    fig.text(0.02, 0.67, "Tumor", fontsize=12, fontweight="bold", ha="left", va="center")
    fig.text(0.02, 0.28, "Non-Tumor", fontsize=12, fontweight="bold", ha="left", va="center")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Leave space on the right for a colorbar
    plt.subplots_adjust(
        left=0.12,
        right=0.93,   # make room for legend
        top=0.90,
        bottom=0.05,
        wspace=0.0,
        hspace=0.0,
    )

    # === ATTENTION LEGEND BAR (BLUE -> RED) ===
    cmap = mpl.cm.coolwarm  # blue (low) -> red (high)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add small vertical colorbar on the right
    cax = fig.add_axes([0.95, 0.15, 0.015, 0.7])  # [x, y, width, height]
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Attention", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Create a 2x3 panel figure from 6 attention images (3 tumor, 3 benign; one per FM)."
    )

    # Tumor row
    ap.add_argument("--tumor-conch", type=str, required=True, help="Tumor example image for Conch.")
    ap.add_argument("--tumor-hoptimus", type=str, required=True, help="Tumor example image for H-Optimus.")
    ap.add_argument("--tumor-musk", type=str, required=True, help="Tumor example image for MUSK.")

    # Benign row
    ap.add_argument("--benign-conch", type=str, required=True, help="Benign example image for Conch.")
    ap.add_argument("--benign-hoptimus", type=str, required=True, help="Benign example image for H-Optimus.")
    ap.add_argument("--benign-musk", type=str, required=True, help="Benign example image for MUSK.")

    ap.add_argument("--out", type=str, required=True, help="Output figure path (e.g., .png).")
    ap.add_argument("--title", type=str, default="Attention Maps (Tumor vs Benign)", help="Figure title.")

    ap.add_argument("--col-title-conch", type=str, default="Conch v1.5", help="Column title for Conch.")
    ap.add_argument("--col-title-hoptimus", type=str, default="H-Optimus-1", help="Column title for H-Optimus.")
    ap.add_argument("--col-title-musk", type=str, default="MUSK", help="Column title for MUSK.")

    return ap.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out)

    save_6panel_figure(
        out_path=out_path,
        tumor_conch=Path(args.tumor_conch),
        tumor_hoptimus=Path(args.tumor_hoptimus),
        tumor_musk=Path(args.tumor_musk),
        benign_conch=Path(args.benign_conch),
        benign_hoptimus=Path(args.benign_hoptimus),
        benign_musk=Path(args.benign_musk),
        title=args.title,
        col_titles=(args.col_title_conch, args.col_title_hoptimus, args.col_title_musk),
    )

    print(f"[OK] Saved figure: {out_path.resolve()}")


if __name__ == "__main__":
    main()
