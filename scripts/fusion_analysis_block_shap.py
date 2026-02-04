#!/usr/bin/env python3
"""
Block-level SHAP (Section 2): Which resolution does the fusion model rely on?

Reads existing SHAP outputs (per-dim CSV or cached .npy), aggregates mean |SHAP|
per block (5x, 10x, 20x), and reports fraction of total importance per block.
No new SHAP run required.

Usage:
  # From per-dimension CSV (e.g. from shap_stability_correlation.py --save_csv):
  python scripts/fusion_analysis_block_shap.py \
    --shap_csv shap_stability_out_conch/shap_vs_stability_per_dim.csv \
    --block_dims 768 768 768 \
    --block_names 5x 10x 20x \
    --out_dir fusion_analysis_block_shap_conch

  # From cached SHAP values (shap_values.npy):
  python scripts/fusion_analysis_block_shap.py \
    --shap_npy shap_stability_cache_conch/shap_values.npy \
    --block_dims 768 768 768 \
    --block_names 5x 10x 20x \
    --out_dir fusion_analysis_block_shap_conch
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def block_importance_from_csv(csv_path: str, block_dims: list) -> tuple:
    """
    Read per-dimension SHAP importance from CSV; aggregate by block.
    CSV must have columns: dim_id_global (or row index), shap_importance.

    Returns:
        block_means: list of mean(|SHAP|) per block
        block_sums: list of sum(|SHAP|) per block (for fraction of total)
        total_sum: sum over all dims
    """
    df = pd.read_csv(csv_path)
    if "shap_importance" not in df.columns:
        raise ValueError(f"CSV must have 'shap_importance' column. Got: {list(df.columns)}")

    # Allow dim_id_global or index
    if "dim_id_global" in df.columns:
        df = df.sort_values("dim_id_global").reset_index(drop=True)
    importance = np.asarray(df["shap_importance"], dtype=np.float64)
    n_dims = len(importance)

    total_dim = sum(block_dims)
    if n_dims != total_dim:
        # Infer 3 equal blocks from CSV row count (CONCH=768, MUSK=1024, HOPTIMUS=1536 per block)
        if n_dims % 3 != 0:
            raise ValueError(
                f"CSV has {n_dims} rows but block_dims sum to {total_dim} and rows not divisible by 3. "
                f"block_dims={block_dims}"
            )
        d = n_dims // 3
        block_dims = [d, d, n_dims - 2 * d]

    block_means = []
    block_sums = []
    offset = 0
    for d in block_dims:
        imp_block = importance[offset : offset + d]
        block_means.append(float(np.mean(imp_block)))
        block_sums.append(float(np.sum(imp_block)))
        offset += d

    total_sum = sum(block_sums)
    return block_means, block_sums, total_sum, block_dims


def block_importance_from_npy(npy_path: str, block_dims: list) -> tuple:
    """
    Load shap_values.npy (shape n_samples x n_features or n_samples x n_features x n_classes),
    compute mean |SHAP| per dimension then aggregate by block.

    Returns:
        block_means, block_sums, total_sum
    """
    shap_vals = np.load(npy_path)
    # Handle binary: out[1] is class 1 SHAP values, shape (n_samples, n_features)
    if shap_vals.ndim == 3:
        # Take class 1 importance
        shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
    abs_shap = np.abs(shap_vals)
    importance = np.mean(abs_shap, axis=0)  # (n_features,)
    n_dims = len(importance)

    total_dim = sum(block_dims)
    if n_dims != total_dim:
        if n_dims % 3 != 0:
            raise ValueError(
                f"SHAP array has {n_dims} features but block_dims sum to {total_dim} and n_dims not divisible by 3. "
                f"block_dims={block_dims}"
            )
        d = n_dims // 3
        block_dims = [d, d, n_dims - 2 * d]

    block_means = []
    block_sums = []
    offset = 0
    for d in block_dims:
        imp_block = importance[offset : offset + d]
        block_means.append(float(np.mean(imp_block)))
        block_sums.append(float(np.sum(imp_block)))
        offset += d

    total_sum = sum(block_sums)
    return block_means, block_sums, total_sum, block_dims


def run_block_shap(
    block_dims: list,
    block_names: list,
    out_dir: str,
    shap_csv: str = None,
    shap_npy: str = None,
):
    if (shap_csv is None) == (shap_npy is None):
        raise ValueError("Exactly one of --shap_csv or --shap_npy must be set.")

    if len(block_dims) != len(block_names):
        raise ValueError("block_dims and block_names must have the same length.")

    if shap_csv:
        block_means, block_sums, total_sum, block_dims = block_importance_from_csv(shap_csv, block_dims)
    else:
        block_means, block_sums, total_sum, block_dims = block_importance_from_npy(shap_npy, block_dims)

    # Fractions of total importance (by sum per block)
    fractions = [s / total_sum if total_sum > 0 else 0.0 for s in block_sums]

    # Report
    print("\n========== Block-level SHAP: which resolution does fusion rely on? ==========")
    print(f"  Total importance (sum |SHAP| over all dims): {total_sum:.6f}")
    print()
    for name, mean_imp, block_sum, frac in zip(block_names, block_means, block_sums, fractions):
        print(f"  {name}:  mean(|SHAP|) = {mean_imp:.6f},  sum = {block_sum:.6f},  fraction = {frac:.2%}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    # Table
    table_path = os.path.join(out_dir, "block_level_shap.csv")
    rows = [
        {
            "block": name,
            "mean_abs_shap": mean_imp,
            "sum_abs_shap": block_sum,
            "fraction_of_total": frac,
            "n_dims": d,
        }
        for name, mean_imp, block_sum, frac, d in zip(
            block_names, block_means, block_sums, fractions, block_dims
        )
    ]
    pd.DataFrame(rows).to_csv(table_path, index=False)
    print(f"[INFO] Table saved to {table_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(block_names))
    ax.bar(x, fractions, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(block_names))))
    ax.set_xticks(x)
    ax.set_xticklabels(block_names)
    ax.set_ylabel("Fraction of total SHAP importance")
    ax.set_title("Fusion model: which resolution block does it rely on?")
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "block_level_shap_bars.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Bar chart saved to {bar_path}")

    return block_means, block_sums, fractions


def main():
    parser = argparse.ArgumentParser(
        description="Block-level SHAP: aggregate per-dim SHAP into 5x/10x/20x blocks and report fraction of importance."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shap_csv", help="Path to per-dimension CSV (e.g. shap_vs_stability_per_dim.csv)")
    group.add_argument("--shap_npy", help="Path to cached shap_values.npy")
    parser.add_argument(
        "--block_dims",
        nargs="+",
        type=int,
        required=True,
        help="Feature dimension per block, e.g. 768 768 768 for 5x 10x 20x CONCH",
    )
    parser.add_argument(
        "--block_names",
        nargs="+",
        default=["5x", "10x", "20x"],
        help="Labels for each block (default: 5x 10x 20x)",
    )
    parser.add_argument("--out_dir", default="fusion_analysis_block_shap", help="Output directory for table and plot")

    args = parser.parse_args()

    if not os.path.isfile(args.shap_csv or args.shap_npy):
        p = args.shap_csv or args.shap_npy
        raise FileNotFoundError(f"SHAP input not found: {p}")

    run_block_shap(
        block_dims=args.block_dims,
        block_names=args.block_names,
        out_dir=args.out_dir,
        shap_csv=args.shap_csv,
        shap_npy=args.shap_npy,
    )
    print("Done.")


if __name__ == "__main__":
    main()
