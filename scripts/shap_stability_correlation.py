#!/usr/bin/env python3
"""
SHAP vs dimension stability: compute per-dimension SHAP importance for a fusion
model, join with per-dimension stability (mean_mad etc.), run correlation tests,
and plot SHAP importance vs instability.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import shap
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import fusion model components (same architecture as training)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from fusion_model import FusedFeatureDataset, DeepMLP


def get_block_dims(dirs):
    """Infer feature dimension per dir from first .h5 in each directory."""
    block_dims = []
    for d in dirs:
        files = [f for f in os.listdir(d) if f.endswith(".h5")]
        if not files:
            raise ValueError(f"No .h5 files in {d}")
        path = os.path.join(d, sorted(files)[0])
        with h5py.File(path, "r") as f:
            if "features" not in f:
                raise ValueError(f"{path}: missing 'features'")
            block_dims.append(int(f["features"].shape[1]))
    return block_dims


def build_global_to_stability_mapping(block_dims, model_names):
    """Return list of (model_name, dim_id) for each global feature index."""
    mapping = []
    offset = 0
    for dim_size, name in zip(block_dims, model_names):
        for local_d in range(dim_size):
            mapping.append((name, local_d))
        offset += dim_size
    return mapping


def load_stability_and_align(stability_csv, mapping, instability_col="mean_mad"):
    """
    Load stability CSV and return aligned arrays: instability, rank_by_stability
    (and optionally other cols) for each global dimension.
    """
    df = pd.read_csv(stability_csv)
    if "model" not in df.columns or "dim_id" not in df.columns:
        raise ValueError(f"Stability CSV must have 'model' and 'dim_id' columns. Got: {list(df.columns)}")
    if instability_col not in df.columns:
        raise ValueError(f"Instability column '{instability_col}' not in CSV. Available: {list(df.columns)}")

    instability = []
    rank_by_stability = []
    missing = []

    for g, (model_name, dim_id) in enumerate(mapping):
        row = df[(df["model"] == model_name) & (df["dim_id"] == dim_id)]
        if row.empty:
            missing.append((g, model_name, dim_id))
            continue
        instability.append(float(row[instability_col].iloc[0]))
        rank_by_stability.append(int(row["rank_by_stability"].iloc[0]) if "rank_by_stability" in df.columns else None)

    if missing:
        raise ValueError(
            f"Stability CSV missing rows for {len(missing)} dimensions. "
            f"First few: {missing[:5]}. Ensure CSV has (model, dim_id) for each block dimension."
        )

    return np.array(instability), rank_by_stability, df


def compute_shap_importance(model, X_np, background_size=50, to_explain_size=200, nsamples=500, cache_dir=None):
    """
    Run KernelExplainer on model; return per-dimension importance = mean(|SHAP|) over samples.
    model: callable-friendly (forward with tensor), will be set to eval().cpu().
    X_np: (N, D) numpy array (e.g. scaled).
    """
    model.eval()
    device = next(model.parameters()).device
    on_cpu = device.type == "cpu"

    def model_predict(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            if not on_cpu:
                t = t.cuda()
            logits = model(t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    # Background
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        bg_path = os.path.join(cache_dir, "background.npy")
        expl_path = os.path.join(cache_dir, "to_explain.npy")
        shap_path = os.path.join(cache_dir, "shap_values.npy")
    else:
        bg_path = expl_path = shap_path = None

    if cache_dir and os.path.exists(bg_path):
        background = np.load(bg_path)
        to_explain = np.load(expl_path)
    else:
        km = shap.kmeans(X_np, min(background_size, len(X_np)))
        background = km.data
        n_explain = min(to_explain_size, len(X_np))
        to_explain = X_np[:n_explain]
        if cache_dir:
            np.save(bg_path, background)
            np.save(expl_path, to_explain)

    if cache_dir and os.path.exists(shap_path):
        shap_vals = np.load(shap_path)
    else:
        explainer = shap.KernelExplainer(model_predict, background)
        out = explainer.shap_values(to_explain, nsamples=nsamples)
        shap_vals = out[1] if isinstance(out, list) else out
        if cache_dir:
            np.save(shap_path, shap_vals)

    # (n_samples, n_features) -> mean |SHAP| per feature
    abs_shap = np.abs(shap_vals)
    if abs_shap.ndim == 3:
        abs_shap = abs_shap.mean(axis=2)
    importance = np.mean(abs_shap, axis=0)
    return importance


def run_correlation_tests(instability, shap_importance):
    """Spearman and Pearson correlation with p-values."""
    spearman_r, spearman_p = stats.spearmanr(instability, shap_importance)
    pearson_r, pearson_p = stats.pearsonr(instability, shap_importance)
    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "n": len(instability),
    }


def plot_scatter(instability, shap_importance, model_per_dim, out_dir, instability_col="mean_mad"):
    """Scatter: x=instability, y=SHAP importance; color by model. Regression line and correlation."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(instability)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    model_per_dim = np.asarray(model_per_dim)
    models = list(np.unique(model_per_dim))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    for i, m in enumerate(models):
        mask = model_per_dim == m
        ax.scatter(instability[mask], shap_importance[mask], label=m, alpha=0.6, s=12, color=colors[i % len(colors)])

    # Overall regression line (skip if constant or < 2 points)
    if n >= 2 and np.ptp(instability) > 0 and np.ptp(shap_importance) > 0:
        z = np.polyfit(instability, shap_importance, 1)
        x_line = np.linspace(instability.min(), instability.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.8, label="Overall fit")

    r, p = stats.spearmanr(instability, shap_importance)
    if np.isnan(r):
        r, p = 0.0, 1.0
    ax.set_xlabel(f"Instability ({instability_col})")
    ax.set_ylabel("SHAP importance (mean |SHAP|)")
    ax.set_title(f"SHAP importance vs dimension instability (Spearman r={r:.3f}, p={p:.2e}, n={len(instability)})")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_vs_stability_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_by_model(instability, shap_importance, model_per_dim, out_dir, instability_col="mean_mad"):
    """One scatter per model (facet or separate small plots)."""
    os.makedirs(out_dir, exist_ok=True)
    model_per_dim = np.asarray(model_per_dim)
    models = list(np.unique(model_per_dim))
    n_models = len(models)
    if n_models == 0:
        return
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, m in enumerate(models):
        ax = axes[i]
        mask = model_per_dim == m
        x_i = np.asarray(instability)[mask]
        y_i = np.asarray(shap_importance)[mask]
        ax.scatter(x_i, y_i, alpha=0.6, s=15)
        if len(x_i) > 1:
            r, p_val = stats.spearmanr(x_i, y_i)
            if np.isnan(r):
                r, p_val = 0.0, 1.0
            if np.ptp(x_i) > 0 and np.ptp(y_i) > 0:
                z = np.polyfit(x_i, y_i, 1)
                x_line = np.linspace(x_i.min(), x_i.max(), 50)
                ax.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.8)
            ax.set_title(f"{m} (r={r:.3f}, p={p_val:.2e}, n={len(x_i)})")
        else:
            ax.set_title(f"{m} (n={len(x_i)})")
        ax.set_xlabel(f"Instability ({instability_col})")
        ax.set_ylabel("SHAP importance")
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_vs_stability_by_model.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="SHAP vs dimension stability: correlation between SHAP importance and instability."
    )
    parser.add_argument("--model", required=True, help="Path to fusion model checkpoint (.pth)")
    parser.add_argument("--dirs", nargs=3, required=True, metavar="DIR", help="Exactly 3 H5 directories (same order as training)")
    parser.add_argument("--model_names", nargs=3, required=True, metavar="NAME", help="Model names in same order as --dirs (e.g. musk hoptimus1 conch_v15)")
    parser.add_argument("--stability_csv", default="resolution_similarity_out/per_dimension_stability.csv", help="Path to per_dimension_stability.csv")
    parser.add_argument("--out_dir", default="shap_stability_out", help="Output directory for plots and optional CSV")
    parser.add_argument("--cache_dir", default=None, help="Cache directory for SHAP background and values (optional)")
    parser.add_argument("--instability_col", default="mean_mad", help="Column in stability CSV to use as instability (e.g. mean_mad, std_across_resolutions)")
    parser.add_argument("--background_size", type=int, default=50, help="Kmeans background size for SHAP")
    parser.add_argument("--to_explain_size", type=int, default=200, help="Number of samples to explain")
    parser.add_argument("--nsamples", type=int, default=500, help="Nsamples for KernelExplainer")
    parser.add_argument("--no_scale", action="store_true", help="Do not StandardScaler the features before SHAP")
    parser.add_argument("--save_csv", action="store_true", help="Save per-dimension table (dim_id_global, model, dim_id_local, instability, shap_importance, rank_by_stability)")

    args = parser.parse_args()

    dirs = list(args.dirs)
    model_names = list(args.model_names)
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.stability_csv):
        raise FileNotFoundError(f"Stability CSV not found: {args.stability_csv}")

    print("[INFO] Loading dataset (FusedFeatureDataset)...")
    dataset = FusedFeatureDataset(dirs)
    X = dataset.features.numpy()
    input_dim = X.shape[1]
    n_samples = X.shape[0]
    print(f"[INFO] Loaded {n_samples:,} samples, {input_dim} features")

    block_dims = get_block_dims(dirs)
    print(f"[INFO] Block dims: {block_dims} -> {model_names}")

    if sum(block_dims) != input_dim:
        raise ValueError(f"Block dims {block_dims} sum to {sum(block_dims)} but dataset has input_dim={input_dim}")

    mapping = build_global_to_stability_mapping(block_dims, model_names)
    assert len(mapping) == input_dim

    print("[INFO] Loading stability CSV and aligning...")
    instability, rank_by_stability, stab_df = load_stability_and_align(
        args.stability_csv, mapping, instability_col=args.instability_col
    )

    print("[INFO] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMLP(input_dim=input_dim).to(device)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    if args.no_scale:
        X_use = X
    else:
        scaler = StandardScaler()
        X_use = scaler.fit_transform(X)
    X_use = X_use.astype(np.float32)

    print("[INFO] Computing SHAP importance (this may take a while)...")
    shap_importance = compute_shap_importance(
        model,
        X_use,
        background_size=args.background_size,
        to_explain_size=args.to_explain_size,
        nsamples=args.nsamples,
        cache_dir=args.cache_dir,
    )
    assert len(shap_importance) == input_dim

    results = run_correlation_tests(instability, shap_importance)
    print("\n========== CORRELATION: Instability vs SHAP importance ==========")
    sr, sp = results["spearman_r"], results["spearman_p"]
    pr, pp = results["pearson_r"], results["pearson_p"]
    sr_str = f"{sr:.4f}" if np.isfinite(sr) else "nan"
    pr_str = f"{pr:.4f}" if np.isfinite(pr) else "nan"
    sp_str = f"{sp:.2e}" if np.isfinite(sp) else "nan"
    pp_str = f"{pp:.2e}" if np.isfinite(pp) else "nan"
    print(f"  Spearman r = {sr_str}  (p = {sp_str})")
    print(f"  Pearson r  = {pr_str}  (p = {pp_str})")
    print(f"  n (dims)   = {results['n']}")
    if np.isfinite(results["spearman_p"]) and results["spearman_p"] < 0.05 and np.isfinite(results["spearman_r"]):
        if results["spearman_r"] > 0:
            print("  Interpretation: Significant positive correlation — more unstable dimensions tend to have higher SHAP importance.")
        else:
            print("  Interpretation: Significant negative correlation — more stable dimensions tend to have higher SHAP importance.")
    else:
        print("  Interpretation: No significant correlation at alpha=0.05.")

    model_per_dim = [m[0] for m in mapping]
    plot_scatter(instability, shap_importance, model_per_dim, args.out_dir, instability_col=args.instability_col)
    plot_by_model(instability, shap_importance, model_per_dim, args.out_dir, instability_col=args.instability_col)
    print(f"[INFO] Plots saved to {args.out_dir}")

    if args.save_csv:
        os.makedirs(args.out_dir, exist_ok=True)
        rows = []
        for g in range(input_dim):
            model_name, dim_id = mapping[g]
            row = {
                "dim_id_global": g,
                "model": model_name,
                "dim_id_local": dim_id,
                args.instability_col: float(instability[g]),
                "shap_importance": float(shap_importance[g]),
            }
            if rank_by_stability[g] is not None:
                row["rank_by_stability"] = rank_by_stability[g]
            rows.append(row)
        out_csv = os.path.join(args.out_dir, "shap_vs_stability_per_dim.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[INFO] Per-dimension CSV saved to {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
