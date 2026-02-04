#!/usr/bin/env python3
"""
resolution_similarity_embeddings.py

Compute similarity metrics across resolutions of FM embedding outputs (same FOV):
- 5x 512px, 10x 1024px, 20x 2048px (trident_processed layout).

Answers:
1. How stable are the features across resolutions (per model)?
2. Are these features genuinely different, and how different?
3. Per-dimension: which embedding dimensions change most across resolutions?

Outputs:
- slide_level_whole_embedding.csv  (per-slide means/stds of similarity metrics)
- summary_whole_embedding.csv     (aggregated over slides, per model and pair)
- per_dimension_stability.csv     (per-dim mean abs diff, std across resolutions)
- summary_report.txt             (short text summary)
- (optional) plots/              (histograms, box plots, per-dim bar)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


# -------------------------
# Resolution config
# -------------------------
RESOLUTIONS = ["5x_512px_0px_overlap", "10x_1024px_0px_overlap", "20x_2048px_0px_overlap"]
RES_TO_SCALE: Dict[str, Tuple[float, float]] = {
    "5x_512px_0px_overlap": (1.0, 1.0),
    "10x_1024px_0px_overlap": (0.5, 0.5),   # divide by 2 to get 5x ref
    "20x_2048px_0px_overlap": (0.25, 0.25), # divide by 4 to get 5x ref
}
DEFAULT_MODELS = ["conch_v15", "musk", "hoptimus1"]
MODEL_TO_FEAT_DIR = {
    "conch_v15": "features_conch_v15",
    "musk": "features_musk",
    "hoptimus1": "features_hoptimus1",
}


# -------------------------
# H5 loading (reuse logic from compare_h5_feature_similarity.py)
# -------------------------
COORD_CANDIDATES = ["coords", "coord", "coordinates", "patch_coords"]
FEAT_CANDIDATES = ["features", "feats", "embeddings", "embedding", "x", "X", "repr", "reps"]


def _is_numeric_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def _read_dataset(h5: h5py.File, key: str) -> np.ndarray:
    return np.array(h5[key])


def find_dataset_by_name(h5: h5py.File, candidates: List[str]) -> Optional[str]:
    keys = set(h5.keys())
    for c in candidates:
        if c in keys:
            return c
    lower_map = {k.lower(): k for k in h5.keys()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_coords_key(h5: h5py.File) -> Optional[str]:
    k = find_dataset_by_name(h5, COORD_CANDIDATES)
    if k is not None:
        arr = _read_dataset(h5, k)
        if arr.ndim == 2 and arr.shape[1] == 2 and _is_numeric_array(arr):
            return k
    for key in h5.keys():
        arr = _read_dataset(h5, key)
        if arr.ndim == 2 and arr.shape[1] == 2 and _is_numeric_array(arr):
            return key
    return None


def find_features_key(h5: h5py.File) -> Optional[str]:
    k = find_dataset_by_name(h5, FEAT_CANDIDATES)
    if k is not None:
        arr = _read_dataset(h5, k)
        if arr.ndim == 2 and _is_numeric_array(arr):
            return k
    best_key = None
    best_size = -1
    for key in h5.keys():
        arr = _read_dataset(h5, key)
        if arr.ndim == 2 and _is_numeric_array(arr):
            size = arr.shape[0] * arr.shape[1]
            if size > best_size:
                best_size = size
                best_key = key
    return best_key


def load_h5_coords_features(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Load coords (N,2) and features (N,D) from an H5. Coords may be None."""
    with h5py.File(path, "r") as f:
        coords_key = find_coords_key(f)
        feats_key = find_features_key(f)
        if feats_key is None:
            raise RuntimeError(f"No 2D numeric feature dataset found in {path}")
        feats = _read_dataset(f, feats_key).astype(np.float32)
        coords = None
        if coords_key is not None:
            coords = _read_dataset(f, coords_key).astype(np.int64)
    return coords, feats


# -------------------------
# Coord normalization and alignment
# -------------------------
def coords_to_ref(coords: np.ndarray, scale_x: float, scale_y: float, grid_step: Optional[int]) -> np.ndarray:
    """Convert at-mag coords to 5x reference. Optionally quantize to grid_step (e.g. 512)."""
    ref = coords.astype(np.float64)
    ref[:, 0] *= scale_x
    ref[:, 1] *= scale_y
    if grid_step is not None and grid_step > 0:
        ref = (np.round(ref / grid_step) * grid_step).astype(np.int64)
    else:
        ref = ref.astype(np.int64)
    return ref


def ref_to_key(ref: np.ndarray) -> np.ndarray:
    """Pack (x_ref, y_ref) into a single int64 key for matching."""
    return (ref[:, 0].astype(np.int64) << 32) | (ref[:, 1].astype(np.int64) & ((1 << 32) - 1))


def align_three_resolutions(
    coords_5x: np.ndarray,
    feats_5x: np.ndarray,
    coords_10x: np.ndarray,
    feats_10x: np.ndarray,
    coords_20x: np.ndarray,
    feats_20x: np.ndarray,
    grid_step: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize coords to 5x ref, intersect by ref key, return aligned feature arrays.
    Returns (feats_5x_aligned, feats_10x_aligned, feats_20x_aligned) each (N, D).
    """
    s5 = RES_TO_SCALE["5x_512px_0px_overlap"]
    s10 = RES_TO_SCALE["10x_1024px_0px_overlap"]
    s20 = RES_TO_SCALE["20x_2048px_0px_overlap"]

    ref_5 = coords_to_ref(coords_5x, s5[0], s5[1], grid_step)
    ref_10 = coords_to_ref(coords_10x, s10[0], s10[1], grid_step)
    ref_20 = coords_to_ref(coords_20x, s20[0], s20[1], grid_step)

    key_5 = ref_to_key(ref_5)
    key_10 = ref_to_key(ref_10)
    key_20 = ref_to_key(ref_20)

    set_5 = set(key_5)
    set_10 = set(key_10)
    set_20 = set(key_20)
    common = np.array(sorted(set_5 & set_10 & set_20), dtype=np.int64)
    if common.size == 0:
        raise RuntimeError("No overlapping ref coords across the three resolutions.")

    map_5 = {k: i for i, k in enumerate(key_5)}
    map_10 = {k: i for i, k in enumerate(key_10)}
    map_20 = {k: i for i, k in enumerate(key_20)}

    idx_5 = np.array([map_5[k] for k in common], dtype=np.int64)
    idx_10 = np.array([map_10[k] for k in common], dtype=np.int64)
    idx_20 = np.array([map_20[k] for k in common], dtype=np.int64)

    return feats_5x[idx_5], feats_10x[idx_10], feats_20x[idx_20]


# -------------------------
# Similarity metrics
# -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    an = np.linalg.norm(a, axis=1) + eps
    bn = np.linalg.norm(b, axis=1) + eps
    return (a * b).sum(axis=1) / (an * bn)


def l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt((d * d).sum(axis=1))


def l1_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=1)


def corr_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = (a0 * b0).sum(axis=1)
    den = (np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)) + eps
    return num / den


# -------------------------
# Whole-embedding and per-dimension stats
# -------------------------
def compute_whole_embedding_metrics(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Returns dict of metric_name -> {mean, std, median, n}."""
    cos = cosine_sim(feats_a, feats_b)
    l2 = l2_dist(feats_a, feats_b)
    l1 = l1_dist(feats_a, feats_b)
    corr = corr_sim(feats_a, feats_b)
    out = {}
    for name, vals in [("cosine", cos), ("l2", l2), ("l1", l1), ("corr", corr)]:
        out[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "n": int(vals.size),
        }
    return out


def compute_per_dimension_stability(
    feats_5x: np.ndarray,
    feats_10x: np.ndarray,
    feats_20x: np.ndarray,
) -> np.ndarray:
    """
    Returns array of shape (D,) with mean absolute difference across the three
    resolutions per dimension (averaged over tiles). Lower = more stable.
    """
    d = feats_5x.shape[1]
    mad_5_10 = np.abs(feats_5x - feats_10x).mean(axis=0)
    mad_5_20 = np.abs(feats_5x - feats_20x).mean(axis=0)
    mad_10_20 = np.abs(feats_10x - feats_20x).mean(axis=0)
    std_3 = np.std(np.stack([feats_5x, feats_10x, feats_20x], axis=1), axis=1).mean(axis=0)
    return np.stack([mad_5_10, mad_5_20, mad_10_20, std_3], axis=0)


# -------------------------
# Slide iteration and processing
# -------------------------
def list_slides_in_dir(d: Path) -> List[str]:
    stems = []
    for f in d.glob("*.h5"):
        if f.suffix.lower() == ".h5" and f.is_file():
            stems.append(f.stem)
    return sorted(set(stems))


def process_slide(
    root: Path,
    model: str,
    slide_stem: str,
    grid_step: Optional[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Load H5 for slide at all three resolutions (same model), align, return
    (feats_5x, feats_10x, feats_20x) aligned and n_tiles. Returns None if skip.
    """
    feat_dir_name = MODEL_TO_FEAT_DIR[model]
    paths = {}
    for res in RESOLUTIONS:
        p = root / res / feat_dir_name / f"{slide_stem}.h5"
        if not p.is_file():
            return None
        paths[res] = p

    coords_5x, feats_5x = load_h5_coords_features(paths["5x_512px_0px_overlap"])
    coords_10x, feats_10x = load_h5_coords_features(paths["10x_1024px_0px_overlap"])
    coords_20x, feats_20x = load_h5_coords_features(paths["20x_2048px_0px_overlap"])

    if coords_5x is None or coords_10x is None or coords_20x is None:
        return None

    if feats_5x.shape[1] != feats_10x.shape[1] or feats_5x.shape[1] != feats_20x.shape[1]:
        return None

    try:
        a5, a10, a20 = align_three_resolutions(
            coords_5x, feats_5x,
            coords_10x, feats_10x,
            coords_20x, feats_20x,
            grid_step,
        )
    except RuntimeError:
        return None

    n = a5.shape[0]
    return (a5, a10, a20, n)


# -------------------------
# Main aggregation and writing
# -------------------------
def run(
    root: Path,
    models: List[str],
    out_dir: Path,
    grid_step: Optional[int] = 512,
    plots: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_level_rows: List[Dict] = []
    summary_whole_rows: List[Dict] = []
    per_dim_rows: List[Dict] = []

    # Collect slide stems from first resolution/model
    feat_dir_name = MODEL_TO_FEAT_DIR[models[0]]
    slide_stems = list_slides_in_dir(root / RESOLUTIONS[0] / feat_dir_name)
    for res in RESOLUTIONS[1:]:
        stems_r = set(list_slides_in_dir(root / res / feat_dir_name))
        slide_stems = [s for s in slide_stems if s in stems_r]
    slide_stems = sorted(slide_stems)

    for model in models:
        feat_dir_name = MODEL_TO_FEAT_DIR[model]
        # Slides present in all three resolutions for this model
        stems_5 = set(list_slides_in_dir(root / RESOLUTIONS[0] / feat_dir_name))
        stems_10 = set(list_slides_in_dir(root / RESOLUTIONS[1] / feat_dir_name))
        stems_20 = set(list_slides_in_dir(root / RESOLUTIONS[2] / feat_dir_name))
        common_stems = sorted(stems_5 & stems_10 & stems_20)

        pair_metrics_slide: Dict[str, List[float]] = {
            "5x_10x_cosine": [], "5x_10x_l2": [], "5x_10x_l1": [], "5x_10x_corr": [],
            "5x_20x_cosine": [], "5x_20x_l2": [], "5x_20x_l1": [], "5x_20x_corr": [],
            "10x_20x_cosine": [], "10x_20x_l2": [], "10x_20x_l1": [], "10x_20x_corr": [],
        }
        all_feats_5x: List[np.ndarray] = []
        all_feats_10x: List[np.ndarray] = []
        all_feats_20x: List[np.ndarray] = []

        for slide_stem in common_stems:
            result = process_slide(root, model, slide_stem, grid_step)
            if result is None:
                continue
            feats_5x, feats_10x, feats_20x, n_tiles = result

            # Whole-embedding per pair
            for pair_name, (fa, fb) in [
                ("5x_10x", (feats_5x, feats_10x)),
                ("5x_20x", (feats_5x, feats_20x)),
                ("10x_20x", (feats_10x, feats_20x)),
            ]:
                m = compute_whole_embedding_metrics(fa, fb)
                for metric in ["cosine", "l2", "l1", "corr"]:
                    key = f"{pair_name}_{metric}"
                    slide_level_rows.append({
                        "slide": slide_stem,
                        "model": model,
                        "pair": pair_name,
                        "metric": metric,
                        "mean": m[metric]["mean"],
                        "std": m[metric]["std"],
                        "n_tiles": n_tiles,
                    })
                    pair_metrics_slide[key].append(m[metric]["mean"])

            all_feats_5x.append(feats_5x)
            all_feats_10x.append(feats_10x)
            all_feats_20x.append(feats_20x)

        # Summary whole-embedding (aggregate over slides for this model)
        for key, vals in pair_metrics_slide.items():
            if not vals:
                continue
            pair_name = key.rsplit("_", 1)[0]
            metric = key.split("_")[-1]
            summary_whole_rows.append({
                "model": model,
                "pair": pair_name,
                "metric": metric,
                "mean_of_means": float(np.mean(vals)),
                "std_across_slides": float(np.std(vals)),
                "median_of_means": float(np.median(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
                "n_slides": len(vals),
            })

        # Per-dimension (aggregate over all tiles from all slides for this model)
        if all_feats_5x:
            F5 = np.vstack(all_feats_5x)
            F10 = np.vstack(all_feats_10x)
            F20 = np.vstack(all_feats_20x)
            stab = compute_per_dimension_stability(F5, F10, F20)
            D = F5.shape[1]
            mean_mad = (stab[0] + stab[1] + stab[2]) / 3.0
            rank = np.argsort(mean_mad)
            for d in range(D):
                per_dim_rows.append({
                    "dim_id": d,
                    "model": model,
                    "mean_abs_diff_5x_10x": float(stab[0, d]),
                    "mean_abs_diff_5x_20x": float(stab[1, d]),
                    "mean_abs_diff_10x_20x": float(stab[2, d]),
                    "std_across_resolutions": float(stab[3, d]),
                    "mean_mad": float(mean_mad[d]),
                    "rank_by_stability": int(np.where(rank == d)[0][0]) + 1,
                })

    # Write CSVs
    if slide_level_rows:
        pd.DataFrame(slide_level_rows).to_csv(out_dir / "slide_level_whole_embedding.csv", index=False)
        print(f"[OK] Wrote {out_dir / 'slide_level_whole_embedding.csv'}")
    if summary_whole_rows:
        pd.DataFrame(summary_whole_rows).to_csv(out_dir / "summary_whole_embedding.csv", index=False)
        print(f"[OK] Wrote {out_dir / 'summary_whole_embedding.csv'}")
    if per_dim_rows:
        pd.DataFrame(per_dim_rows).to_csv(out_dir / "per_dimension_stability.csv", index=False)
        print(f"[OK] Wrote {out_dir / 'per_dimension_stability.csv'}")

    # Summary report
    report_lines = [
        "=== Resolution similarity report (same FOV: 5x/512, 10x/1024, 20x/2048) ===",
        "",
        "1. How stable are the features across resolutions (per model)?",
        "   See summary_whole_embedding.csv: mean_of_means and std_across_slides per (model, pair, metric).",
        "   Higher cosine/corr = more stable; lower l2/l1 = more stable.",
        "",
        "2. Are these features genuinely different, and how different?",
        "   Same table: p05/p95 and std_across_slides indicate spread. Compare pairs (e.g. 5x vs 20x) across models.",
        "",
        "3. Per-dimension stability",
        "   See per_dimension_stability.csv: rank_by_stability (1 = most stable), mean_abs_diff_* and std_across_resolutions.",
        "",
    ]
    if summary_whole_rows:
        df = pd.DataFrame(summary_whole_rows)
        report_lines.append("Summary (cosine mean_of_means by model and pair):")
        for model in df["model"].unique():
            sub = df[(df["model"] == model) & (df["metric"] == "cosine")]
            for _, r in sub.iterrows():
                report_lines.append(f"  {model} {r['pair']}: {r['mean_of_means']:.4f} (n_slides={r['n_slides']})")
        report_lines.append("")
        report_lines.append("Summary (per-dimension: top 5 most stable and top 5 least stable, first model):")
        if per_dim_rows:
            pdf = pd.DataFrame(per_dim_rows)
            m0 = pdf["model"].iloc[0]
            sub = pdf[pdf["model"] == m0].sort_values("rank_by_stability")
            report_lines.append(f"  Model: {m0}")
            for _, r in sub.head(5).iterrows():
                report_lines.append(f"    dim {r['dim_id']} rank={r['rank_by_stability']} mean_mad={r['mean_mad']:.4f}")
            for _, r in sub.tail(5).iterrows():
                report_lines.append(f"    dim {r['dim_id']} rank={r['rank_by_stability']} mean_mad={r['mean_mad']:.4f}")
    report_path = out_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] Wrote {report_path}")

    if plots and (slide_level_rows or summary_whole_rows):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            if slide_level_rows:
                sdf = pd.DataFrame(slide_level_rows)
                sdf["model_pair"] = sdf["model"] + " " + sdf["pair"]
                for metric in ["cosine", "l2"]:
                    sub = sdf[sdf["metric"] == metric]
                    if sub.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sub.boxplot(column="mean", by="model_pair", ax=ax)
                    plt.suptitle(f"Whole-embedding {metric} by model and resolution pair")
                    plt.xlabel("model / pair")
                    plt.ylabel(metric)
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"whole_embedding_{metric}.png", dpi=150, bbox_inches="tight")
                    plt.close()
            if per_dim_rows:
                pdf = pd.DataFrame(per_dim_rows)
                for model in pdf["model"].unique():
                    sub = pdf[pdf["model"] == model].sort_values("mean_mad")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    n_show = min(50, len(sub))
                    ax.bar(range(n_show), sub["mean_mad"].values[:n_show], color="steelblue", alpha=0.8)
                    ax.set_xlabel("Dimension (sorted by stability, first 50)")
                    ax.set_ylabel("Mean absolute diff (avg over pairs)")
                    ax.set_title(f"Per-dimension stability ({model})")
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"per_dim_stability_{model}.png", dpi=150, bbox_inches="tight")
                    plt.close()
            print(f"[OK] Wrote plots to {plots_dir}")
        except Exception as e:
            print(f"[WARN] Plots skipped: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute similarity metrics across 5x/10x/20x FM embeddings (same FOV)."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/scratch/pioneer/users/sxk2517/trident_processed"),
        help="Root dir containing 5x_512px_0px_overlap, 10x_1024px_0px_overlap, 20x_2048px_0px_overlap",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model names (default: {DEFAULT_MODELS})",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/scratch/pioneer/users/sxk2517/resolution_similarity_out"),
        help="Output directory for CSVs and report",
    )
    ap.add_argument(
        "--ref_grid_step",
        type=int,
        default=512,
        help="Quantize ref coords to this step in 5x space (0 = no quantize)",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Write optional histogram/box plots",
    )
    args = ap.parse_args()
    grid_step = args.ref_grid_step if args.ref_grid_step > 0 else None
    run(
        root=args.root,
        models=args.models,
        out_dir=args.out_dir,
        grid_step=grid_step,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
