#!/usr/bin/env python3
"""
compare_feature_similarity_batch.py

Compare baseline (normal) H5 features vs augmented H5 features for MANY slides, organized as:

AUG ROOT (from your feature extraction script):
  aug_root/
    none/
    affine_L0/
    affine_L1/
    ...
(each contains one .h5 per slide)

BASELINE DIR (normal features):
  baseline_dir/
    <slide>.h5
    ...

This script will:
- For each augmentation subfolder under --aug_root:
    - Match each augmented .h5 to the baseline .h5 with the same filename stem (or close variants)
    - Align by coords if present in both, else align by index
    - Compute per-tile similarity metrics: cosine, corr, l2, l1
    - Aggregate across ALL matched slides in that augmentation folder
    - Output per-augmentation:
        tile_level.csv, summary.csv, group_diff.csv, consistency_auc.csv, summary.json
- Also writes a top-level master_summary.csv across all augmentation folders.

Notes:
- If labels exist, it will test tumor vs non-tumor differences in "consistency"
- "Consistency-only tile classification" ROC/AUC is also computed when labels exist

Requirements:
  pip install h5py numpy pandas scikit-learn scipy matplotlib
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


# -------------------------
# H5 loading + autodetection
# -------------------------
COORD_CANDIDATES = ["coords", "coord", "coordinates", "patch_coords"]
FEAT_CANDIDATES = ["features", "feats", "embeddings", "embedding", "x", "X", "repr", "reps"]
LABEL_CANDIDATES = ["labels", "label", "y", "Y", "target", "targets", "tumor", "is_tumor", "class"]


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


def find_labels_key(h5: h5py.File, n_expected: Optional[int] = None) -> Optional[str]:
    k = find_dataset_by_name(h5, LABEL_CANDIDATES)
    if k is not None:
        arr = _read_dataset(h5, k).squeeze()
        if arr.ndim == 1 and _is_numeric_array(arr):
            if n_expected is None or arr.shape[0] == n_expected:
                return k
    for key in h5.keys():
        arr = _read_dataset(h5, key).squeeze()
        if arr.ndim != 1 or not _is_numeric_array(arr):
            continue
        if n_expected is not None and arr.shape[0] != n_expected:
            continue
        uniq = np.unique(arr[~np.isnan(arr)]).astype(np.float64)
        if len(uniq) <= 5:
            return key
    return None


def load_h5_triplet(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Dict[str, str]]:
    with h5py.File(path, "r") as h5:
        coords_key = find_coords_key(h5)
        feats_key = find_features_key(h5)
        if feats_key is None:
            raise RuntimeError(f"No 2D numeric feature dataset found in {path}")

        feats = _read_dataset(h5, feats_key).astype(np.float32)

        coords = None
        if coords_key is not None:
            coords = _read_dataset(h5, coords_key).astype(np.int64)

        labels_key = find_labels_key(h5, n_expected=feats.shape[0])
        labels = None
        if labels_key is not None:
            labels = _read_dataset(h5, labels_key).squeeze().astype(np.int64)

        keys_used = {"coords": coords_key or "", "features": feats_key or "", "labels": labels_key or ""}
        return coords, feats, labels, keys_used


# -------------------------
# Alignment
# -------------------------
def coords_to_key(coords: np.ndarray) -> np.ndarray:
    coords = coords.astype(np.int64)
    return (coords[:, 0] << 32) | coords[:, 1]


def align_by_coords(
    coords_a: np.ndarray, feats_a: np.ndarray, labels_a: Optional[np.ndarray],
    coords_b: np.ndarray, feats_b: np.ndarray, labels_b: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    key_a = coords_to_key(coords_a)
    key_b = coords_to_key(coords_b)

    map_a = {k: i for i, k in enumerate(key_a)}
    map_b = {k: i for i, k in enumerate(key_b)}

    common = np.array(sorted(set(map_a.keys()).intersection(set(map_b.keys()))), dtype=np.int64)
    if common.size == 0:
        raise RuntimeError("No overlapping coords between the two files.")

    idx_a = np.array([map_a[k] for k in common], dtype=np.int64)
    idx_b = np.array([map_b[k] for k in common], dtype=np.int64)

    feats_a2 = feats_a[idx_a]
    feats_b2 = feats_b[idx_b]
    la2 = labels_a[idx_a] if labels_a is not None else None
    lb2 = labels_b[idx_b] if labels_b is not None else None

    x = (common >> 32).astype(np.int64)
    y = (common & ((1 << 32) - 1)).astype(np.int64)
    coords_aligned = np.stack([x, y], axis=1)

    return feats_a2, feats_b2, la2, lb2, coords_aligned


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


def summarize_metric(values: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


# -------------------------
# Stats helpers
# -------------------------
def mann_whitney_p(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue)
    except Exception as e:
        raise RuntimeError("scipy required for Mann–Whitney U. Install: pip install scipy") from e


def welch_ttest_p(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import ttest_ind
        res = ttest_ind(x, y, equal_var=False)
        return float(res.pvalue)
    except Exception as e:
        raise RuntimeError("scipy required for Welch t-test. Install: pip install scipy") from e


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(x, y, alternative="two-sided").statistic
        n = x.size
        m = y.size
        return float((2.0 * u) / (n * m) - 1.0)
    except Exception:
        rng = np.random.default_rng(0)
        n_samp = min(20000, x.size * y.size)
        xi = rng.integers(0, x.size, size=n_samp)
        yi = rng.integers(0, y.size, size=n_samp)
        comp = np.sign(x[xi] - y[yi])
        return float(np.mean(comp))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


# -------------------------
# ROC/AUC + Youden J
# -------------------------
def _youden_j_threshold(y_true: np.ndarray, scores: np.ndarray, higher_is_more_tumor: bool) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    scores_for_roc = scores if higher_is_more_tumor else -scores

    auc = float(roc_auc_score(y_true, scores_for_roc))
    fpr, tpr, thresholds = roc_curve(y_true, scores_for_roc)

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr_internal = float(thresholds[best_idx])
    best_J = float(J[best_idx])

    y_pred = (scores_for_roc >= best_thr_internal).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")

    best_thr_original = best_thr_internal if higher_is_more_tumor else -best_thr_internal

    return {
        "auc": auc,
        "best_threshold_original_score_space": float(best_thr_original),
        "best_J": best_J,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
        "score_direction": "higher=more_tumor" if higher_is_more_tumor else "lower=more_tumor",
    }


def compute_auc_for_all_metrics(tile_df: pd.DataFrame, label_col: str, out_csv_path: Path) -> pd.DataFrame:
    y_true = tile_df[label_col].astype(int).to_numpy()

    metric_specs = {
        "cosine": {"col": "cosine", "higher_is_more_consistent": True},
        "corr":   {"col": "corr",   "higher_is_more_consistent": True},
        "l2":     {"col": "l2",     "higher_is_more_consistent": False},
        "l1":     {"col": "l1",     "higher_is_more_consistent": False},
    }

    rows = []
    for metric_name, spec in metric_specs.items():
        col = str(spec["col"])
        higher_is_more_consistent = bool(spec["higher_is_more_consistent"])
        higher_is_more_tumor = higher_is_more_consistent  # assumes "more consistent => more tumor-like"

        scores = tile_df[col].astype(float).to_numpy()
        res = _youden_j_threshold(y_true=y_true, scores=scores, higher_is_more_tumor=higher_is_more_tumor)

        res.update({
            "metric": metric_name,
            "column": col,
            "n": int(len(tile_df)),
            "tumor_n": int(tile_df[label_col].sum()),
            "non_tumor_n": int(len(tile_df) - tile_df[label_col].sum()),
        })
        rows.append(res)

    auc_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    auc_df.to_csv(out_csv_path, index=False)
    return auc_df


# -------------------------
# Matching: baseline file for an augmented file
# -------------------------
def index_h5_by_stem(root: Path, recursive: bool) -> Dict[str, Path]:
    it = root.rglob("*.h5") if recursive else root.glob("*.h5")
    idx: Dict[str, Path] = {}
    for p in it:
        if not p.is_file():
            continue
        idx[p.stem] = p
    return idx


def resolve_baseline_for_aug(aug_h5: Path, baseline_idx: Dict[str, Path]) -> Optional[Path]:
    """
    Try a few matching keys:
      1) exact stem match (recommended: both are <slide_stem>.h5)
      2) if aug stem has suffix like "_features" etc, strip common tails
      3) if baseline stored with full filename match, try by name
    """
    s = aug_h5.stem
    if s in baseline_idx:
        return baseline_idx[s]

    # try stripping common suffixes if present
    suffixes = ["_features", "_feats", "_emb", "_embeddings", "_repr", "_reps", "_patches"]
    for suf in suffixes:
        if s.endswith(suf):
            s2 = s[: -len(suf)]
            if s2 in baseline_idx:
                return baseline_idx[s2]

    # try removing augmentation hints if someone accidentally saved them into filename
    # e.g. <slide>__affine_L0.h5
    for token in ["__", "_"]:
        if token in s:
            head = s.split(token, 1)[0]
            if head in baseline_idx:
                return baseline_idx[head]

    return None


# -------------------------
# Per-pair compute
# -------------------------
def compare_pair(
    baseline_path: Path,
    aug_path: Path,
    label_precedence: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    coords_b, feats_b, labels_b, keys_b = load_h5_triplet(baseline_path)
    coords_a, feats_a, labels_a, keys_a = load_h5_triplet(aug_path)

    if coords_b is not None and coords_a is not None:
        feats_b2, feats_a2, lb2, la2, coords2 = align_by_coords(
            coords_b, feats_b, labels_b, coords_a, feats_a, labels_a
        )
        coords_out = coords2
    else:
        n = min(feats_b.shape[0], feats_a.shape[0])
        feats_b2 = feats_b[:n]
        feats_a2 = feats_a[:n]
        lb2 = labels_b[:n] if labels_b is not None else None
        la2 = labels_a[:n] if labels_a is not None else None
        coords_out = None

    labels = None
    if lb2 is not None and la2 is not None:
        labels = lb2 if label_precedence == "baseline" else la2
    elif lb2 is not None:
        labels = lb2
    elif la2 is not None:
        labels = la2

    # metrics
    cos = cosine_sim(feats_b2, feats_a2)
    l2 = l2_dist(feats_b2, feats_a2)
    l1 = l1_dist(feats_b2, feats_a2)
    corr = corr_sim(feats_b2, feats_a2)

    metrics = {"cosine": cos, "corr": corr, "l2": l2, "l1": l1}
    direction = {
        "cosine": "higher=more_consistent",
        "corr": "higher=more_consistent",
        "l2": "lower=more_consistent",
        "l1": "lower=more_consistent",
    }

    # tile-level df
    df = pd.DataFrame(metrics)
    if coords_out is not None:
        df.insert(0, "x", coords_out[:, 0])
        df.insert(1, "y", coords_out[:, 1])
    if labels is not None:
        df["label"] = labels.astype(int)

    # summary df
    rows = []
    for mname, vals in metrics.items():
        s = summarize_metric(vals)
        rows.append({"group": "all", "metric": mname, "direction": direction[mname], **s})

    if labels is not None:
        tumor_mask = (labels.astype(int) == 1)
        non_mask = ~tumor_mask
        for mname, vals in metrics.items():
            rows.append({"group": "tumor", "metric": mname, "direction": direction[mname], **summarize_metric(vals[tumor_mask])})
            rows.append({"group": "non_tumor", "metric": mname, "direction": direction[mname], **summarize_metric(vals[non_mask])})

    summary_df = pd.DataFrame(rows)

    # tests + AUC (optional; caller may do it on aggregated data, but we return per-pair too)
    group_diff_df = None
    auc_df = None
    if labels is not None:
        tumor_mask = (labels.astype(int) == 1)
        non_mask = ~tumor_mask
        tests_rows = []
        for mname, vals in metrics.items():
            x = vals[tumor_mask]
            y = vals[non_mask]
            if x.size < 5 or y.size < 5:
                tests_rows.append({
                    "metric": mname,
                    "direction": direction[mname],
                    "n_tumor": int(x.size),
                    "n_non_tumor": int(y.size),
                    "p_value": np.nan,
                    "q_value_bh": np.nan,
                    "effect_cliffs_delta": np.nan,
                    "mean_tumor": float(np.mean(x)) if x.size else np.nan,
                    "mean_non_tumor": float(np.mean(y)) if y.size else np.nan,
                    "median_tumor": float(np.median(x)) if x.size else np.nan,
                    "median_non_tumor": float(np.median(y)) if y.size else np.nan,
                    "test": "insufficient_n",
                })
                continue

            # default Mann–Whitney; caller can choose t-test globally
            p = mann_whitney_p(x, y)
            d = cliffs_delta(x, y)
            tests_rows.append({
                "metric": mname,
                "direction": direction[mname],
                "n_tumor": int(x.size),
                "n_non_tumor": int(y.size),
                "p_value": float(p),
                "effect_cliffs_delta": float(d),
                "mean_tumor": float(np.mean(x)),
                "mean_non_tumor": float(np.mean(y)),
                "median_tumor": float(np.median(x)),
                "median_non_tumor": float(np.median(y)),
                "test": "mann_whitney_u",
            })

        pvals = np.array([r["p_value"] for r in tests_rows], dtype=np.float64)
        valid = np.isfinite(pvals)
        qvals = np.full_like(pvals, np.nan, dtype=np.float64)
        if valid.any():
            qvals[valid] = benjamini_hochberg(pvals[valid])
        for r, q in zip(tests_rows, qvals):
            r["q_value_bh"] = float(q) if np.isfinite(q) else np.nan

        group_diff_df = pd.DataFrame(tests_rows)

    meta = {
        "baseline_h5": str(baseline_path),
        "aug_h5": str(aug_path),
        "keys_used": {"baseline": keys_b, "aug": keys_a},
        "aligned_by_coords": bool(coords_b is not None and coords_a is not None),
        "n_tiles_used": int(df.shape[0]),
        "has_labels": bool("label" in df.columns),
    }
    return df, summary_df, group_diff_df, auc_df, meta


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, help="Directory of baseline (normal) .h5 feature files")
    ap.add_argument("--aug_root", required=True, help="Root directory with augmentation subfolders (none/, affine_L0/, ...)")
    ap.add_argument("--out_dir", required=True, help="Output directory (writes per-augmentation results)")
    ap.add_argument("--recursive", action="store_true", help="Recursively search baseline_dir and aug_root for .h5 files")
    ap.add_argument("--label_precedence", choices=["baseline", "aug"], default="baseline",
                    help="Which file's labels to use if both present and disagree (default: baseline)")
    ap.add_argument("--require_labels", action="store_true",
                    help="Error if labels cannot be found for an augmentation folder aggregate.")
    ap.add_argument("--skip_auc", action="store_true",
                    help="Skip ROC/AUC + Youden threshold analysis.")
    ap.add_argument("--use_ttest", action="store_true",
                    help="Use Welch t-test instead of Mann–Whitney U for aggregate group_diff (default: Mann–Whitney).")
    ap.add_argument("--plots_dir", default=None, help="If set, saves histograms per augmentation under this dir.")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    aug_root = Path(args.aug_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not baseline_dir.is_dir():
        raise SystemExit(f"--baseline_dir not found: {baseline_dir}")
    if not aug_root.is_dir():
        raise SystemExit(f"--aug_root not found: {aug_root}")

    baseline_idx = index_h5_by_stem(baseline_dir, recursive=args.recursive)
    if not baseline_idx:
        raise SystemExit(f"No baseline .h5 files found under {baseline_dir}")

    # augmentation folders = immediate subdirs with any h5 files
    aug_folders = sorted([p for p in aug_root.iterdir() if p.is_dir()])
    if not aug_folders:
        raise SystemExit(f"No augmentation subfolders found under {aug_root}")

    master_rows = []

    for aug_folder in aug_folders:
        it = aug_folder.rglob("*.h5") if args.recursive else aug_folder.glob("*.h5")
        aug_files = sorted([p for p in it if p.is_file()])
        if not aug_files:
            continue

        aug_name = aug_folder.name
        out_dir = out_root / aug_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==================== AUGMENTATION: {aug_name} ====================")
        print(f"[INFO] Found {len(aug_files)} augmented files in {aug_folder}")

        tile_dfs = []
        metas = []
        matched = 0
        missing = 0
        failed = 0

        for aug_h5 in aug_files:
            base_h5 = resolve_baseline_for_aug(aug_h5, baseline_idx)
            if base_h5 is None:
                missing += 1
                print(f"[WARN] No baseline match for: {aug_h5.name}")
                continue

            try:
                df, _, _, _, meta = compare_pair(base_h5, aug_h5, label_precedence=args.label_precedence)
                df.insert(0, "slide", aug_h5.stem)
                df.insert(1, "augmentation", aug_name)
                tile_dfs.append(df)
                metas.append(meta)
                matched += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR] Failed pair: baseline={base_h5.name} aug={aug_h5.name} err={e}")

        if matched == 0:
            print(f"[WARN] No matched pairs for {aug_name}, skipping outputs.")
            continue

        tile_df = pd.concat(tile_dfs, axis=0, ignore_index=True)

        # labels check
        has_labels = ("label" in tile_df.columns)
        if args.require_labels and not has_labels:
            raise RuntimeError(f"[{aug_name}] No labels found in aggregated tiles, but --require_labels was set.")

        # write tile-level
        tile_csv = out_dir / "tile_level.csv"
        tile_df.to_csv(tile_csv, index=False)
        print(f"[OK] Wrote {tile_csv}  (rows={len(tile_df)})")

        # aggregate summary
        metrics = ["cosine", "corr", "l2", "l1"]
        direction = {
            "cosine": "higher=more_consistent",
            "corr": "higher=more_consistent",
            "l2": "lower=more_consistent",
            "l1": "lower=more_consistent",
        }

        rows = []
        for m in metrics:
            rows.append({"group": "all", "metric": m, "direction": direction[m], **summarize_metric(tile_df[m].to_numpy())})

        if has_labels:
            tumor_mask = (tile_df["label"].astype(int).to_numpy() == 1)
            non_mask = ~tumor_mask
            for m in metrics:
                rows.append({"group": "tumor", "metric": m, "direction": direction[m], **summarize_metric(tile_df.loc[tumor_mask, m].to_numpy())})
                rows.append({"group": "non_tumor", "metric": m, "direction": direction[m], **summarize_metric(tile_df.loc[non_mask, m].to_numpy())})

        summary_df = pd.DataFrame(rows)
        summary_csv = out_dir / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Wrote {summary_csv}")

        # aggregate group diff tests
        group_diff_df = None
        if has_labels:
            tumor_mask = (tile_df["label"].astype(int).to_numpy() == 1)
            non_mask = ~tumor_mask

            tests_rows = []
            for m in metrics:
                x = tile_df.loc[tumor_mask, m].to_numpy(dtype=float)
                y = tile_df.loc[non_mask, m].to_numpy(dtype=float)

                if x.size < 5 or y.size < 5:
                    tests_rows.append({
                        "metric": m,
                        "direction": direction[m],
                        "n_tumor": int(x.size),
                        "n_non_tumor": int(y.size),
                        "p_value": np.nan,
                        "q_value_bh": np.nan,
                        "effect_cliffs_delta": np.nan,
                        "mean_tumor": float(np.mean(x)) if x.size else np.nan,
                        "mean_non_tumor": float(np.mean(y)) if y.size else np.nan,
                        "median_tumor": float(np.median(x)) if x.size else np.nan,
                        "median_non_tumor": float(np.median(y)) if y.size else np.nan,
                        "test": "insufficient_n",
                    })
                    continue

                if args.use_ttest:
                    p = welch_ttest_p(x, y)
                    test_name = "welch_ttest"
                else:
                    p = mann_whitney_p(x, y)
                    test_name = "mann_whitney_u"

                d = cliffs_delta(x, y)
                tests_rows.append({
                    "metric": m,
                    "direction": direction[m],
                    "n_tumor": int(x.size),
                    "n_non_tumor": int(y.size),
                    "p_value": float(p),
                    "effect_cliffs_delta": float(d),
                    "mean_tumor": float(np.mean(x)),
                    "mean_non_tumor": float(np.mean(y)),
                    "median_tumor": float(np.median(x)),
                    "median_non_tumor": float(np.median(y)),
                    "test": test_name,
                })

            pvals = np.array([r["p_value"] for r in tests_rows], dtype=np.float64)
            valid = np.isfinite(pvals)
            qvals = np.full_like(pvals, np.nan, dtype=np.float64)
            if valid.any():
                qvals[valid] = benjamini_hochberg(pvals[valid])
            for r, q in zip(tests_rows, qvals):
                r["q_value_bh"] = float(q) if np.isfinite(q) else np.nan

            group_diff_df = pd.DataFrame(tests_rows)
            group_diff_csv = out_dir / "group_diff.csv"
            group_diff_df.to_csv(group_diff_csv, index=False)
            print(f"[OK] Wrote {group_diff_csv}")
        else:
            print(f"[WARN] [{aug_name}] No labels found; skipping tumor vs non-tumor significance tests.")

        # aggregate AUC
        auc_df = None
        if has_labels and not args.skip_auc:
            auc_csv = out_dir / "consistency_auc.csv"
            auc_df = compute_auc_for_all_metrics(tile_df, label_col="label", out_csv_path=auc_csv)
            print(f"[OK] Wrote {auc_csv}")
        elif has_labels and args.skip_auc:
            print(f"[INFO] [{aug_name}] --skip_auc set; skipping ROC/AUC.")
        else:
            print(f"[WARN] [{aug_name}] No labels; skipping ROC/AUC.")

        # plots per augmentation (optional)
        if args.plots_dir:
            plots_root = Path(args.plots_dir) / aug_name
            plots_root.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt

            for m in metrics:
                plt.figure()
                plt.hist(tile_df[m].to_numpy(dtype=float), bins=50)
                plt.title(f"{aug_name} :: {m}")
                plt.xlabel(m)
                plt.ylabel("count")
                outp = plots_root / f"{m}.png"
                plt.savefig(outp, dpi=200, bbox_inches="tight")
                plt.close()
            print(f"[OK] Wrote plots to {plots_root}")

        # json per augmentation
        summary_json = out_dir / "summary.json"
        out_json = {
            "augmentation": aug_name,
            "baseline_dir": str(baseline_dir),
            "aug_folder": str(aug_folder),
            "matched_pairs": matched,
            "missing_pairs": missing,
            "failed_pairs": failed,
            "meta_examples_first_3": metas[:3],
            "summary_rows": rows,
            "auc_results": auc_df.to_dict(orient="records") if auc_df is not None else None,
        }
        with open(summary_json, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"[OK] Wrote {summary_json}")

        # master summary row
        master_rows.append({
            "augmentation": aug_name,
            "matched_pairs": matched,
            "missing_pairs": missing,
            "failed_pairs": failed,
            "tile_rows": int(len(tile_df)),
            "has_labels": bool(has_labels),
            "cosine_mean": float(tile_df["cosine"].mean()),
            "corr_mean": float(tile_df["corr"].mean()),
            "l2_mean": float(tile_df["l2"].mean()),
            "l1_mean": float(tile_df["l1"].mean()),
            "auc_best": float(auc_df["auc"].max()) if auc_df is not None and not auc_df.empty else np.nan,
            "auc_best_metric": str(auc_df.sort_values("auc", ascending=False).iloc[0]["metric"]) if auc_df is not None and not auc_df.empty else "",
        })

    # write master summary
    master_df = pd.DataFrame(master_rows).sort_values("augmentation")
    master_csv = out_root / "master_summary.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\n[OK] Wrote {master_csv}")


if __name__ == "__main__":
    main()
