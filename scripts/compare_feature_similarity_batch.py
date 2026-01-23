#!/usr/bin/env python3
"""
compare_feature_similarity_batch.py

Compare baseline (normal) H5 features vs augmented H5 features for many slides, organized as:

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

For each augmentation subfolder under --aug_root, this script:
- Matches augmented .h5 files to baseline .h5 files
- Aligns by coords if present in both files (preferred), else aligns by index
- Computes per-tile similarity metrics: cosine, corr, l2, l1
- Aggregates across all matched slides in that augmentation folder
- Writes per-augmentation outputs:
    tile_level.csv, summary.csv, group_diff.csv, consistency_auc.csv, summary.json
- Writes a top-level master_summary.csv across all augmentation folders

Labels:
- If a label vector exists (0/1) in either baseline or aug file, it is propagated per tile.
- Tumor vs non-tumor comparisons are computed only on rows with valid labels in {0,1}.

Requirements:
  pip install h5py numpy pandas scikit-learn scipy matplotlib
"""

import argparse
import json
import traceback
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


def _mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")

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

    # fallback: pick largest 2D numeric dataset
    best_key = None
    best_size = -1
    for key in h5.keys():
        arr = _read_dataset(h5, key)
        if arr.ndim == 2 and _is_numeric_array(arr):
            size = int(arr.shape[0]) * int(arr.shape[1])
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

    # fallback heuristic: 1D numeric with small number of unique values
    for key in h5.keys():
        arr = _read_dataset(h5, key).squeeze()
        if arr.ndim != 1 or not _is_numeric_array(arr):
            continue
        if n_expected is not None and arr.shape[0] != n_expected:
            continue
        # allow NaNs by filtering them
        arr_f = arr.astype(np.float64)
        arr_f = arr_f[np.isfinite(arr_f)]
        if arr_f.size == 0:
            continue
        uniq = np.unique(arr_f)
        if len(uniq) <= 10:
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
            # keep as float to preserve NaNs if present
            labels = _read_dataset(h5, labels_key).squeeze()
            labels = labels.astype(np.float32) if _is_numeric_array(labels) else None

        keys_used = {"coords": coords_key or "", "features": feats_key or "", "labels": labels_key or ""}
        return coords, feats, labels, keys_used


# -------------------------
# Alignment (coords preferred)
# -------------------------
def coords_to_key(coords: np.ndarray) -> np.ndarray:
    # Pack (x,y) into a uint64 key: (x<<32)|y
    coords_u = coords.astype(np.uint64, copy=False)
    return (coords_u[:, 0] << np.uint64(32)) | coords_u[:, 1]


def align_by_coords(
    coords_b: np.ndarray, feats_b: np.ndarray, labels_b: Optional[np.ndarray],
    coords_a: np.ndarray, feats_a: np.ndarray, labels_a: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    key_b = coords_to_key(coords_b)
    key_a = coords_to_key(coords_a)

    # Check for duplicates (can break deterministic alignment)
    if np.unique(key_b).shape[0] != key_b.shape[0]:
        raise RuntimeError("Baseline coords contain duplicates; cannot align safely by coords.")
    if np.unique(key_a).shape[0] != key_a.shape[0]:
        raise RuntimeError("Aug coords contain duplicates; cannot align safely by coords.")

    sb = np.argsort(key_b)
    sa = np.argsort(key_a)
    key_b_sorted = key_b[sb]
    key_a_sorted = key_a[sa]

    common, ib, ia = np.intersect1d(key_b_sorted, key_a_sorted, assume_unique=True, return_indices=True)
    if common.size == 0:
        raise RuntimeError("No overlapping coords between the two files.")

    idx_b = sb[ib]
    idx_a = sa[ia]

    feats_b2 = feats_b[idx_b]
    feats_a2 = feats_a[idx_a]
    lb2 = labels_b[idx_b] if labels_b is not None else None
    la2 = labels_a[idx_a] if labels_a is not None else None

    x = (common >> np.uint64(32)).astype(np.int64)
    y = (common & np.uint64((1 << 32) - 1)).astype(np.int64)
    coords_aligned = np.stack([x, y], axis=1)

    return feats_b2, feats_a2, lb2, la2, coords_aligned


# -------------------------
# Similarity metrics
# -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    an = np.linalg.norm(a, axis=1) + eps
    bn = np.linalg.norm(b, axis=1) + eps
    return (a * b).sum(axis=1) / (an * bn)


def l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt((d * d).sum(axis=1))


def l1_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=1)


def corr_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = (a0 * b0).sum(axis=1)
    den = (np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)) + eps
    return num / den


def summarize_metric(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
            "p05": float("nan"), "p25": float("nan"), "p75": float("nan"), "p95": float("nan"),
        }
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
        raise RuntimeError("scipy required for Mann-Whitney U. Install: pip install scipy") from e


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
        # fallback: sampling approximation
        rng = np.random.default_rng(0)
        n_samp = min(20000, x.size * y.size)
        if n_samp <= 0:
            return float("nan")
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
def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _youden_j_threshold(y_true: np.ndarray, scores: np.ndarray, higher_is_more_tumor: bool) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    scores_for_roc = scores if higher_is_more_tumor else -scores

    auc = _safe_roc_auc(y_true, scores_for_roc)
    if not np.isfinite(auc):
        return {
            "auc": float("nan"),
            "best_threshold_original_score_space": float("nan"),
            "best_J": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "accuracy": float("nan"),
            "score_direction": "higher=more_tumor" if higher_is_more_tumor else "lower=more_tumor",
        }

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
    # Only evaluate on valid binary labels
    df = tile_df.copy()
    df = df[df[label_col].isin([0, 1])].copy()
    if df.empty:
        auc_df = pd.DataFrame([])
        auc_df.to_csv(out_csv_path, index=False)
        return auc_df

    y_true = df[label_col].astype(int).to_numpy()

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
        # Interpreting tumor direction as: tumor has "more consistent" tiles
        higher_is_more_tumor = higher_is_more_consistent

        scores = df[col].astype(float).to_numpy()
        res = _youden_j_threshold(y_true=y_true, scores=scores, higher_is_more_tumor=higher_is_more_tumor)

        res.update({
            "metric": metric_name,
            "column": col,
            "n": int(len(df)),
            "tumor_n": int((df[label_col] == 1).sum()),
            "non_tumor_n": int((df[label_col] == 0).sum()),
        })
        rows.append(res)

    auc_df = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    auc_df.to_csv(out_csv_path, index=False)
    return auc_df


# -------------------------
# Matching: baseline file for an augmented file
# -------------------------
def index_h5_by_stem(root: Path, recursive: bool) -> Dict[str, Path]:
    it = root.rglob("*.h5") if recursive else root.glob("*.h5")
    idx: Dict[str, Path] = {}
    for p in it:
        if p.is_file():
            idx[p.stem] = p
    return idx


def resolve_baseline_for_aug(aug_h5: Path, baseline_idx: Dict[str, Path]) -> Optional[Path]:
    """
    Try a few matching keys:
      1) exact stem match
      2) strip common suffixes
      3) strip trailing augmentation hints like <slide>__affine_L0
    """
    s = aug_h5.stem
    if s in baseline_idx:
        return baseline_idx[s]

    suffixes = ["_features", "_feats", "_emb", "_embeddings", "_repr", "_reps", "_patches"]
    for suf in suffixes:
        if s.endswith(suf):
            s2 = s[: -len(suf)]
            if s2 in baseline_idx:
                return baseline_idx[s2]

    for token in ["__", "_"]:
        if token in s:
            head = s.split(token, 1)[0]
            if head in baseline_idx:
                return baseline_idx[head]

    return None


# -------------------------
# Per-pair compute
# -------------------------
def _merge_labels(lb2: Optional[np.ndarray], la2: Optional[np.ndarray], precedence: str) -> Optional[np.ndarray]:
    if lb2 is None and la2 is None:
        return None
    if lb2 is None:
        return la2
    if la2 is None:
        return lb2

    if precedence == "baseline":
        chosen = lb2
        other = la2
    else:
        chosen = la2
        other = lb2

    # Fill NaNs in chosen from other, so we keep as many labels as possible
    out = chosen.astype(np.float32, copy=True)
    if out.ndim != 1:
        out = out.squeeze()
    other = other.astype(np.float32, copy=False).squeeze()

    if out.shape != other.shape:
        return out

    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = other[mask]
    return out


def compare_pair(
    baseline_path: Path,
    aug_path: Path,
    label_precedence: str,
) -> Tuple[pd.DataFrame, Dict]:
    coords_b, feats_b, labels_b, keys_b = load_h5_triplet(baseline_path)
    coords_a, feats_a, labels_a, keys_a = load_h5_triplet(aug_path)

    if feats_b.shape[1] != feats_a.shape[1]:
        raise RuntimeError(f"Feature dim mismatch: baseline D={feats_b.shape[1]} vs aug D={feats_a.shape[1]}")

    if coords_b is not None and coords_a is not None:
        feats_b2, feats_a2, lb2, la2, coords2 = align_by_coords(
            coords_b, feats_b, labels_b, coords_a, feats_a, labels_a
        )
        coords_out = coords2
        aligned_by_coords = True
    else:
        n = min(feats_b.shape[0], feats_a.shape[0])
        feats_b2 = feats_b[:n]
        feats_a2 = feats_a[:n]
        lb2 = labels_b[:n] if labels_b is not None else None
        la2 = labels_a[:n] if labels_a is not None else None
        coords_out = None
        aligned_by_coords = False

    labels = _merge_labels(lb2, la2, precedence=label_precedence)

    # metrics
    cos = cosine_sim(feats_b2, feats_a2)
    l2 = l2_dist(feats_b2, feats_a2)
    l1 = l1_dist(feats_b2, feats_a2)
    corr = corr_sim(feats_b2, feats_a2)

    df = pd.DataFrame({"cosine": cos, "corr": corr, "l2": l2, "l1": l1})
    if coords_out is not None:
        df.insert(0, "x", coords_out[:, 0])
        df.insert(1, "y", coords_out[:, 1])

    if labels is not None:
        df["label"] = labels

    meta = {
        "baseline_h5": str(baseline_path),
        "aug_h5": str(aug_path),
        "keys_used": {"baseline": keys_b, "aug": keys_a},
        "aligned_by_coords": bool(aligned_by_coords),
        "n_tiles_used": int(df.shape[0]),
        "has_labels": bool("label" in df.columns),
    }
    return df, meta


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
                    help="Use Welch t-test instead of Mann-Whitney U for aggregate group_diff (default: Mann-Whitney).")
    ap.add_argument("--plots_dir", default=None, help="If set, saves histograms per augmentation under this dir.")
    ap.add_argument("--debug", action="store_true", help="Print tracebacks for failed pairs.")
    ap.add_argument("--fail_fast", action="store_true", help="Stop immediately on the first failed pair.")
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

    aug_folders = sorted([p for p in aug_root.iterdir() if p.is_dir()])
    if not aug_folders:
        raise SystemExit(f"No augmentation subfolders found under {aug_root}")

    metrics = ["cosine", "corr", "l2", "l1"]
    direction = {
        "cosine": "higher=more_consistent",
        "corr": "higher=more_consistent",
        "l2": "lower=more_consistent",
        "l1": "lower=more_consistent",
    }

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
                df, meta = compare_pair(base_h5, aug_h5, label_precedence=args.label_precedence)
                df.insert(0, "slide", aug_h5.stem)
                df.insert(1, "augmentation", aug_name)
                tile_dfs.append(df)
                metas.append(meta)
                matched += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR] Failed pair in aug={aug_name} slide={aug_h5.stem}")
                print(f"        baseline={base_h5}")
                print(f"        augfile={aug_h5}")
                print(f"        err={e}")
                if args.debug:
                    print(traceback.format_exc())
                if args.fail_fast:
                    raise

        if matched == 0:
            print(f"[WARN] No matched pairs for {aug_name}, skipping outputs.")
            continue

        tile_df = pd.concat(tile_dfs, axis=0, ignore_index=True)

        # Normalize labels: keep only {0,1}, set everything else to NaN
        if "label" in tile_df.columns:
            tile_df["label"] = pd.to_numeric(tile_df["label"], errors="coerce")
            tile_df.loc[~tile_df["label"].isin([0, 1]), "label"] = np.nan

            bad = int(tile_df["label"].isna().sum())
            if bad > 0:
                bad_slides = tile_df.loc[tile_df["label"].isna(), "slide"].unique()
                print(f"[WARN] {bad} tiles have missing/invalid labels (not 0/1).")
                print(f"[WARN] Affected slides (first 10): {bad_slides[:10]}")

        has_labels = ("label" in tile_df.columns) and (tile_df["label"].isin([0, 1]).any())
        if args.require_labels and not has_labels:
            raise RuntimeError(f"[{aug_name}] No valid labels (0/1) found, but --require_labels was set.")

        # Write tile-level
        tile_csv = out_dir / "tile_level.csv"
        tile_df.to_csv(tile_csv, index=False)
        print(f"[OK] Wrote {tile_csv} (rows={len(tile_df)})")

        # Summary
        summary_rows = []
        for m in metrics:
            summary_rows.append({
                "group": "all", "metric": m, "direction": direction[m],
                **summarize_metric(tile_df[m].to_numpy(dtype=float))
            })

        if has_labels:
            lbl = tile_df["label"]
            label_valid = lbl.isin([0, 1])
            lbl_int = lbl[label_valid].astype(int).to_numpy()
            tumor_mask = (lbl_int == 1)

            for m in metrics:
                vals = tile_df.loc[label_valid, m].to_numpy(dtype=float)
                x = vals[tumor_mask]
                y = vals[~tumor_mask]
                summary_rows.append({"group": "tumor", "metric": m, "direction": direction[m], **summarize_metric(x)})
                summary_rows.append({"group": "non_tumor", "metric": m, "direction": direction[m], **summarize_metric(y)})

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Wrote {summary_csv}")

        # Group diff tests
        group_diff_df = None
        if has_labels:
            lbl = tile_df["label"]
            label_valid = lbl.isin([0, 1])
            lbl_int = lbl[label_valid].astype(int).to_numpy()
            tumor_mask = (lbl_int == 1)

            tests_rows = []
            for m in metrics:
                vals = tile_df.loc[label_valid, m].to_numpy(dtype=float)
                x = vals[tumor_mask]
                y = vals[~tumor_mask]

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
            finite = np.isfinite(pvals)
            qvals = np.full_like(pvals, np.nan, dtype=np.float64)
            if finite.any():
                qvals[finite] = benjamini_hochberg(pvals[finite])
            for r, q in zip(tests_rows, qvals):
                r["q_value_bh"] = float(q) if np.isfinite(q) else np.nan

            group_diff_df = pd.DataFrame(tests_rows)
            group_diff_csv = out_dir / "group_diff.csv"
            group_diff_df.to_csv(group_diff_csv, index=False)
            print(f"[OK] Wrote {group_diff_csv}")
        else:
            print(f"[WARN] [{aug_name}] No valid labels; skipping tumor vs non-tumor significance tests.")

        # AUC
        auc_df = None
        if has_labels and not args.skip_auc:
            auc_csv = out_dir / "consistency_auc.csv"
            auc_df = compute_auc_for_all_metrics(tile_df, label_col="label", out_csv_path=auc_csv)
            print(f"[OK] Wrote {auc_csv}")
        elif has_labels and args.skip_auc:
            print(f"[INFO] [{aug_name}] --skip_auc set; skipping ROC/AUC.")
        else:
            print(f"[WARN] [{aug_name}] No valid labels; skipping ROC/AUC.")

        # Plots (optional)
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

        # JSON summary
        summary_json = out_dir / "summary.json"
        out_json = {
            "augmentation": aug_name,
            "baseline_dir": str(baseline_dir),
            "aug_folder": str(aug_folder),
            "matched_pairs": matched,
            "missing_pairs": missing,
            "failed_pairs": failed,
            "meta_examples_first_3": metas[:3],
            "summary_rows": summary_rows,
            "auc_results": auc_df.to_dict(orient="records") if auc_df is not None else None,
        }
        with open(summary_json, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"[OK] Wrote {summary_json}")

        # Master row
                # -------------------------
        # Master row (ALL + tumor/non-tumor if labels exist)
        # -------------------------
        master_row = {
            "augmentation": aug_name,
            "matched_pairs": matched,
            "missing_pairs": missing,
            "failed_pairs": failed,
            "tile_rows": int(len(tile_df)),
            "has_labels": bool(has_labels),

            # ALL tiles
            "cosine_mean_all": float(tile_df["cosine"].mean()),
            "corr_mean_all": float(tile_df["corr"].mean()),
            "l2_mean_all": float(tile_df["l2"].mean()),
            "l1_mean_all": float(tile_df["l1"].mean()),

            # AUC summary (still computed on labeled subset)
            "auc_best": float(auc_df["auc"].max()) if auc_df is not None and not auc_df.empty else np.nan,
            "auc_best_metric": str(auc_df.sort_values("auc", ascending=False).iloc[0]["metric"])
                               if auc_df is not None and not auc_df.empty else "",
        }

        if has_labels:
            labeled = tile_df[tile_df["label"].isin([0, 1])].copy()
            tumor_df = labeled[labeled["label"] == 1]
            non_df = labeled[labeled["label"] == 0]

            master_row.update({
                "tile_rows_labeled": int(len(labeled)),
                "tile_rows_tumor": int(len(tumor_df)),
                "tile_rows_non_tumor": int(len(non_df)),

                "cosine_mean_tumor": float(tumor_df["cosine"].mean()) if len(tumor_df) else np.nan,
                "corr_mean_tumor": float(tumor_df["corr"].mean()) if len(tumor_df) else np.nan,
                "l2_mean_tumor": float(tumor_df["l2"].mean()) if len(tumor_df) else np.nan,
                "l1_mean_tumor": float(tumor_df["l1"].mean()) if len(tumor_df) else np.nan,

                "cosine_mean_non_tumor": float(non_df["cosine"].mean()) if len(non_df) else np.nan,
                "corr_mean_non_tumor": float(non_df["corr"].mean()) if len(non_df) else np.nan,
                "l2_mean_non_tumor": float(non_df["l2"].mean()) if len(non_df) else np.nan,
                "l1_mean_non_tumor": float(non_df["l1"].mean()) if len(non_df) else np.nan,
            })
        else:
            # keep columns present even when labels missing (so CSV schema is consistent)
            master_row.update({
                "tile_rows_labeled": 0,
                "tile_rows_tumor": 0,
                "tile_rows_non_tumor": 0,
                "cosine_mean_tumor": np.nan,
                "corr_mean_tumor": np.nan,
                "l2_mean_tumor": np.nan,
                "l1_mean_tumor": np.nan,
                "cosine_mean_non_tumor": np.nan,
                "corr_mean_non_tumor": np.nan,
                "l2_mean_non_tumor": np.nan,
                "l1_mean_non_tumor": np.nan,
            })

        master_rows.append(master_row)


    # Write master summary
    master_df = pd.DataFrame(master_rows).sort_values("augmentation")
    master_csv = out_root / "master_summary.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\n[OK] Wrote {master_csv}")


if __name__ == "__main__":
    main()
