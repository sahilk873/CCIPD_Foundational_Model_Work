#!/usr/bin/env python3
"""
train_consistency_all_enet_dose.py

Build a "kitchen sink" consistency-feature dataset from ALL reasonable augmentation folders
present under --aug_root (with L0/L1/L2 severity support when available), then train an
Elastic-Net logistic regression model so the model can automatically downweight / discard
unimportant features (many coefficients go to ~0, sometimes exactly 0).

What it uses
------------
A) Single-level features (per augmentation folder):
   For each augmentation folder <aug_name> that contains files <slide>.h5:
     - compute cosine, corr, l1 distance between baseline and augmented embeddings
     - apply metric-specific transforms:
         cosine/corr -> Fisher z (atanh after clipping)
         l1          -> -log1p(distance)  (direction aligned so higher is more "tumor-ish")
     - winsorize + robust-scale PER AUGMENTATION and PER METRIC

B) Dose–response features (per family with L0/L1/L2 available):
   For families detected from names like "<family>_L0", "<family>_L1", "<family>_L2",
   and for each metric in {cosine, corr, l1}:
     - compute transformed values s0,s1,s2 at L0/L1/L2
     - derive:
         slope     = (s2 - s0)/2
         min_val   = min(s0,s1,s2)
         curvature = (s2 - 2*s1 + s0)
         rel_drop  = (s2 - s0)/(abs(s0)+eps)
     - winsorize + robust-scale PER FAMILY and PER METRIC for each derived feature

C) Model:
   Elastic-Net Logistic Regression (solver='saga'), class_weight='balanced'
   This performs feature selection/shrinkage automatically.

IMPORTANT EVALUATION NOTE
-------------------------
This script builds features at the TILE level. For pathology, you should strongly consider
a slide-level split to avoid leakage. This script includes an optional slide-group split.

Outputs:
- Prints AUC, accuracy, sensitivity, specificity using Youden threshold
- Prints number of non-zero coefficients + top coefficients (feature importance)

Dependencies:
- numpy, pandas, h5py, scikit-learn

Example:
  python train_consistency_all_enet_dose.py \
    --baseline_dir /scratch/baseline_h5 \
    --aug_root /scratch/aug_root \
    --split slide \
    --l1_ratio 0.6 \
    --C 0.5
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


# -------------------------------
# H5 autodetection
# -------------------------------
COORD_KEYS = ["coords", "coordinates", "patch_coords"]
FEAT_KEYS = ["features", "embeddings", "repr", "x"]
LABEL_KEYS = ["label", "labels", "tumor", "is_tumor", "y"]


def read_first_numeric(h5, candidates, ndim=None):
    for k in h5.keys():
        if k.lower() in candidates:
            arr = np.asarray(h5[k])
            if ndim is None or arr.ndim == ndim:
                return arr
    for k in h5.keys():
        arr = np.asarray(h5[k])
        if ndim is None or arr.ndim == ndim:
            return arr
    return None


def load_h5(path: Path):
    with h5py.File(path, "r") as h5:
        feats = read_first_numeric(h5, FEAT_KEYS, ndim=2)
        coords = read_first_numeric(h5, COORD_KEYS, ndim=2)
        labels = read_first_numeric(h5, LABEL_KEYS, ndim=1)

        if feats is None:
            raise ValueError(f"No feature matrix found in {path}")

        feats = np.asarray(feats, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.int64) if coords is not None else None
        labels = np.asarray(labels).astype(int) if labels is not None else None
        return feats, coords, labels


# -------------------------------
# Alignment helpers
# -------------------------------
def coords_to_key(c: np.ndarray) -> np.ndarray:
    c0 = c[:, 0].astype(np.int64)
    c1 = c[:, 1].astype(np.int64)
    return (c0 << 32) | (c1 & 0xFFFFFFFF)


def align_by_coords_or_index(
    base_feats: np.ndarray,
    base_coords: Optional[np.ndarray],
    base_labels: Optional[np.ndarray],
    aug_feats: np.ndarray,
    aug_coords: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns: (base_aligned, aug_aligned, labels_aligned, coord_keys_aligned)
    """
    if base_coords is None or aug_coords is None:
        n = min(len(base_feats), len(aug_feats))
        labels = base_labels[:n] if base_labels is not None else None
        return base_feats[:n], aug_feats[:n], labels, None

    kb = coords_to_key(base_coords)
    ka = coords_to_key(aug_coords)
    map_b = {k: i for i, k in enumerate(kb)}
    map_a = {k: i for i, k in enumerate(ka)}
    common = sorted(set(map_b) & set(map_a))
    if not common:
        empty = base_feats[:0]
        labels = base_labels[:0] if base_labels is not None else None
        return empty, empty, labels, np.asarray([], dtype=np.uint64)

    ib = np.array([map_b[k] for k in common], dtype=np.int64)
    ia = np.array([map_a[k] for k in common], dtype=np.int64)
    labels = base_labels[ib] if base_labels is not None else None
    return base_feats[ib], aug_feats[ia], labels, np.asarray(common, dtype=np.uint64)


# -------------------------------
# Metrics
# -------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return (a * b).sum(1) / denom


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(1, keepdims=True)
    b0 = b - b.mean(1, keepdims=True)
    denom = np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1) + 1e-12
    return (a0 * b0).sum(1) / denom


def l1_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(1)


def compute_metrics(base_al: np.ndarray, aug_al: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "cosine": cosine(base_al, aug_al).astype(np.float32),
        "corr": corr(base_al, aug_al).astype(np.float32),
        "l1": l1_dist(base_al, aug_al).astype(np.float32),
    }


# -------------------------------
# Transforms (direction-align)
# -------------------------------
def transform_values(metric: str, values: np.ndarray) -> np.ndarray:
    eps = 1e-6
    v = values.astype(np.float64)
    if metric in ("cosine", "corr"):
        v = np.clip(v, -1 + eps, 1 - eps)
        return np.arctanh(v)
    if metric == "l1":
        v = np.maximum(v, 0.0)
        return -np.log1p(v)
    raise ValueError(metric)


# -------------------------------
# Scaling utilities
# -------------------------------
def winsorize_and_scale(x: np.ndarray, clip_lo: float, clip_hi: float) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = np.percentile(x, [clip_lo, clip_hi])
    x = np.clip(x, lo, hi)
    scaler = RobustScaler()
    return scaler.fit_transform(x.reshape(-1, 1)).ravel().astype(np.float32)


def parse_family_and_level(aug_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Detects patterns like "he_L0", "elastic_L2" etc.
    Returns (family, level) or (None, None) if no match.
    """
    if "_L" not in aug_name:
        return None, None
    fam, lvl = aug_name.rsplit("_L", 1)
    if not lvl.isdigit():
        return None, None
    return fam, int(lvl)


# -------------------------------
# Build wide dataset
# -------------------------------
def build_feature_table(
    baseline_dir: Path,
    aug_root: Path,
) -> pd.DataFrame:
    """
    Produces a wide per-tile table keyed by (slide, coord_key) with:
      - label
      - per-augmentation per-metric raw values:  <aug>__<metric>
    Only augmentations that have matching <slide>.h5 to baseline are included.
    """
    base_map = {p.stem: p for p in baseline_dir.glob("*.h5")}
    if not base_map:
        raise RuntimeError(f"No baseline .h5 files found in {baseline_dir}")

    aug_dirs = sorted([p for p in aug_root.iterdir() if p.is_dir()])
    if not aug_dirs:
        raise RuntimeError(f"No augmentation folders found under {aug_root}")

    all_rows: List[pd.DataFrame] = []

    for stem, base_path in base_map.items():
        base_feats, base_coords, base_labels = load_h5(base_path)
        if base_labels is None or len(base_labels) == 0:
            continue

        slide_frames: List[pd.DataFrame] = []

        for aug_dir in aug_dirs:
            aug_name = aug_dir.name
            aug_path = aug_dir / f"{stem}.h5"
            if not aug_path.exists():
                continue

            aug_feats, aug_coords, _ = load_h5(aug_path)
            b_al, a_al, labels_al, coord_keys = align_by_coords_or_index(
                base_feats, base_coords, base_labels, aug_feats, aug_coords
            )
            if labels_al is None or len(labels_al) == 0:
                continue

            m = compute_metrics(b_al, a_al)

            if coord_keys is None:
                coord_keys = np.arange(len(labels_al), dtype=np.int64)

            df_aug = pd.DataFrame(
                {
                    "slide": stem,
                    "coord_key": coord_keys,
                    "label": labels_al,
                    f"{aug_name}__cosine": m["cosine"],
                    f"{aug_name}__corr": m["corr"],
                    f"{aug_name}__l1": m["l1"],
                }
            )
            slide_frames.append(df_aug)

        if not slide_frames:
            continue

        # Inner join across all augmentations present for this slide (keeps only common tiles)
        df_slide = slide_frames[0]
        for nxt in slide_frames[1:]:
            df_slide = df_slide.merge(nxt, on=["slide", "coord_key", "label"], how="inner")

        all_rows.append(df_slide)

    if not all_rows:
        raise RuntimeError("No aligned labeled rows found. Check baseline_dir/aug_root and labels/coords in H5s.")

    return pd.concat(all_rows, ignore_index=True)


# -------------------------------
# Feature engineering: all single + all dose
# -------------------------------
def engineer_features(
    df_wide: pd.DataFrame,
    clip_lo: float,
    clip_hi: float,
    include_single: bool = True,
    include_dose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Returns:
      X, y, feature_names, groups(slide ids)
    """
    df = df_wide.copy()

    # Identify all (aug, metric) columns
    metric_cols = [c for c in df.columns if "__" in c and c not in ("slide", "coord_key")]
    # Transform all of them in-place
    for col in metric_cols:
        metric = col.split("__")[-1]
        df[col] = transform_values(metric, df[col].to_numpy(dtype=np.float32))

    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    # ---- Single-level features: winsorize+scale per column ----
    if include_single:
        for col in sorted(metric_cols):
            # skip label-ish columns; metric_cols already excludes slide/coord_key/label
            scaled = winsorize_and_scale(df[col].to_numpy(), clip_lo, clip_hi)
            feature_blocks.append(scaled.reshape(-1, 1))
            feature_names.append(f"single__{col}")

    # ---- Dose-response features: for each family that has L0/L1/L2 ----
    if include_dose:
        # Find families present
        # Example columns: he_L0__cosine, he_L1__cosine, he_L2__cosine etc.
        aug_names = sorted({c.split("__")[0] for c in metric_cols})
        fam_to_levels: Dict[str, Dict[int, str]] = {}
        for aug in aug_names:
            fam, lvl = parse_family_and_level(aug)
            if fam is None or lvl is None:
                continue
            fam_to_levels.setdefault(fam, {})[lvl] = aug

        eps = 1e-6
        for fam, lvl_map in sorted(fam_to_levels.items()):
            if not all(l in lvl_map for l in (0, 1, 2)):
                continue  # need L0/L1/L2
            aug0, aug1, aug2 = lvl_map[0], lvl_map[1], lvl_map[2]

            for metric in ("cosine", "corr", "l1"):
                c0 = f"{aug0}__{metric}"
                c1 = f"{aug1}__{metric}"
                c2 = f"{aug2}__{metric}"
                if c0 not in df.columns or c1 not in df.columns or c2 not in df.columns:
                    continue

                s0 = df[c0].to_numpy(dtype=np.float64)
                s1 = df[c1].to_numpy(dtype=np.float64)
                s2 = df[c2].to_numpy(dtype=np.float64)

                slope = (s2 - s0) / 2.0
                min_val = np.minimum(np.minimum(s0, s1), s2)
                curvature = (s2 - 2.0 * s1 + s0)
                rel_drop = (s2 - s0) / (np.abs(s0) + eps)

                derived = {
                    "slope": slope,
                    "min": min_val,
                    "curvature": curvature,
                    "rel_drop": rel_drop,
                }

                for name, arr in derived.items():
                    scaled = winsorize_and_scale(arr, clip_lo, clip_hi)
                    feature_blocks.append(scaled.reshape(-1, 1))
                    feature_names.append(f"dose__{fam}__{metric}__{name}")

    if not feature_blocks:
        raise RuntimeError("No features produced. Check augmentation folders and naming.")

    X = np.hstack(feature_blocks).astype(np.float32)
    y = df["label"].to_numpy(dtype=int)
    groups = df["slide"].to_numpy()

    return X, y, feature_names, groups


# -------------------------------
# Evaluation
# -------------------------------
def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y, s)
    thr = youden_threshold(y, s)
    yp = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return {
        "auc": float(auc),
        "threshold": float(thr),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
    }


def summarize_coefficients(model: LogisticRegression, feature_names: List[str], top_k: int = 25):
    coefs = model.coef_.ravel()
    nz = np.flatnonzero(np.abs(coefs) > 1e-8)
    print(f"nonzero_coefs: {len(nz)} / {len(coefs)}")

    if len(nz) == 0:
        return

    # Top positive (more tumor-ish) and negative (more non-tumor-ish)
    order_pos = np.argsort(coefs)[::-1]
    order_neg = np.argsort(coefs)

    print(f"\nTop +{top_k} coefficients:")
    for i in order_pos[:top_k]:
        print(f"  {coefs[i]: .4f}  {feature_names[i]}")

    print(f"\nTop -{top_k} coefficients:")
    for i in order_neg[:top_k]:
        print(f"  {coefs[i]: .4f}  {feature_names[i]}")


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--aug_root", required=True)

    ap.add_argument(
        "--split",
        choices=["tile", "slide"],
        default="slide",
        help="Split strategy. 'slide' avoids leakage by holding out entire slides (recommended).",
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--clip_lo", type=float, default=1.0)
    ap.add_argument("--clip_hi", type=float, default=99.0)

    ap.add_argument("--include_single", action="store_true", help="Include all single-level aug features.")
    ap.add_argument("--include_dose", action="store_true", help="Include all dose-response features (families with L0/L1/L2).")
    ap.set_defaults(include_single=True, include_dose=True)

    # Elastic-net hyperparameters
    ap.add_argument("--C", type=float, default=0.5, help="Inverse regularization strength (smaller -> more shrinkage)")
    ap.add_argument("--l1_ratio", type=float, default=0.6, help="0=ridge, 1=lasso. Elastic-net in (0,1).")

    ap.add_argument("--print_topk", type=int, default=30, help="How many top +/- coefficients to print.")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    aug_root = Path(args.aug_root)

    df_wide = build_feature_table(baseline_dir, aug_root)
    X, y, feature_names, groups = engineer_features(
        df_wide,
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
        include_single=args.include_single,
        include_dose=args.include_dose,
    )

    # Split
    if args.split == "slide":
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, va_idx = next(gss.split(X, y, groups=groups))
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
    else:
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )

    # Elastic-Net Logistic Regression (feature selection/shrinkage)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=args.l1_ratio,
        C=args.C,
        class_weight="balanced",
        max_iter=8000,
        random_state=args.seed,
        n_jobs=1,  # saga supports n_jobs in newer sklearn; keep conservative
    )

    model.fit(Xtr, ytr)
    scores = model.predict_proba(Xva)[:, 1]

    metrics = evaluate(yva, scores)

    print("\n==== ALL-AUG + DOSE–RESPONSE (ELASTIC-NET) ====")
    print(f"split       : {args.split}")
    print(f"n_samples   : {X.shape[0]}")
    print(f"n_features  : {X.shape[1]}")
    print(f"C          : {args.C}")
    print(f"l1_ratio    : {args.l1_ratio}")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("==============================================\n")

    summarize_coefficients(model, feature_names, top_k=args.print_topk)


if __name__ == "__main__":
    main()
