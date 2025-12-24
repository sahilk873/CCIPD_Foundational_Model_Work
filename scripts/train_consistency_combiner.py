#!/usr/bin/env python3
"""
train_consistency_combiner.py

Train a consistency-only combiner model using feature stability statistics.

Fixes implemented (0–4):
0) Correct cosine/corr norms (per-row axis=1) and numerical stability.
1) Normalize within each augmentation (per-augmentation RobustScaler).
2) Handle heavy tails with per-augmentation winsorization (percentile clipping).
3) Make metric shapes more compatible:
   - Fisher z-transform (atanh) for corr/cosine after clipping to (-1, 1)
   - log1p transform for L1/L2 distances (then direction align via negation)
4) Add augmentation one-hot features so the combiner can learn aug-specific weights.

Pipeline:
- Load baseline + augmented H5s:
    baseline_dir/<slide>.h5
    aug_root/<augmentation>/<slide>.h5
- Align tiles by coords
- Compute per-tile consistency metrics
- Apply transforms + tail clipping per augmentation
- Scale per augmentation
- Add augmentation one-hots
- Train combiner on consistency features ONLY
- 80/20 train/validation split (stratified)
- Report AUC + accuracy + sensitivity + specificity (threshold via Youden’s J)

No embeddings are used for training, only stability metrics derived from embeddings.
"""

import argparse
from pathlib import Path
from typing import Optional, List

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


# -------------------------
# Dataset autodetection
# -------------------------
COORD_KEYS = ["coords", "coordinates", "patch_coords"]
FEAT_KEYS = ["features", "embeddings", "repr", "x"]
LABEL_KEYS = ["label", "labels", "tumor", "is_tumor", "y"]


def read_first_numeric(h5, candidates, ndim=None):
    """Find first dataset whose key matches candidates (case-insensitive), else first numeric of ndim."""
    # First pass: key match
    for k in h5.keys():
        if k.lower() in candidates:
            arr = np.asarray(h5[k])
            if ndim is None or arr.ndim == ndim:
                return arr
    # Fallback: first numeric array of requested ndim
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
            raise ValueError(f"No 2D feature array found in {path}")
        feats = np.asarray(feats, dtype=np.float32)

        if coords is not None:
            coords = np.asarray(coords)
        if labels is not None:
            labels = np.asarray(labels).astype(int)

        return feats, coords, labels


# -------------------------
# Alignment
# -------------------------
def coords_to_key(c: np.ndarray) -> np.ndarray:
    """Pack (x,y) int coords into a single uint64 key."""
    c0 = c[:, 0].astype(np.int64)
    c1 = c[:, 1].astype(np.int64)
    return (c0 << 32) | (c1 & 0xFFFFFFFF)


def align(feats_a, coords_a, labels_a, feats_b, coords_b, labels_b):
    """
    Align two feature matrices by coords intersection; return (fa_aligned, fb_aligned, labels_aligned_from_a_if_present)
    If coords missing, align by min length.
    """
    if coords_a is None or coords_b is None:
        n = min(len(feats_a), len(feats_b))
        la = labels_a[:n] if labels_a is not None else None
        return feats_a[:n], feats_b[:n], la

    ka = coords_to_key(coords_a)
    kb = coords_to_key(coords_b)

    map_a = {k: i for i, k in enumerate(ka)}
    map_b = {k: i for i, k in enumerate(kb)}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        return feats_a[:0], feats_b[:0], (labels_a[:0] if labels_a is not None else None)

    ia = np.array([map_a[k] for k in common], dtype=np.int64)
    ib = np.array([map_b[k] for k in common], dtype=np.int64)

    la = labels_a[ia] if labels_a is not None else None
    return feats_a[ia], feats_b[ib], la


# -------------------------
# Consistency metrics (Fix 0)
# -------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # per-row dot / (per-row norm * per-row norm)
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    return (a * b).sum(1) / denom


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # per-row Pearson correlation
    a0 = a - a.mean(1, keepdims=True)
    b0 = b - b.mean(1, keepdims=True)
    denom = (np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)) + 1e-12
    return (a0 * b0).sum(1) / denom


def l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(1))


def l1_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(1)


FEATURES: List[str] = ["cosine", "corr", "l2", "l1"]


# -------------------------
# Fix 3: metric transforms
# -------------------------
def apply_metric_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Fisher z for corr/cosine (atanh) after clipping to (-1, 1)
    - log1p on distances then direction-align (negate)
    """
    out = df.copy()

    eps = 1e-6
    for col in ["cosine", "corr"]:
        x = out[col].to_numpy(dtype=np.float64)
        x = np.clip(x, -1 + eps, 1 - eps)
        out[col] = np.arctanh(x)  # Fisher z

    # Distances -> log1p then negate so "higher = more tumor-ish" direction is aligned (consistent with corr/cosine)
    for col in ["l2", "l1"]:
        x = out[col].to_numpy(dtype=np.float64)
        x = np.log1p(np.maximum(x, 0.0))
        out[col] = -x

    return out


# -------------------------
# Fix 2: tail clipping per augmentation (winsorize)
# -------------------------
def clip_by_group(df: pd.DataFrame, cols: List[str], group_col: str = "augmentation",
                  lo: float = 1.0, hi: float = 99.0) -> pd.DataFrame:
    out = df.copy()
    for aug, g in out.groupby(group_col):
        idx = g.index
        for col in cols:
            vals = g[col].to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            qlo, qhi = np.percentile(vals, [lo, hi])
            out.loc[idx, col] = np.clip(vals, qlo, qhi)
    return out


# -------------------------
# Fix 1: per-augmentation robust scaling
# -------------------------
def per_aug_robust_scale(df: pd.DataFrame, cols: List[str], group_col: str = "augmentation") -> np.ndarray:
    Xn = np.zeros((len(df), len(cols)), dtype=np.float32)
    for aug, idx in df.groupby(group_col).groups.items():
        scaler = RobustScaler()
        Xn[idx, :] = scaler.fit_transform(df.loc[idx, cols].to_numpy(dtype=np.float32))
    return Xn


# -------------------------
# Build dataset
# -------------------------
def build_dataset(baseline_dir: Path, aug_root: Path) -> pd.DataFrame:
    rows = []
    base_idx = {p.stem: p for p in baseline_dir.glob("*.h5")}

    for aug_dir in sorted([p for p in aug_root.iterdir() if p.is_dir()]):
        for aug_h5 in aug_dir.glob("*.h5"):
            stem = aug_h5.stem
            if stem not in base_idx:
                continue

            fb, cb, lb = load_h5(base_idx[stem])
            fa, ca, la = load_h5(aug_h5)

            fb2, fa2, labels = align(fb, cb, lb, fa, ca, la)
            if labels is None or len(labels) == 0:
                continue

            rows.append(pd.DataFrame({
                "label": labels,
                "cosine": cosine(fb2, fa2),
                "corr": corr(fb2, fa2),
                "l2": l2_dist(fb2, fa2),   # distance (transform later)
                "l1": l1_dist(fb2, fa2),   # distance (transform later)
                "augmentation": aug_dir.name
            }))

    if not rows:
        raise ValueError(f"No aligned labeled rows found. Check baseline_dir={baseline_dir} aug_root={aug_root}")

    return pd.concat(rows, ignore_index=True)


# -------------------------
# Train + evaluate
# -------------------------
def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def evaluate(y_true: np.ndarray, scores: np.ndarray):
    auc = roc_auc_score(y_true, scores)
    thr = youden_threshold(y_true, scores)
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": auc,
        "threshold": thr,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, help="Directory with baseline .h5 files (one per slide)")
    ap.add_argument("--aug_root", required=True, help="Root directory with subfolders per augmentation, each containing .h5 files")
    ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--clip_lo", type=float, default=1.0, help="Lower percentile for per-augmentation winsorization (default: 1)")
    ap.add_argument("--clip_hi", type=float, default=99.0, help="Upper percentile for per-augmentation winsorization (default: 99)")
    args = ap.parse_args()

    df = build_dataset(Path(args.baseline_dir), Path(args.aug_root))

    # Fix 3: transforms for distribution compatibility
    df = apply_metric_transforms(df)

    # Fix 2: per-augmentation tail clipping after transforms
    df = clip_by_group(df, FEATURES, group_col="augmentation", lo=args.clip_lo, hi=args.clip_hi)

    # Fix 1: per-augmentation scaling of the metric features
    X_metrics = per_aug_robust_scale(df, FEATURES, group_col="augmentation")

    # Fix 4: add augmentation one-hots so model can learn aug-specific weighting
    aug_oh = pd.get_dummies(df["augmentation"], prefix="aug").to_numpy(dtype=np.float32)
    X = np.hstack([X_metrics, aug_oh]).astype(np.float32)

    y = df["label"].to_numpy(dtype=int)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if args.model == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            alpha=1e-3,
            max_iter=1000,
            random_state=42,
        )
    else:
        model = LogisticRegression(max_iter=2000, class_weight="balanced")

    model.fit(Xtr, ytr)
    scores = model.predict_proba(Xva)[:, 1]

    metrics = evaluate(yva, scores)

    print("\n========== CONSISTENCY COMBINER PERFORMANCE ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:12s}: {v:.4f}")
        else:
            print(f"{k:12s}: {v}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()