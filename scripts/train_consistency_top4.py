#!/usr/bin/env python3
"""
train_consistency_top4.py

Train a consistency-only classifier using ONLY the four best
augmentation–metric pairings:

  he_L1      → L1
  he_L0      → L1
  affine_L0  → L1
  elastic_L0 → cosine

Pipeline:
- Load baseline + selected augmented H5s
- Align tiles by coords
- Compute the single winning consistency metric per augmentation
- Apply metric-specific transforms
- Winsorize + robust-scale per augmentation
- Concatenate features across augmentations
- Train classifier (logistic regression, MLP, shallow GBDT, or GAM/EBM)
- Evaluate with ROC AUC + Youden threshold

No embeddings are used directly.

GAM option:
- Uses Explainable Boosting Machine (EBM) from `interpret` with interactions=0
  which makes it an additive (GAM-like) model.
  Install: pip install interpret
"""

import argparse
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


# -------------------------------------------------
# Selected augmentation–metric pairings (fixed)
# -------------------------------------------------
SELECTED: Dict[str, str] = {
    "he_L1": "l1",
    "he_L0": "l1",
    "affine_L0": "l1",
    "elastic_L0": "cosine",
}


# -------------------------------------------------
# H5 autodetection
# -------------------------------------------------
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

        return (
            feats.astype(np.float32),
            coords.astype(np.int64) if coords is not None else None,
            labels.astype(int) if labels is not None else None,
        )


# -------------------------------------------------
# Alignment
# -------------------------------------------------
def coords_to_key(c):
    return (c[:, 0].astype(np.int64) << 32) | (c[:, 1].astype(np.int64) & 0xFFFFFFFF)


def align(fa, ca, la, fb, cb):
    if ca is None or cb is None:
        n = min(len(fa), len(fb))
        return fa[:n], fb[:n], la[:n]

    ka, kb = coords_to_key(ca), coords_to_key(cb)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}

    common = sorted(set(ia) & set(ib))
    if not common:
        return fa[:0], fb[:0], la[:0]

    idx_a = np.array([ia[k] for k in common])
    idx_b = np.array([ib[k] for k in common])

    return fa[idx_a], fb[idx_b], la[idx_a]


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def cosine(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return (a * b).sum(1) / denom


def l1_dist(a, b):
    return np.abs(a - b).sum(1)


# -------------------------------------------------
# Dataset construction
# -------------------------------------------------
def build_dataset(baseline_dir: Path, aug_root: Path) -> pd.DataFrame:
    base_map = {p.stem: p for p in baseline_dir.glob("*.h5")}
    rows = []

    for aug_name, metric in SELECTED.items():
        aug_dir = aug_root / aug_name
        if not aug_dir.exists():
            continue

        for aug_h5 in aug_dir.glob("*.h5"):
            stem = aug_h5.stem
            if stem not in base_map:
                continue

            fb, cb, lb = load_h5(base_map[stem])
            fa, ca, _ = load_h5(aug_h5)

            fb2, fa2, labels = align(fb, cb, lb, fa, ca)
            if labels is None or len(labels) == 0:
                continue

            if metric == "l1":
                vals = l1_dist(fb2, fa2)
            else:  # cosine
                vals = cosine(fb2, fa2)

            rows.append(pd.DataFrame(
                {
                    "label": labels,
                    "value": vals,
                    "augmentation": aug_name,
                    "metric": metric,
                }
            ))

    if not rows:
        raise RuntimeError("No valid data found.")

    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------
# Transforms + scaling
# -------------------------------------------------
def transform_and_scale(df, clip_lo=1, clip_hi=99):
    out = df.copy()

    eps = 1e-6
    is_cos = out["metric"] == "cosine"
    is_l1 = out["metric"] == "l1"

    out.loc[is_cos, "value"] = np.arctanh(
        np.clip(out.loc[is_cos, "value"], -1 + eps, 1 - eps)
    )

    out.loc[is_l1, "value"] = -np.log1p(
        np.maximum(out.loc[is_l1, "value"], 0.0)
    )

    X = np.zeros(len(out), dtype=np.float32)
    for aug, idx in out.groupby("augmentation").groups.items():
        vals = out.loc[idx, "value"].to_numpy()
        lo, hi = np.percentile(vals, [clip_lo, clip_hi])
        vals = np.clip(vals, lo, hi)

        scaler = RobustScaler()
        X[idx] = scaler.fit_transform(vals.reshape(-1, 1)).ravel()

    out["X"] = X
    return out


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def evaluate(y, s):
    auc = roc_auc_score(y, s)
    thr = youden_threshold(y, s)
    yp = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return {
        "auc": auc,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--aug_root", required=True)
    ap.add_argument(
        "--model",
        choices=["logreg", "mlp", "gbdt2", "gbdt3", "gam"],
        default="logreg",
        help="Model type. 'gam' uses interpret's ExplainableBoostingClassifier with interactions=0.",
    )
    ap.add_argument(
        "--clip_lo",
        type=float,
        default=1.0,
        help="Lower percentile for winsorization per augmentation (default: 1)",
    )
    ap.add_argument(
        "--clip_hi",
        type=float,
        default=99.0,
        help="Upper percentile for winsorization per augmentation (default: 99)",
    )
    ap.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = ap.parse_args()

    df = build_dataset(Path(args.baseline_dir), Path(args.aug_root))
    df = transform_and_scale(df, clip_lo=args.clip_lo, clip_hi=args.clip_hi)

    # One feature per augmentation
    # Note: this uses df.index so each row (tile) becomes one training example with 4 augmentation features.
    X = df.pivot_table(index=df.index, columns="augmentation", values="X").fillna(0.0).to_numpy()
    y = df["label"].to_numpy()

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    if args.model == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(8,),
            alpha=1e-3,
            max_iter=1000,
            random_state=args.seed,
        )

    elif args.model == "gbdt2":
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=args.seed,
        )

    elif args.model == "gbdt3":
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=args.seed,
        )

    elif args.model == "gam":
        # GAM-like additive model via EBM (interactions=0)
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as e:
            raise SystemExit(
                "Missing dependency for --model gam. Install with: pip install interpret"
            ) from e

        model = ExplainableBoostingClassifier(
            interactions=0,     # ensures additive (GAM-like) behavior
            random_state=args.seed,
        )

    else:
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.seed,
        )

    model.fit(Xtr, ytr)

    # predict_proba API differs slightly across some models; handle uniformly
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(Xva)[:, 1]
    else:
        # Fallback: some models might only expose decision_function
        scores = model.decision_function(Xva)

    metrics = evaluate(yva, scores)

    print("\n==== TOP-4 CONSISTENCY MODEL ====")
    print(f"model       : {args.model}")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("================================\n")


if __name__ == "__main__":
    main()
