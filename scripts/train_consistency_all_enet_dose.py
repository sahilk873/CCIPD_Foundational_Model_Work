#!/usr/bin/env python3
"""
train_consistency_all_enet_dose.py  (NaN-safe + reporting)

Key guarantees:
- All-NaN groups are skipped
- Dose-response features only computed when valid
- No NaNs ever reach sklearn
- Elastic-Net performs feature selection automatically
- Explicit reporting of dead features + ENet pruning
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


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


def align_by_coords_or_index(bf, bc, bl, af, ac):
    if bc is None or ac is None:
        n = min(len(bf), len(af))
        return bf[:n], af[:n], bl[:n], None

    kb, ka = coords_to_key(bc), coords_to_key(ac)
    mb = {k: i for i, k in enumerate(kb)}
    ma = {k: i for i, k in enumerate(ka)}

    common = sorted(set(mb) & set(ma))
    if not common:
        return bf[:0], af[:0], bl[:0], np.array([], dtype=np.uint64)

    ib = np.array([mb[k] for k in common])
    ia = np.array([ma[k] for k in common])

    return bf[ib], af[ia], bl[ib], np.array(common, dtype=np.uint64)


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def cosine(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def corr(a, b):
    a0 = a - a.mean(1, keepdims=True)
    b0 = b - b.mean(1, keepdims=True)
    return (a0 * b0).sum(1) / (np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1) + 1e-12)


def l1_dist(a, b):
    return np.abs(a - b).sum(1)


METRIC_FUNCS = {"cosine": cosine, "corr": corr, "l1": l1_dist}


# -------------------------------------------------
# Transforms
# -------------------------------------------------
def transform(metric, v):
    eps = 1e-6
    v = v.astype(np.float64)

    if metric in ("cosine", "corr"):
        return np.arctanh(np.clip(v, -1 + eps, 1 - eps))
    if metric == "l1":
        return -np.log1p(np.maximum(v, 0.0))
    raise ValueError(metric)


def winsorize_and_scale_aligned(x_full, clip_lo, clip_hi):
    x = x_full.astype(np.float64)
    mask = np.isfinite(x)

    if mask.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(x[mask], [clip_lo, clip_hi])
    x_clip = np.clip(x, lo, hi)

    scaler = RobustScaler()
    out = np.zeros_like(x_clip, dtype=np.float64)
    out[mask] = scaler.fit_transform(x_clip[mask].reshape(-1, 1)).ravel()
    out[~mask] = 0.0
    return out.astype(np.float32)


# -------------------------------------------------
# Build wide table
# -------------------------------------------------
def build_feature_table(baseline_dir: Path, aug_root: Path) -> pd.DataFrame:
    base_map = {p.stem: p for p in baseline_dir.glob("*.h5")}
    rows = []

    for stem, base_path in base_map.items():
        bf, bc, bl = load_h5(base_path)
        if bl is None:
            continue

        for aug_dir in aug_root.iterdir():
            aug_path = aug_dir / f"{stem}.h5"
            if not aug_path.exists():
                continue

            af, ac, _ = load_h5(aug_path)
            b_al, a_al, lbl, keys = align_by_coords_or_index(bf, bc, bl, af, ac)
            if len(lbl) == 0:
                continue

            df = pd.DataFrame({
                "slide": stem,
                "coord": keys if keys is not None else np.arange(len(lbl)),
                "label": lbl,
            })

            for m, fn in METRIC_FUNCS.items():
                df[f"{aug_dir.name}__{m}"] = fn(b_al, a_al)

            rows.append(df)

    if not rows:
        raise RuntimeError("No valid aligned data found.")

    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------
# Feature engineering
# -------------------------------------------------
def engineer_features(df, clip_lo, clip_hi):
    feature_blocks = []
    feature_names = []

    metric_cols = [c for c in df.columns if "__" in c]

    for col in metric_cols:
        metric = col.split("__")[-1]
        vals = transform(metric, df[col].to_numpy())
        scaled = winsorize_and_scale_aligned(vals, clip_lo, clip_hi)
        feature_blocks.append(scaled.reshape(-1, 1))
        feature_names.append(f"single__{col}")

    families = {}
    for col in metric_cols:
        aug = col.split("__")[0]
        if "_L" not in aug:
            continue
        fam, lvl = aug.rsplit("_L", 1)
        if lvl.isdigit():
            families.setdefault(fam, {})[int(lvl)] = aug

    for fam, lvls in families.items():
        if not all(l in lvls for l in (0, 1, 2)):
            continue

        for metric in ("cosine", "corr", "l1"):
            try:
                s0 = transform(metric, df[f"{fam}_L0__{metric}"].to_numpy())
                s1 = transform(metric, df[f"{fam}_L1__{metric}"].to_numpy())
                s2 = transform(metric, df[f"{fam}_L2__{metric}"].to_numpy())
            except KeyError:
                continue

            slope = (s2 - s0) / 2.0
            curve = s2 - 2 * s1 + s0

            for name, arr in {"slope": slope, "curve": curve}.items():
                scaled = winsorize_and_scale_aligned(arr, clip_lo, clip_hi)
                feature_blocks.append(scaled.reshape(-1, 1))
                feature_names.append(f"dose__{fam}__{metric}__{name}")

    X = np.hstack(feature_blocks)
    X = np.nan_to_num(X)
    y = df["label"].to_numpy()
    groups = df["slide"].to_numpy()
    return X, y, feature_names, groups


# -------------------------------------------------
# Reporting utilities
# -------------------------------------------------
def summarize_feature_activity(X, names):
    zero_mask = np.all(X == 0.0, axis=0)
    print("\n==== FEATURE ACTIVITY (PRE-MODEL) ====")
    print(f"total_features        : {X.shape[1]}")
    print(f"all_zero_features     : {int(zero_mask.sum())}")
    print(f"nonzero_features      : {int((~zero_mask).sum())}")
    print("====================================\n")
    return zero_mask


def summarize_enet_coefficients(model, names, top_k=None, only_nonzero=False):
    coefs = model.coef_.ravel()
    nz = np.abs(coefs) > 1e-8

    print("\n==== ELASTIC-NET FEATURE SELECTION ====")
    print(f"total_features        : {len(coefs)}")
    print(f"zeroed_by_enet        : {int((~nz).sum())}")
    print(f"retained_by_enet      : {int(nz.sum())}")

    order = np.argsort(np.abs(coefs))[::-1]
    if only_nonzero:
        order = order[nz[order]]
    if top_k is not None:
        order = order[:top_k]

    print(f"\nAll {len(order)} coefficients (sorted by |coef|):")
    for i in order:
        print(f"{coefs[i]: .6f}  {names[i]}")
    print("=====================================\n")


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    t = thr[np.argmax(tpr - fpr)]
    yp = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return {
        "auc": roc_auc_score(y, s),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--aug_root", required=True)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--l1_ratio", type=float, default=0.6)
    ap.add_argument("--split", choices=["tile", "slide"], default="slide")
    ap.add_argument("--clip_lo", type=float, default=1.0)
    ap.add_argument("--clip_hi", type=float, default=99.0)
    args = ap.parse_args()

    df = build_feature_table(Path(args.baseline_dir), Path(args.aug_root))
    X, y, names, groups = engineer_features(df, args.clip_lo, args.clip_hi)

    summarize_feature_activity(X, names)

    if args.split == "slide":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, va = next(gss.split(X, y, groups))
        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, stratify=y, test_size=0.2)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=args.l1_ratio,
        C=args.C,
        class_weight="balanced",
        max_iter=8000,
    )

    model.fit(Xtr, ytr)
    scores = model.predict_proba(Xva)[:, 1]
    metrics = evaluate(yva, scores)

    print("\n==== ALL-AUG + DOSE (NaN-SAFE ELASTIC-NET) ====")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("============================================\n")

    summarize_enet_coefficients(model, names, top_k=None, only_nonzero=False)


if __name__ == "__main__":
    main()
