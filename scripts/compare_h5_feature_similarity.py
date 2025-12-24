#!/usr/bin/env python3
"""
compare_h5_feature_similarity.py

Baseline vs Augmented H5 feature comparison + tumor vs non-tumor consistency difference tests.

Outputs:
- tile_level.csv
- summary.csv
- group_diff.csv
- summary.json
- (optional) plots/*.png

NEW:
- consistency_auc.csv (ROC/AUC + Youden J optimal threshold for each metric, using consistency alone as classifier)
- Prints a readable summary to stdout (including group summaries + significance tests + AUC summary).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import pandas as pd

# NEW: AUC / ROC
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
    """
    Cliff's delta using Mann–Whitney U relation when scipy is available.
    Positive means x tends to be larger than y.
    """
    try:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(x, y, alternative="two-sided").statistic
        n = x.size
        m = y.size
        return float((2.0 * u) / (n * m) - 1.0)
    except Exception:
        # fallback approximate via sampling (works without scipy)
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


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}"
    return f"{x:.4f}"


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_summaries(summary_df: pd.DataFrame) -> None:
    print("\n========== CONSISTENCY SUMMARY ==========")
    for metric in summary_df["metric"].unique():
        sub = summary_df[summary_df["metric"] == metric]
        direction = sub["direction"].iloc[0] if "direction" in sub.columns else ""
        print(f"\nMetric: {metric} ({direction})")
        for grp in ["all", "tumor", "non_tumor"]:
            row = sub[sub["group"] == grp]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"  {grp:9s} n={int(r['n']):6d} "
                f"mean={fmt_num(r['mean'])} std={fmt_num(r['std'])} "
                f"median={fmt_num(r['median'])} "
                f"p25={fmt_num(r['p25'])} p75={fmt_num(r['p75'])}"
            )
    print("========================================\n")


def print_group_tests(group_diff_df: pd.DataFrame) -> None:
    print("\n========== TUMOR vs NON-TUMOR DIFFERENCE ==========")
    for _, r in group_diff_df.iterrows():
        metric = r["metric"]
        direction = r.get("direction", "")
        test = r.get("test", "")
        print(f"\nMetric: {metric} ({direction}) | test={test}")
        print(
            f"  tumor n={int(r['n_tumor'])} mean={fmt_num(r['mean_tumor'])} median={fmt_num(r['median_tumor'])}"
        )
        print(
            f"  non   n={int(r['n_non_tumor'])} mean={fmt_num(r['mean_non_tumor'])} median={fmt_num(r['median_non_tumor'])}"
        )
        print(
            f"  p={fmt_p(float(r['p_value']))}  q(BH)={fmt_p(float(r.get('q_value_bh', np.nan)))}  "
            f"Cliff's δ={fmt_num(float(r.get('effect_cliffs_delta', np.nan)))}"
        )
    print("===================================================\n")


# -------------------------
# NEW: ROC/AUC + Youden J helpers
# -------------------------
def _youden_j_threshold(y_true: np.ndarray, scores: np.ndarray, higher_is_more_tumor: bool) -> Dict[str, float]:
    """
    Compute ROC + AUC + Youden's J optimal threshold for a single score.

    sklearn assumes higher score => more likely positive (tumor=1).
    If higher_is_more_tumor is False, we sign-flip scores internally so that higher => tumor.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    # Flip sign if needed so that "higher = more tumor"
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

    # Convert threshold back to original score space for user readability
    best_thr_original = best_thr_internal if higher_is_more_tumor else -best_thr_internal

    return {
        "auc": auc,
        "best_threshold_internal": best_thr_internal,
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


def compute_auc_for_all_metrics(
    tile_df: pd.DataFrame,
    label_col: str,
    metric_specs: Dict[str, Dict[str, object]],
    out_csv_path: Path,
) -> pd.DataFrame:
    """
    metric_specs: metric_name -> {"col": <column name>, "higher_is_more_consistent": bool}

    We assume empirical direction you observed:
    more consistent => more tumor-like

    Therefore:
      - similarity metrics (higher=more_consistent) => higher=more_tumor
      - distance metrics (lower=more_consistent)   => lower=more_tumor
    """
    y_true = tile_df[label_col].astype(int).to_numpy()

    rows = []
    for metric_name, spec in metric_specs.items():
        col = str(spec["col"])
        higher_is_more_consistent = bool(spec["higher_is_more_consistent"])
        higher_is_more_tumor = higher_is_more_consistent  # because "more consistent => tumor"

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


def print_auc_summary(auc_df: pd.DataFrame) -> None:
    print("\n========== CONSISTENCY-ONLY TILE CLASSIFICATION (ROC/AUC) ==========")
    for _, r in auc_df.iterrows():
        print(
            f"Metric: {str(r['metric']):<6} | AUC={float(r['auc']):.4f} | "
            f"Youden-thr({str(r['score_direction'])})={fmt_num(float(r['best_threshold_original_score_space']))} | "
            f"sens={float(r['sensitivity']):.3f} spec={float(r['specificity']):.3f} acc={float(r['accuracy']):.3f}"
        )
    print("===============================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_h5", required=True, help="H5 with baseline (non-augmented) features")
    ap.add_argument("--aug_h5", required=True, help="H5 with augmented features")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--label_precedence", choices=["baseline", "aug"], default="baseline",
                    help="Which file's labels to use if both present and disagree (default: baseline)")
    ap.add_argument("--require_labels", action="store_true",
                    help="Error if labels cannot be found. Otherwise script runs without stratified summaries/tests.")
    ap.add_argument("--plots_dir", default=None, help="If set, save metric histograms here")
    ap.add_argument("--use_ttest", action="store_true",
                    help="Use Welch t-test instead of Mann–Whitney U (default: Mann–Whitney).")
    ap.add_argument("--print_topk", type=int, default=0,
                    help="If >0, print top-k most/least consistent tiles per metric (by label)")

    # NEW: allow disabling AUC if you want (default on when labels exist)
    ap.add_argument("--skip_auc", action="store_true",
                    help="Skip ROC/AUC + Youden threshold analysis (default: run when labels exist).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_h5)
    aug_path = Path(args.aug_h5)

    coords_b, feats_b, labels_b, keys_b = load_h5_triplet(baseline_path)
    coords_a, feats_a, labels_a, keys_a = load_h5_triplet(aug_path)

    print("[INFO] Baseline keys used:", keys_b)
    print("[INFO] Aug keys used:", keys_a)
    print(f"[INFO] Baseline feats: {feats_b.shape} | Aug feats: {feats_a.shape}")

    if coords_b is not None and coords_a is not None:
        feats_b2, feats_a2, lb2, la2, coords2 = align_by_coords(
            coords_b, feats_b, labels_b, coords_a, feats_a, labels_a
        )
        print(f"[INFO] Aligned by coords: {feats_b2.shape[0]} tiles matched")
    else:
        n = min(feats_b.shape[0], feats_a.shape[0])
        feats_b2 = feats_b[:n]
        feats_a2 = feats_a[:n]
        lb2 = labels_b[:n] if labels_b is not None else None
        la2 = labels_a[:n] if labels_a is not None else None
        coords2 = None
        print(f"[WARN] Missing coords in one or both files; aligned by index for n={n}")

    labels = None
    if lb2 is not None and la2 is not None:
        if not np.array_equal(lb2, la2):
            print("[WARN] Labels differ between files after alignment.")
        labels = lb2 if args.label_precedence == "baseline" else la2
    elif lb2 is not None:
        labels = lb2
    elif la2 is not None:
        labels = la2

    if args.require_labels and labels is None:
        raise RuntimeError("Labels not found in either H5, but --require_labels was set.")

    # metrics
    cos = cosine_sim(feats_b2, feats_a2)
    l2 = l2_dist(feats_b2, feats_a2)
    l1 = l1_dist(feats_b2, feats_a2)
    corr = corr_sim(feats_b2, feats_a2)

    metrics = {
        "cosine": cos,   # higher = more consistent
        "corr": corr,    # higher = more consistent
        "l2": l2,        # lower = more consistent
        "l1": l1,        # lower = more consistent
    }
    direction = {
        "cosine": "higher=more_consistent",
        "corr": "higher=more_consistent",
        "l2": "lower=more_consistent",
        "l1": "lower=more_consistent",
    }

    # tile-level df
    df = pd.DataFrame({k: v for k, v in metrics.items()})
    if coords2 is not None:
        df.insert(0, "x", coords2[:, 0])
        df.insert(1, "y", coords2[:, 1])
    if labels is not None:
        df["label"] = labels.astype(int)

    tile_csv = out_dir / "tile_level.csv"
    df.to_csv(tile_csv, index=False)
    print(f"[OK] Wrote {tile_csv}")

    # summary df
    rows = []

    def add_summary(group_name: str, mask: Optional[np.ndarray]):
        for mname, vals in metrics.items():
            vv = vals if mask is None else vals[mask]
            s = summarize_metric(vv)
            rows.append({"group": group_name, "metric": mname, "direction": direction[mname], **s})

    add_summary("all", None)

    if labels is not None:
        tumor_mask = (labels.astype(int) == 1)
        non_mask = ~tumor_mask
        add_summary("tumor", tumor_mask)
        add_summary("non_tumor", non_mask)

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Wrote {summary_csv}")

    # tests
    tests_rows = []
    group_diff_df = None
    if labels is not None:
        tumor_mask = (labels.astype(int) == 1)
        non_mask = ~tumor_mask

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

            p = welch_ttest_p(x, y) if args.use_ttest else mann_whitney_p(x, y)
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
                "test": "welch_ttest" if args.use_ttest else "mann_whitney_u",
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

    # NEW: AUC/ROC using consistency alone
    auc_df = None
    if labels is not None and not args.skip_auc:
        metric_specs = {
            "cosine": {"col": "cosine", "higher_is_more_consistent": True},
            "corr":   {"col": "corr",   "higher_is_more_consistent": True},
            "l2":     {"col": "l2",     "higher_is_more_consistent": False},
            "l1":     {"col": "l1",     "higher_is_more_consistent": False},
        }
        auc_csv = out_dir / "consistency_auc.csv"
        auc_df = compute_auc_for_all_metrics(df, label_col="label", metric_specs=metric_specs, out_csv_path=auc_csv)
        print(f"[OK] Wrote {auc_csv}")

    # print summaries to stdout
    print_summaries(summary_df)
    if group_diff_df is not None:
        print_group_tests(group_diff_df)
    else:
        print("[WARN] No labels found; skipping tumor vs non-tumor significance tests.\n")

    if auc_df is not None:
        print_auc_summary(auc_df)
    elif labels is not None and args.skip_auc:
        print("[INFO] --skip_auc set; skipping ROC/AUC analysis.\n")
    elif labels is None:
        print("[WARN] No labels found; skipping ROC/AUC analysis.\n")

    # optional: print top-k most/least consistent tiles
    if args.print_topk and args.print_topk > 0:
        k = int(args.print_topk)
        print("\n========== TOP-K TILES (BY METRIC) ==========")
        for mname, vals in metrics.items():
            print(f"\nMetric: {mname} ({direction[mname]})")
            # for l1/l2, "more consistent" means smaller values
            if mname in ("l1", "l2"):
                order_best = np.argsort(vals)[:k]
                order_worst = np.argsort(vals)[-k:][::-1]
            else:
                order_best = np.argsort(vals)[-k:][::-1]
                order_worst = np.argsort(vals)[:k]
            print("  Most consistent indices:", order_best.tolist())
            print("  Least consistent indices:", order_worst.tolist())
        print("============================================\n")

    # json
    summary_json = out_dir / "summary.json"
    out_json = {
        "baseline_h5": str(baseline_path),
        "aug_h5": str(aug_path),
        "keys_used": {"baseline": keys_b, "aug": keys_a},
        "summary_rows": rows,
        "group_diff_tests": tests_rows,
        "auc_results": auc_df.to_dict(orient="records") if auc_df is not None else None,
    }
    with open(summary_json, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[OK] Wrote {summary_json}")

    # plots
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        for name, vals in metrics.items():
            plt.figure()
            plt.hist(vals, bins=50)
            plt.title(name)
            plt.xlabel(name)
            plt.ylabel("count")
            outp = plots_dir / f"{name}.png"
            plt.savefig(outp, dpi=200, bbox_inches="tight")
            plt.close()
        print(f"[OK] Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
