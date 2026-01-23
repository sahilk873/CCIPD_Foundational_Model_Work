#!/usr/bin/env python3
"""
stability_select_and_train.py

What this does
--------------
(1) Reads NORMAL (unaugmented) .h5 files from --normal_h5_dir.
    - Must contain embeddings (N,D)
    - Must contain labels (N,)
    - Optional coords (N,2) or similar

(2) Reads AUGMENTED .h5 files stored under augmentation subfolders in --aug_root_dir:
      aug_root_dir/
        gaussian_L0/slide_001.h5
        affine_L0/slide_001.h5
        elastic_L1/slide_001.h5
        ...
    i.e., SAME filename repeated across multiple augmentation folders.

(3) Matches normal and aug by exact filename (e.g., slide_001.h5).

(4) Computes per-dimension instability (stability ranking) ONCE:
      instability[j] = mean_over_tiles( var_over_augviews( embedding[tile, aug, j] ) )

    Then saves:
      out_dir/stability_cache.npz
        - instability (D,)
        - stable_rank (D,)  (argsort(instability), lowest first)
        - dim (D,)
        - metadata (json string)

(5) Subsequent runs can skip recomputation with --use_cache.

(6) Selects top-K most stable dimensions and trains an sklearn MLP using:
      --train_source normal   (default; uses normal embeddings)
      --train_source aug_mean (uses mean across augmented views per tile)

Outputs
-------
- stability_cache.npz
- selected_stable_dims.npy
- stable_dims_mlp.joblib
- run_config.json
- summary.txt

Example
-------
# First run (computes and saves stability ranking)
python stability_select_and_train.py \
  --normal_h5_dir /path/normal \
  --aug_root_dir /path/aug_root \
  --out_dir /path/out \
  --k 512 \
  --min_aug_views 2

# Next run (reuses cached ranking; no stability recompute)
python stability_select_and_train.py \
  --normal_h5_dir /path/normal \
  --aug_root_dir /path/aug_root \
  --out_dir /path/out \
  --k 256 \
  --use_cache

Notes
-----
- Alignment uses coords when available in BOTH files, else assumes identical row ordering.
- Stability computation is "streaming" across cases: we never keep everything in RAM.
- If some cases are missing some augmentation views, that's fine as long as they have >= min_aug_views.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ----------------------------
# H5 key inference
# ----------------------------
EMBED_KEYS = [
    "embeddings", "embedding",
    "features", "feats",
    "x", "X",
    "repr", "representation",
    "z", "Z",
]
COORD_KEYS = [
    "coords", "coordinates",
    "xy", "tile_coords", "patch_coords",
]
LABEL_KEYS = [
    "labels", "label",
    "y", "Y",
    "tumor", "tumor_label",
    "target", "targets",
    "class", "classes",
]


# ----------------------------
# File listing / indexing
# ----------------------------
def list_h5_flat(folder: str) -> List[str]:
    p = Path(folder)
    out: List[str] = []
    for ext in ("*.h5", "*.hdf5", "*.hdf"):
        out.extend([str(x) for x in p.glob(ext)])
    return sorted(out)


def list_h5_recursive(folder: str) -> List[str]:
    p = Path(folder)
    out: List[str] = []
    for ext in ("*.h5", "*.hdf5", "*.hdf"):
        out.extend([str(x) for x in p.rglob(ext)])
    return sorted(out)

def build_aug_index(aug_root_dir: str, allowed_augs: Optional[set[str]] = None) -> Dict[str, List[str]]:
    """
    Returns mapping: filename (basename) -> list of aug file paths across subfolders.

    Filtering:
      If allowed_augs is provided, only include augmented files whose immediate parent folder name
      (e.g. 'gaussian_L0') is in allowed_augs.
    """
    aug_paths = list_h5_recursive(aug_root_dir)
    index: DefaultDict[str, List[str]] = defaultdict(list)

    for p in aug_paths:
        pp = Path(p)
        aug_folder = pp.parent.name  # e.g. gaussian_L0, affine_L2, etc.

        if allowed_augs is not None and aug_folder not in allowed_augs:
            continue

        index[pp.name].append(str(pp))

    return {k: sorted(v) for k, v in index.items()}



# ----------------------------
# H5 readers
# ----------------------------
def _find_first_dataset_key(h5: h5py.File, candidates: List[str]) -> Optional[str]:
    # root
    for k in candidates:
        if k in h5 and isinstance(h5[k], h5py.Dataset):
            return k
    # one-level deep
    for name in h5.keys():
        obj = h5[name]
        if isinstance(obj, h5py.Group):
            for k in candidates:
                path = f"{name}/{k}"
                if path in h5 and isinstance(h5[path], h5py.Dataset):
                    return path
    return None


def read_embeddings_coords_labels(normal_h5_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, str]]:
    meta: Dict[str, str] = {}
    with h5py.File(normal_h5_path, "r") as h5:
        emb_key = _find_first_dataset_key(h5, EMBED_KEYS)
        if emb_key is None:
            raise KeyError(f"No embedding dataset found in normal file {normal_h5_path}. Tried {EMBED_KEYS}")
        X = np.array(h5[emb_key])

        lab_key = _find_first_dataset_key(h5, LABEL_KEYS)
        if lab_key is None:
            raise KeyError(f"No label dataset found in normal file {normal_h5_path}. Tried {LABEL_KEYS}")
        y = np.array(h5[lab_key]).astype(np.int64).reshape(-1)

        coord_key = _find_first_dataset_key(h5, COORD_KEYS)
        coords = np.array(h5[coord_key]) if coord_key is not None else None

        meta["emb_key"] = emb_key
        meta["lab_key"] = lab_key
        meta["coord_key"] = coord_key or ""
    return X, coords, y, meta


def read_embeddings_coords(h5_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, str]]:
    meta: Dict[str, str] = {}
    with h5py.File(h5_path, "r") as h5:
        emb_key = _find_first_dataset_key(h5, EMBED_KEYS)
        if emb_key is None:
            raise KeyError(f"No embedding dataset found in {h5_path}. Tried {EMBED_KEYS}")
        X = np.array(h5[emb_key])

        coord_key = _find_first_dataset_key(h5, COORD_KEYS)
        coords = np.array(h5[coord_key]) if coord_key is not None else None

        meta["emb_key"] = emb_key
        meta["coord_key"] = coord_key or ""
    return X, coords, meta


# ----------------------------
# Alignment
# ----------------------------
def _coords_to_tuples(c: np.ndarray) -> List[Tuple[int, ...]]:
    c = np.asarray(c)
    if c.ndim == 1:
        return [(int(v),) for v in c.tolist()]
    return [tuple(int(v) for v in row.tolist()) for row in c]


def align_X_to_y(
    X: np.ndarray,
    coords_X: Optional[np.ndarray],
    y: np.ndarray,
    coords_y: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align rows of X to rows of y by coords when possible, else assume same ordering.
    """
    if coords_X is None or coords_y is None:
        n = min(len(y), X.shape[0])
        return X[:n], y[:n]

    tx = _coords_to_tuples(coords_X)
    ty = _coords_to_tuples(coords_y)
    map_y = {coord: idx for idx, coord in enumerate(ty)}

    ix: List[int] = []
    iy: List[int] = []
    for i, coord in enumerate(tx):
        j = map_y.get(coord)
        if j is not None:
            ix.append(i)
            iy.append(j)

    if len(ix) == 0:
        n = min(len(y), X.shape[0])
        return X[:n], y[:n]

    X2 = X[np.array(ix)]
    y2 = y[np.array(iy)]
    return X2, y2


# ----------------------------
# Stability computation (streaming)
# ----------------------------
def compute_instability_for_case(
    aug_paths_for_case: List[str],
    y: np.ndarray,
    coords_y: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    """
    Reads each augmentation view file (N,D), aligns to y, stacks to (A,N,D),
    computes instability per dim = mean over tiles of var across A.

    Returns:
      instab_case: (D,)
      A_used: number of augmentation views used
    """
    views: List[np.ndarray] = []
    D: Optional[int] = None
    n_min: Optional[int] = None

    for p in aug_paths_for_case:
        X_aug, coords_aug, _ = read_embeddings_coords(p)
        X_aligned, y_aligned = align_X_to_y(X_aug, coords_aug, y, coords_y)

        if D is None:
            D = X_aligned.shape[1]
        else:
            if X_aligned.shape[1] != D:
                # skip inconsistent dimensionality view
                continue

        if n_min is None:
            n_min = len(y_aligned)
        else:
            n_min = min(n_min, len(y_aligned))

        views.append(X_aligned)

    if D is None or n_min is None or len(views) == 0:
        raise RuntimeError("No valid augmentation views for this case.")

    # Truncate all views to common n
    views = [v[:n_min] for v in views]
    Z = np.stack(views, axis=0)  # (A, n, D)

    # instability: var across A -> (n,D), then mean over n -> (D,)
    instab_case = Z.var(axis=0).mean(axis=0)
    return instab_case, Z.shape[0]


def compute_and_cache_stability(
    normal_files: List[str],
    aug_index: Dict[str, List[str]],
    out_dir: Path,
    min_aug_views: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Computes global instability averaged over cases, saves a cache file, and returns:
      {"instability": (D,), "stable_rank": (D,), "dim": (D,)}
    """
    instability_sum: Optional[np.ndarray] = None
    used_cases = 0
    skipped_cases = 0
    dim_D: Optional[int] = None

    # iterate normals as the source of filenames + labels/coords
    for nf in tqdm(normal_files, desc="Stability pass (compute)"):
        fname = Path(nf).name
        aug_paths = aug_index.get(fname, [])
        if len(aug_paths) < min_aug_views:
            skipped_cases += 1
            continue

        try:
            X_norm, coords_y, y, _ = read_embeddings_coords_labels(nf)
        except Exception:
            skipped_cases += 1
            continue

        try:
            instab_case, A_used = compute_instability_for_case(aug_paths, y, coords_y)
        except Exception:
            skipped_cases += 1
            continue

        if A_used < min_aug_views:
            skipped_cases += 1
            continue

        if dim_D is None:
            dim_D = instab_case.shape[0]
        else:
            if instab_case.shape[0] != dim_D:
                skipped_cases += 1
                continue

        if instability_sum is None:
            instability_sum = np.zeros_like(instab_case, dtype=np.float64)

        instability_sum += instab_case
        used_cases += 1

    if used_cases == 0 or instability_sum is None:
        raise RuntimeError(
            "No usable cases for stability computation.\n"
            "Checklist:\n"
            "- Are there augmented views for each normal filename under aug_root_dir?\n"
            "- Do at least some cases have >= --min_aug_views augmentation files?\n"
            "- Do augmented files contain embeddings under one of keys: "
            f"{EMBED_KEYS} ?\n"
            "- Do normal files contain labels under one of keys: "
            f"{LABEL_KEYS} ?"
        )

    instability = instability_sum / used_cases  # (D,)
    stable_rank = np.argsort(instability)        # lowest (most stable) -> highest
    dim = np.arange(instability.shape[0], dtype=np.int64)

    meta = {
        "used_cases": used_cases,
        "skipped_cases": skipped_cases,
        "min_aug_views": min_aug_views,
        "seed": seed,
        "embed_keys": EMBED_KEYS,
        "coord_keys": COORD_KEYS,
        "label_keys": LABEL_KEYS,
    }

    cache_path = out_dir / "stability_cache.npz"
    np.savez_compressed(
        cache_path,
        instability=instability.astype(np.float32),
        stable_rank=stable_rank.astype(np.int64),
        dim=dim,
        meta=json.dumps(meta),
    )

    print(f"\nSaved stability cache to: {cache_path}")
    print(f"Stability computed from used_cases={used_cases}, skipped_cases={skipped_cases}")
    return {"instability": instability, "stable_rank": stable_rank, "dim": dim}


def load_stability_cache(out_dir: Path) -> Dict[str, np.ndarray]:
    cache_path = out_dir / "stability_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    instability = data["instability"].astype(np.float64)
    stable_rank = data["stable_rank"].astype(np.int64)
    dim = data["dim"].astype(np.int64)
    return {"instability": instability, "stable_rank": stable_rank, "dim": dim}


# ----------------------------
# Training data construction
# ----------------------------
def build_train_matrix(
    normal_files: List[str],
    aug_index: Dict[str, List[str]],
    stable_dims: np.ndarray,
    train_source: str,
    min_aug_views: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    used = 0
    skipped = 0

    for nf in tqdm(normal_files, desc="Build train matrix"):
        fname = Path(nf).name

        try:
            X_norm, coords_y, y, _ = read_embeddings_coords_labels(nf)
        except Exception:
            skipped += 1
            continue

        # Align/truncate X_norm to y (in case label length differs)
        X_norm2, y2 = align_X_to_y(X_norm, coords_y, y, coords_y)

        if train_source == "normal":
            X_use = X_norm2

        elif train_source == "aug_mean":
            aug_paths = aug_index.get(fname, [])
            if len(aug_paths) < min_aug_views:
                skipped += 1
                continue

            # load and align each view, then average across views
            views: List[np.ndarray] = []
            D = X_norm2.shape[1]
            n_min = len(y2)

            for p in aug_paths:
                try:
                    X_aug, coords_aug, _ = read_embeddings_coords(p)
                    Xa, ya = align_X_to_y(X_aug, coords_aug, y, coords_y)
                    if Xa.shape[1] != D:
                        continue
                    n_min = min(n_min, len(ya))
                    views.append(Xa)
                except Exception:
                    continue

            if len(views) < min_aug_views:
                skipped += 1
                continue

            views = [v[:n_min] for v in views]
            X_use = np.stack(views, axis=0).mean(axis=0)  # (n_min, D)
            y2 = y2[:n_min]
        else:
            raise ValueError(f"Unknown train_source: {train_source}")

        # Slice stable dims
        n = min(len(y2), X_use.shape[0])
        if n < 10:
            skipped += 1
            continue

        X_list.append(X_use[:n, :][:, stable_dims])
        y_list.append(y2[:n])
        used += 1

    if used == 0:
        raise RuntimeError("No usable files to build training matrix. Check keys and alignment.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    stats = {"used_cases": used, "skipped_cases": skipped}
    return X, y, stats


# ----------------------------
# Metrics
# ----------------------------
def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_h5_dir", required=True, help="Folder of normal (unaugmented) .h5 files.")
    ap.add_argument("--aug_root_dir", required=True, help="Root folder containing augmentation subfolders with .h5 files.")
    ap.add_argument("--out_dir", required=True, help="Output directory (cache + model + results).")

    ap.add_argument("--k", type=int, default=64, help="Number of most stable dimensions to keep.")
    ap.add_argument("--min_aug_views", type=int, default=2, help="Minimum augmentation views per case to compute stability.")
    ap.add_argument("--use_cache", action="store_true", help="Reuse out_dir/stability_cache.npz instead of recomputing stability.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--train_source", choices=["normal", "aug_mean"], default="normal",
                    help="Where to get embeddings for training the MLP.")
    ap.add_argument("--test_size", type=float, default=0.2)

    # MLP knobs
    ap.add_argument("--hidden", type=str, default="1024,512,256,128", help="Comma-separated hidden layer sizes, e.g. '256,128'")
    ap.add_argument("--max_iter", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument(
    "--allowed_augs",
    type=str,
    default="",
    help="Comma-separated augmentation folder names to use (e.g. 'gaussian_L0,gaussian_L1,affine_L0'). "
         "If empty, uses all augmentation folders under --aug_root_dir."
)
    ap.add_argument(
        "--allowed_augs_file",
        type=str,
        default="",
        help="Optional text file with one augmentation folder name per line to use. Overrides --allowed_augs if provided."
    )

    args = ap.parse_args()

    allowed_augs: Optional[set[str]] = None

    if args.allowed_augs_file:
        p = Path(args.allowed_augs_file)
        names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        allowed_augs = set(names)
    elif args.allowed_augs.strip():
        allowed_augs = set([x.strip() for x in args.allowed_augs.split(",") if x.strip()])


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index files
    normal_files = list_h5_flat(args.normal_h5_dir)
    if len(normal_files) == 0:
        raise FileNotFoundError(f"No normal .h5 files found in {args.normal_h5_dir}")

    aug_index = build_aug_index(args.aug_root_dir, allowed_augs=allowed_augs)

    if len(aug_index) == 0:
        raise FileNotFoundError(f"No augmented .h5 files found under {args.aug_root_dir}")

    # Compute/load stability ranking
    if args.use_cache:
        cache = load_stability_cache(out_dir)
        instability = cache["instability"]
        stable_rank = cache["stable_rank"]
        print(f"Loaded stability cache from {out_dir / 'stability_cache.npz'}")
    else:
        cache = compute_and_cache_stability(
            normal_files=normal_files,
            aug_index=aug_index,
            out_dir=out_dir,
            min_aug_views=int(args.min_aug_views),
            seed=int(args.seed),
        )
        instability = cache["instability"]
        stable_rank = cache["stable_rank"]

    D = int(instability.shape[0])
    k = int(args.k)
    if k <= 0:
        k = min(512, D)
    k = min(k, D)

    stable_dims = stable_rank[:k]


    #CHANGE HERE TO GET UNSTABLE IF YOU WANT
    stable_dims = np.sort(stable_dims)

    # Save selected dims explicitly (for convenience)
    np.save(out_dir / "selected_stable_dims.npy", stable_dims)
    np.save(out_dir / "instability_all_dims.npy", instability.astype(np.float32))

    # Build training matrix on selected dims
    X, y, build_stats = build_train_matrix(
        normal_files=normal_files,
        aug_index=aug_index,
        stable_dims=stable_dims,
        train_source=args.train_source,
        min_aug_views=int(args.min_aug_views),
    )

    # Split and train
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(args.alpha),
        batch_size=int(args.batch_size),
        learning_rate_init=1e-3,
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        verbose=False,
    )
    clf.fit(X_trs, y_tr)

    # Evaluate
    if len(np.unique(y_te)) == 2:
        prob = clf.predict_proba(X_tes)[:, 1]
        auc = float(roc_auc_score(y_te, prob))
    else:
        prob = None
        auc = float("nan")

    pred = clf.predict(X_tes)
    acc = float(accuracy_score(y_te, pred))
    f1 = float(f1_score(y_te, pred, zero_division=0))
    sens, spec = sensitivity_specificity(y_te, pred)

    metrics = {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "n_total": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "dims_total": int(D),
        "dims_selected": int(len(stable_dims)),
        "min_aug_views": int(args.min_aug_views),
        "train_source": args.train_source,
        "build_used_cases": build_stats["used_cases"],
        "build_skipped_cases": build_stats["skipped_cases"],
    }

    print("\n=== Stable ranking + MLP results ===")
    for k_, v_ in metrics.items():
        print(f"{k_:>18}: {v_}")

    # Save model artifact
    joblib.dump(
        {"scaler": scaler, "mlp": clf, "stable_dims": stable_dims},
        out_dir / "stable_dims_mlp.joblib",
    )

    config = {
        "normal_h5_dir": args.normal_h5_dir,
        "aug_root_dir": args.aug_root_dir,
        "out_dir": str(out_dir),
        "k": int(args.k),
        "min_aug_views": int(args.min_aug_views),
        "use_cache": bool(args.use_cache),
        "seed": int(args.seed),
        "train_source": args.train_source,
        "test_size": float(args.test_size),
        "hidden": args.hidden,
        "max_iter": int(args.max_iter),
        "alpha": float(args.alpha),
        "batch_size": int(args.batch_size),
        "metrics": metrics,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("Stability ranking (cached) + stable-dim selection + MLP training\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n")

    print(f"\nSaved outputs to: {out_dir}")
    print("Cache file:", out_dir / "stability_cache.npz")


if __name__ == "__main__":
    main()
