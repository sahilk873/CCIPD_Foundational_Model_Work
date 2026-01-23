#!/usr/bin/env python3
"""
stability_select_and_train.py  (PyTorch DeepMLP version)

Pipeline
--------
(1) Read NORMAL (unaugmented) .h5 files from --normal_h5_dir.
    - Must contain embeddings (N, D)
    - Must contain labels (N,)
    - Optional coords (N, 2) or similar

(2) Read AUGMENTED .h5 files stored under augmentation subfolders in --aug_root_dir:
      aug_root_dir/
        gaussian_L0/slide_001.h5
        affine_L0/slide_001.h5
        elastic_L1/slide_001.h5
        ...
    Same filename repeated across multiple augmentation folders.

(3) Match normal and aug by exact filename (basename).

(4) Compute per-dimension instability ONCE (streamed over cases):
      instability[j] = mean_over_tiles( var_over_augviews( embedding[tile, aug, j] ) )

    Cache saved to:
      out_dir/stability_cache.npz

(5) Select k most stable dimensions and train a PyTorch DeepMLP.

Gating modes (soft prioritization)
---------------------------------
- gate=none:
    Use base features as-is.

- gate=global:
    Multiply base features by global per-dimension weights derived from instability.

- gate=tile:
    Compute per-tile sigma over augmented views, then gate:
      g(tile, dim) = exp(-alpha * sigma(tile, dim))
    Multiply base features by g.

- gate=global_tile:
    Apply tile gating then global gating.

IMPORTANT NEW WORKFLOW
----------------------
Tile gating (gate=tile or gate=global_tile) now works with BOTH train_source values:

- train_source=aug_mean:
    Base features = mu(tile, dim) = mean over augmented views
    Sigma computed from augmented views
    Then apply tile/global gating.

- train_source=normal:
    Base features = normal FM embedding (unaugmented) per tile
    Sigma computed from augmented views
    Then apply tile/global gating to the NORMAL base features

So tile/global_tile always "see" multiple augmented views (to compute sigma),
but can operate on either the augmented-mean base or the normal base.

Outputs
-------
- stability_cache.npz
- selected_stable_dims.npy
- instability_all_dims.npy
- global_gate_w.npy (if applicable)
- best_model.pth (PyTorch weights)
- stable_dims_mlp.joblib (stores scaler + stable_dims + gating + model_path)
- run_config.json
- summary.txt
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
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
# Aug stack loader (for sigma / mu)
# ----------------------------
def load_aug_stack_for_case(
    fname: str,
    aug_index: Dict[str, List[str]],
    y_ref: np.ndarray,
    coords_y_ref: Optional[np.ndarray],
    D_ref: int,
    min_aug_views: int,
) -> Tuple[np.ndarray, int]:
    """
    Loads augmented views for this case, aligns each view to (y_ref, coords_y_ref),
    and returns:
      Z: (A, n_min, D_ref)
      n_min: int (cropped tile count shared across views and y_ref)
    Raises RuntimeError if insufficient valid views.
    """
    aug_paths = aug_index.get(fname, [])
    if len(aug_paths) < min_aug_views:
        raise RuntimeError("Not enough augmentation files listed for this case.")

    views: List[np.ndarray] = []
    n_min = len(y_ref)

    for p in aug_paths:
        try:
            X_aug, coords_aug, _ = read_embeddings_coords(p)
            Xa, ya = align_X_to_y(X_aug, coords_aug, y_ref, coords_y_ref)
            if Xa.shape[1] != D_ref:
                continue
            n_min = min(n_min, len(ya))
            views.append(Xa)
        except Exception:
            continue

    if len(views) < min_aug_views:
        raise RuntimeError("Not enough valid augmentation views after filtering.")

    views = [v[:n_min] for v in views]
    Z = np.stack(views, axis=0)  # (A, n_min, D)
    return Z, n_min


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
                continue

        if n_min is None:
            n_min = len(y_aligned)
        else:
            n_min = min(n_min, len(y_aligned))

        views.append(X_aligned)

    if D is None or n_min is None or len(views) == 0:
        raise RuntimeError("No valid augmentation views for this case.")

    views = [v[:n_min] for v in views]
    Z = np.stack(views, axis=0)  # (A, n, D)
    instab_case = Z.var(axis=0).mean(axis=0)
    return instab_case, Z.shape[0]


def compute_and_cache_stability(
    normal_files: List[str],
    aug_index: Dict[str, List[str]],
    out_dir: Path,
    min_aug_views: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    instability_sum: Optional[np.ndarray] = None
    used_cases = 0
    skipped_cases = 0
    dim_D: Optional[int] = None

    for nf in tqdm(normal_files, desc="Stability pass (compute)"):
        fname = Path(nf).name
        aug_paths = aug_index.get(fname, [])
        if len(aug_paths) < min_aug_views:
            skipped_cases += 1
            continue

        try:
            _, coords_y, y, _ = read_embeddings_coords_labels(nf)
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
            f"- Do at least some cases have >= --min_aug_views augmentation files? (min_aug_views={min_aug_views})\n"
            f"- Do augmented files contain embeddings under one of keys: {EMBED_KEYS} ?\n"
            f"- Do normal files contain labels under one of keys: {LABEL_KEYS} ?"
        )

    instability = instability_sum / used_cases
    stable_rank = np.argsort(instability)
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
    gate: str,
    gate_alpha: float,
    global_w: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Builds (X, y) across all cases using:
      - base features from train_source (normal or aug_mean)
      - gating from gate mode
    """
    if gate in ("global", "global_tile") and global_w is None:
        raise ValueError(f"--gate {gate} requires global_w, but it was None.")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    used = 0
    skipped = 0

    need_aug_for_sigma = gate in ("tile", "global_tile")
    need_aug_for_mu = train_source == "aug_mean"
    need_aug_stack = need_aug_for_sigma or need_aug_for_mu

    for nf in tqdm(normal_files, desc="Build train matrix"):
        fname = Path(nf).name

        # Load normal (base + labels)
        try:
            X_norm, coords_y, y, _ = read_embeddings_coords_labels(nf)
        except Exception:
            skipped += 1
            continue

        X_norm2, y2 = align_X_to_y(X_norm, coords_y, y, coords_y)

        if X_norm2.ndim != 2 or y2.ndim != 1:
            skipped += 1
            continue

        D = X_norm2.shape[1]

        Z: Optional[np.ndarray] = None
        n_min: Optional[int] = None

        # Load augmented stack if needed (for mu and/or sigma)
        if need_aug_stack:
            try:
                Z_loaded, n_min_loaded = load_aug_stack_for_case(
                    fname=fname,
                    aug_index=aug_index,
                    y_ref=y,
                    coords_y_ref=coords_y,
                    D_ref=D,
                    min_aug_views=min_aug_views,
                )
                Z = Z_loaded
                n_min = n_min_loaded
            except Exception:
                skipped += 1
                continue

        # Decide base features
        if train_source == "normal":
            if n_min is None:
                X_base = X_norm2
            else:
                X_base = X_norm2[:n_min]
                y2 = y2[:n_min]

        elif train_source == "aug_mean":
            if Z is None or n_min is None:
                skipped += 1
                continue
            mu = Z.mean(axis=0)  # (n_min, D)
            X_base = mu
            y2 = y2[:n_min]

        else:
            raise ValueError(f"Unknown train_source: {train_source}")

        # Apply gating
        if gate == "none":
            X_use = X_base

        elif gate == "global":
            X_use = X_base * global_w

        elif gate in ("tile", "global_tile"):
            if Z is None:
                skipped += 1
                continue
            sigma = Z.std(axis=0)  # (n_min, D)
            g = np.exp(-float(gate_alpha) * sigma)
            X_use = X_base * g
            if gate == "global_tile":
                X_use = X_use * global_w

        else:
            raise ValueError(f"Unknown gate: {gate}")

        # Basic sanity and min rows
        n = min(len(y2), X_use.shape[0])
        if n < 10:
            skipped += 1
            continue

        # Apply stable dims selection
        X_list.append(X_use[:n, :][:, stable_dims])
        y_list.append(y2[:n])
        used += 1

    if used == 0:
        raise RuntimeError("No usable files to build training matrix. Check keys, alignment, and augmentation availability.")

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


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sens, spec = sensitivity_specificity(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sensitivity": float(sens),
        "specificity": float(spec),
    }


# ----------------------------
# PyTorch model + training
# ----------------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NumpyTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_model(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

        y_prob.append(probs)
        y_pred.append(preds)
        y_true.append(yb.numpy())

    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_h5_dir", required=True, help="Folder of normal (unaugmented) .h5 files.")
    ap.add_argument("--aug_root_dir", required=True, help="Root folder containing augmentation subfolders with .h5 files.")
    ap.add_argument("--out_dir", required=True, help="Output directory (cache + model + results).")

    ap.add_argument("--k", type=int, default=64, help="Number of most stable dimensions to keep.")
    ap.add_argument("--min_aug_views", type=int, default=2, help="Minimum augmentation views per case to compute stability and/or tile gating.")
    ap.add_argument("--use_cache", action="store_true", help="Reuse out_dir/stability_cache.npz instead of recomputing stability.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--train_source",
        choices=["normal", "aug_mean"],
        default="normal",
        help=(
            "Where to get base embeddings for training. "
            "normal: use unaugmented FM output. "
            "aug_mean: use mean over augmented views."
        ),
    )

    ap.add_argument(
        "--gate",
        choices=["none", "global", "tile", "global_tile"],
        default="tile",
        help=(
            "Soft gating mode. "
            "tile/global_tile compute per-tile sigma from multiple augmented views and gate by exp(-alpha*sigma). "
            "These can operate on either normal base features or aug_mean base features depending on --train_source. "
            "global uses per-dimension weights derived from instability cache."
        ),
    )
    ap.add_argument("--gate_alpha", type=float, default=1.0, help="Alpha for tile gating: exp(-alpha*sigma).")
    ap.add_argument("--gate_eps", type=float, default=1e-8, help="Epsilon for global stability weight computation.")

    ap.add_argument("--test_size", type=float, default=0.2)

    # PyTorch training knobs
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--drop_last", action="store_true", help="Drop last partial batch in train loader.")

    ap.add_argument(
        "--allowed_augs",
        type=str,
        default="",
        help="Comma-separated augmentation folder names to use. If empty, uses all folders."
    )
    ap.add_argument(
        "--allowed_augs_file",
        type=str,
        default="",
        help="Optional text file with one augmentation folder name per line to use. Overrides --allowed_augs if provided."
    )

    ap.add_argument("--save_path", type=str, default="best_model.pth", help="Filename for best model weights (in out_dir).")

    args = ap.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    allowed_augs: Optional[set[str]] = None
    if args.allowed_augs_file:
        p = Path(args.allowed_augs_file)
        names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        allowed_augs = set(names)
    elif args.allowed_augs.strip():
        allowed_augs = set([x.strip() for x in args.allowed_augs.split(",") if x.strip()])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

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

    # Global gating weights (w): higher for more stable dimensions
    global_w: Optional[np.ndarray] = None
    if args.gate in ("global", "global_tile"):
        eps = float(args.gate_eps)
        stability = 1.0 / (instability + eps)
        global_w = stability / (stability.mean() + eps)
        global_w = global_w.astype(np.float32)

    # Select stable dims
    k = int(args.k)
    if k <= 0:
        k = min(512, D)
    k = min(k, D)

    stable_dims = stable_rank[:k]
    stable_dims = np.sort(stable_dims)

    np.save(out_dir / "selected_stable_dims.npy", stable_dims)
    np.save(out_dir / "instability_all_dims.npy", instability.astype(np.float32))
    if global_w is not None:
        np.save(out_dir / "global_gate_w.npy", global_w.astype(np.float32))

    # Build training matrix (stable dims already applied)
    X, y, build_stats = build_train_matrix(
        normal_files=normal_files,
        aug_index=aug_index,
        stable_dims=stable_dims,
        train_source=args.train_source,
        min_aug_views=int(args.min_aug_views),
        gate=args.gate,
        gate_alpha=float(args.gate_alpha),
        global_w=global_w,
    )

    # Split
    '''X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Standardize
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    # Torch datasets
    train_ds = NumpyTensorDataset(X_trs, y_tr)
    val_ds = NumpyTensorDataset(X_tes, y_te)'''

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # NO StandardScaler: use raw features
    X_trs = X_tr
    X_tes = X_te

    # Torch datasets
    train_ds = NumpyTensorDataset(X_trs, y_tr)
    val_ds = NumpyTensorDataset(X_tes, y_te)


    # Class weights + sampler
    train_labels_t = torch.tensor(y_tr, dtype=torch.long)
    class_counts = torch.bincount(train_labels_t, minlength=2).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels_t]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        sampler=sampler,
        drop_last=bool(args.drop_last),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Model + loss + optimizer
    input_dim = int(X_trs.shape[1])
    model = DeepMLP(input_dim=input_dim, hidden_dim=int(args.hidden_dim)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    best_acc = -1.0
    best_path = out_dir / str(args.save_path)

    for epoch in range(int(args.epochs)):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        y_true, y_pred, y_prob = eval_model(model, val_loader, device)
        m = compute_metrics_binary(y_true, y_pred, y_prob)

        print(
            f"\n[Epoch {epoch+1}/{int(args.epochs)}] "
            f"loss={loss:.4f}  acc={m['acc']:.4f}  auc={m['auc']:.4f}  "
            f"f1={m['f1']:.4f}  sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}"
        )

        if m["acc"] > best_acc:
            best_acc = float(m["acc"])
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved best model (acc={best_acc:.4f}) to {best_path}")

    # Final eval with best weights loaded
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    y_true, y_pred, y_prob = eval_model(model, val_loader, device)
    final_metrics = compute_metrics_binary(y_true, y_pred, y_prob)

    metrics = {
        **final_metrics,
        "n_total": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "dims_total": int(D),
        "dims_selected": int(len(stable_dims)),
        "min_aug_views": int(args.min_aug_views),
        "train_source": args.train_source,
        "gate": args.gate,
        "gate_alpha": float(args.gate_alpha),
        "build_used_cases": build_stats["used_cases"],
        "build_skipped_cases": build_stats["skipped_cases"],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "best_model_path": str(best_path),
    }

    print("\n=== Stability ranking + gated PyTorch DeepMLP results (best checkpoint) ===")
    for k_, v_ in metrics.items():
        print(f"{k_:>18}: {v_}")

    # Save artifact (joblib) with everything needed to reproduce inference
    joblib.dump(
        {
            #"scaler": scaler,
            "stable_dims": stable_dims,
            "train_source": args.train_source,
            "gate": args.gate,
            "gate_alpha": float(args.gate_alpha),
            "gate_eps": float(args.gate_eps),
            "global_w": global_w,
            "pytorch_model": "DeepMLP(input_dim=k, hidden_dim=hidden_dim)",
            "hidden_dim": int(args.hidden_dim),
            "state_dict_path": str(best_path),
        },
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
        "gate": args.gate,
        "gate_alpha": float(args.gate_alpha),
        "gate_eps": float(args.gate_eps),
        "test_size": float(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "save_path": str(best_path),
        "metrics": metrics,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("Stability ranking (cached) + stable-dim selection + gating + PyTorch DeepMLP training\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n")

    print(f"\nSaved outputs to: {out_dir}")
    print("Cache file:", out_dir / "stability_cache.npz")
    print("Best weights:", best_path)


if __name__ == "__main__":
    main()
