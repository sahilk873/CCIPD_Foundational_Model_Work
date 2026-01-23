#!/usr/bin/env python3
"""
build_stability_embeddings.py

MODIFIED:
- Copies the baseline feature matrix (whatever dataset is detected as "features")
  into the output stability .h5 as dataset name: "features"
  (aligned + filtered to the same coords/valid-label mask used for stability outputs)

Everything else unchanged.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np


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
    for key in h5.keys():
        arr = _read_dataset(h5, key).squeeze()
        if arr.ndim != 1 or not _is_numeric_array(arr):
            continue
        if n_expected is not None and arr.shape[0] != n_expected:
            continue
        arr_f = arr.astype(np.float64)
        arr_f = arr_f[np.isfinite(arr_f)]
        if arr_f.size == 0:
            continue
        uniq = np.unique(arr_f)
        if len(uniq) <= 10:
            return key
    return None


def load_h5_triplet(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Backward-compatible helper used for augmented files."""
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
            labels = _read_dataset(h5, labels_key).squeeze()
            labels = labels.astype(np.float32) if _is_numeric_array(labels) else None

        return coords, feats, labels


def load_baseline_with_key(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], str]:
    """
    Load baseline like load_h5_triplet, but also return the detected feature key
    so we can copy baseline features into the output.
    """
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
            labels = _read_dataset(h5, labels_key).squeeze()
            labels = labels.astype(np.float32) if _is_numeric_array(labels) else None

        return coords, feats, labels, feats_key


# -------------------------
# Alignment (coords preferred)
# -------------------------
def coords_to_key(coords: np.ndarray) -> np.ndarray:
    coords_u = coords.astype(np.uint64, copy=False)
    return (coords_u[:, 0] << np.uint64(32)) | coords_u[:, 1]


def align_by_coords(
    coords_ref: np.ndarray,
    feats_ref: np.ndarray,
    labels_ref: Optional[np.ndarray],
    coords_other: np.ndarray,
    feats_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Align 'other' to 'ref' by intersecting coords. Returns:
      ref_feats_aligned, other_feats_aligned, ref_labels_aligned, coords_aligned
    """
    key_r = coords_to_key(coords_ref)
    key_o = coords_to_key(coords_other)

    if np.unique(key_r).shape[0] != key_r.shape[0]:
        raise RuntimeError("Reference coords contain duplicates; cannot align safely.")
    if np.unique(key_o).shape[0] != key_o.shape[0]:
        raise RuntimeError("Other coords contain duplicates; cannot align safely.")

    sr = np.argsort(key_r)
    so = np.argsort(key_o)
    key_r_sorted = key_r[sr]
    key_o_sorted = key_o[so]

    common, ir, io = np.intersect1d(
        key_r_sorted, key_o_sorted, assume_unique=True, return_indices=True
    )
    if common.size == 0:
        raise RuntimeError("No overlapping coords between files.")

    idx_r = sr[ir]
    idx_o = so[io]

    feats_r2 = feats_ref[idx_r]
    feats_o2 = feats_other[idx_o]
    labels_r2 = labels_ref[idx_r] if labels_ref is not None else None

    x = (common >> np.uint64(32)).astype(np.int64)
    y = (common & np.uint64((1 << 32) - 1)).astype(np.int64)
    coords2 = np.stack([x, y], axis=1)

    return feats_r2, feats_o2, labels_r2, coords2


# -------------------------
# File indexing + matching
# -------------------------
def index_h5_by_stem(root: Path, recursive: bool) -> Dict[str, Path]:
    it = root.rglob("*.h5") if recursive else root.glob("*.h5")
    idx: Dict[str, Path] = {}
    for p in it:
        if p.is_file():
            idx[p.stem] = p
    return idx


def parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def choose_aug_folders(
    aug_root: Path, include: List[str], exclude: List[str], levels: List[str]
) -> List[Path]:
    all_folders = sorted([p for p in aug_root.iterdir() if p.is_dir()])
    if not all_folders:
        return []

    def norm(x: str) -> str:
        return x.strip().lower()

    include_set = {norm(x) for x in include}
    exclude_set = {norm(x) for x in exclude}
    levels_set = {norm(x) for x in levels}

    chosen = []
    for p in all_folders:
        n = norm(p.name)
        if include_set and n not in include_set:
            continue
        if exclude_set and n in exclude_set:
            continue
        if levels_set and not any(tok in n for tok in levels_set):
            continue
        chosen.append(p)

    return chosen


# -------------------------
# Stability computation
# -------------------------
def iqr_along_axis0(X: np.ndarray) -> np.ndarray:
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    return q75 - q25


def compute_stability_vectors(
    stack_kd: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = stack_kd.mean(axis=0)
    sd = stack_kd.std(axis=0, ddof=0)
    iq = iqr_along_axis0(stack_kd)
    cv = sd / (np.abs(mu) + eps)
    return mu, sd, iq, cv


def valid_binary_labels(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if labels is None:
        return None
    y = labels.astype(np.float32, copy=False).squeeze()
    y2 = y.copy()
    mask01 = np.isin(y2, [0.0, 1.0])
    y2[~mask01] = np.nan
    return y2


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, help="Directory of baseline (normal) .h5 feature files")
    ap.add_argument("--aug_root", required=True, help="Root directory with augmentation subfolders (none/, affine_L0/, ...)")
    ap.add_argument("--out_dir", required=True, help="Output directory to write new stability .h5 files")
    ap.add_argument("--recursive", action="store_true", help="Recursively search baseline_dir and augmentation subfolders for .h5 files")

    ap.add_argument("--aug_include", default=None,
                    help='Comma-separated augmentation folder names to include (exact match). Example: "hsv_L1,elastic_L1,he"')
    ap.add_argument("--aug_exclude", default=None,
                    help='Comma-separated augmentation folder names to exclude. Example: "jpeg_L2,noise_L2"')
    ap.add_argument("--levels", default=None,
                    help='Comma-separated tokens; only include augmentation folders whose name contains any token. Example: "L1,L2"')

    ap.add_argument("--require_all_augs", action="store_true",
                    help="If set, skip a slide unless ALL selected augmentation files exist for that slide.")
    ap.add_argument("--min_k", type=int, default=2,
                    help="Minimum number of augmentations (K) required for a patch to compute stability. Default=2.")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for CV denominator. Default=1e-6.")
    ap.add_argument("--compression", type=int, default=4, help="gzip compression level for output H5 datasets (0-9). Default=4.")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    aug_root = Path(args.aug_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_dir.is_dir():
        raise SystemExit(f"--baseline_dir not found: {baseline_dir}")
    if not aug_root.is_dir():
        raise SystemExit(f"--aug_root not found: {aug_root}")

    include = parse_csv_list(args.aug_include)
    exclude = parse_csv_list(args.aug_exclude)
    levels = parse_csv_list(args.levels)

    aug_folders = choose_aug_folders(aug_root, include=include, exclude=exclude, levels=levels)
    if not aug_folders:
        raise SystemExit("No augmentation subfolders selected (check --aug_include/--aug_exclude/--levels).")

    print("========== STABILITY BUILD ==========")
    print(f"[INFO] aug_root: {aug_root}")
    print(f"[INFO] baseline_dir: {baseline_dir}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] Selected augmentation folders ({len(aug_folders)}):")
    for p in aug_folders:
        print(f"  - {p.name}")
    print("=====================================")

    baseline_idx = index_h5_by_stem(baseline_dir, recursive=args.recursive)
    if not baseline_idx:
        raise SystemExit(f"No baseline .h5 files found under {baseline_dir}")

    baseline_files = sorted(baseline_idx.values())

    slides_done = 0
    slides_skipped = 0
    total_patches_written = 0

    for bpath in baseline_files:
        stem = bpath.stem
        out_path = out_dir / f"{stem}.h5"
        try:
            # MOD: load baseline feats + remember which key it came from
            coords_b, feats_b, labels_b, feats_key_b = load_baseline_with_key(bpath)
            labels_b = valid_binary_labels(labels_b)
            if labels_b is None:
                print(f"[SKIP] {stem}: baseline has no labels dataset")
                slides_skipped += 1
                continue

            aligned_feats = []
            aligned_coords = None
            aligned_labels = None
            aligned_baseline_feats = None  # MOD: baseline feats aligned to same coords/mask

            missing_aug = 0
            used_aug = 0

            for afolder in aug_folders:
                aug_h5 = afolder / f"{stem}.h5"
                if not aug_h5.exists() and args.recursive:
                    hits = list(afolder.rglob(f"{stem}.h5"))
                    aug_h5 = hits[0] if hits else aug_h5

                if not aug_h5.exists():
                    missing_aug += 1
                    if args.require_all_augs:
                        break
                    else:
                        continue

                try:
                    coords_a, feats_a, _labels_a = load_h5_triplet(aug_h5)

                    if feats_a.shape[1] != feats_b.shape[1]:
                        raise RuntimeError(f"Feature dim mismatch D={feats_b.shape[1]} vs {feats_a.shape[1]}")

                    if coords_b is not None and coords_a is not None:
                        fb2, fa2, lb2, coords2 = align_by_coords(coords_b, feats_b, labels_b, coords_a, feats_a)

                        if aligned_coords is None:
                            aligned_coords = coords2
                            aligned_labels = lb2
                            aligned_baseline_feats = fb2.astype(np.float32, copy=False)  # MOD
                        else:
                            if coords2.shape != aligned_coords.shape or not np.array_equal(coords2, aligned_coords):
                                raise RuntimeError("Coords mismatch vs previously aligned augmentation (inconsistent overlap).")

                        aligned_feats.append(fa2.astype(np.float32, copy=False))
                        used_aug += 1
                    else:
                        # index alignment fallback
                        n = min(feats_b.shape[0], feats_a.shape[0])
                        fb2 = feats_b[:n]
                        fa2 = feats_a[:n]
                        lb2 = labels_b[:n]

                        if aligned_coords is None:
                            aligned_coords = coords_b[:n] if coords_b is not None else None
                            aligned_labels = lb2
                            aligned_baseline_feats = fb2.astype(np.float32, copy=False)  # MOD
                        else:
                            if aligned_labels.shape[0] != n:
                                raise RuntimeError("Index-alignment size mismatch across augmentations.")

                        aligned_feats.append(fa2.astype(np.float32, copy=False))
                        used_aug += 1

                except Exception as e:
                    print(f"[WARN] {stem}: skipping aug {afolder.name} due to error: {e}")
                    if args.require_all_augs:
                        missing_aug += 1
                        break
                    continue

            if args.require_all_augs and (missing_aug > 0 or used_aug != len(aug_folders)):
                print(f"[SKIP] {stem}: missing/failed augmentations (require_all_augs set).")
                slides_skipped += 1
                continue

            if used_aug < args.min_k:
                print(f"[SKIP] {stem}: only K={used_aug} augmentations available (< min_k={args.min_k}).")
                slides_skipped += 1
                continue

            if aligned_labels is None or aligned_baseline_feats is None:
                print(f"[SKIP] {stem}: alignment produced no data.")
                slides_skipped += 1
                continue

            y = aligned_labels.astype(np.float32, copy=False).squeeze()
            valid = np.isfinite(y) & np.isin(y, [0.0, 1.0])
            if valid.sum() == 0:
                print(f"[SKIP] {stem}: no valid binary labels after alignment.")
                slides_skipped += 1
                continue

            # Stack augmentations: list of (N,D) -> (K,N,D)
            F = np.stack(aligned_feats, axis=0)  # (K, N, D)
            K, N, D = F.shape

            # Keep only valid-labeled patches
            Fv = F[:, valid, :]                    # (K, Nv, D)
            yv = y[valid].astype(np.int64)         # (Nv,)
            coords_v = aligned_coords[valid] if aligned_coords is not None else None
            base_feats_v = aligned_baseline_feats[valid]  # MOD: (Nv, D)

            Nv = Fv.shape[1]

            mean_out = np.empty((Nv, D), dtype=np.float32)
            std_out = np.empty((Nv, D), dtype=np.float32)
            iqr_out = np.empty((Nv, D), dtype=np.float32)
            cv_out = np.empty((Nv, D), dtype=np.float32)

            for i in range(Nv):
                stack_kd = Fv[:, i, :]  # (K,D)
                mu, sd, iq, cv = compute_stability_vectors(stack_kd, eps=args.eps)
                mean_out[i] = mu.astype(np.float32, copy=False)
                std_out[i] = sd.astype(np.float32, copy=False)
                iqr_out[i] = iq.astype(np.float32, copy=False)
                cv_out[i] = cv.astype(np.float32, copy=False)

            # Write output .h5
            with h5py.File(out_path, "w") as h5o:
                comp = "gzip" if args.compression > 0 else None
                comp_lvl = args.compression if args.compression > 0 else None

                # MOD: copy baseline features into output (aligned + filtered)
                h5o.create_dataset("features", data=base_feats_v, compression=comp, compression_opts=comp_lvl)

                h5o.create_dataset("mean", data=mean_out, compression=comp, compression_opts=comp_lvl)
                h5o.create_dataset("std", data=std_out, compression=comp, compression_opts=comp_lvl)
                h5o.create_dataset("iqr", data=iqr_out, compression=comp, compression_opts=comp_lvl)
                h5o.create_dataset("cv", data=cv_out, compression=comp, compression_opts=comp_lvl)
                h5o.create_dataset("label", data=yv.astype(np.int64))

                if coords_v is not None:
                    h5o.create_dataset("coords", data=coords_v.astype(np.int64))

                # helpful metadata
                h5o.attrs["K_used"] = int(used_aug)
                h5o.attrs["augmentations_used"] = ",".join([p.name for p in aug_folders])

                # extra helpful metadata: what baseline key we detected
                h5o.attrs["baseline_features_key_detected"] = str(feats_key_b)

            slides_done += 1
            total_patches_written += int(Nv)
            print(f"[OK] {stem}: wrote {out_path.name} (patches={Nv}, D={D}, K={used_aug})")

        except Exception as e:
            print(f"[SKIP] {stem}: error: {e}")
            slides_skipped += 1
            continue

    print("\n========== DONE ==========")
    print(f"slides_written  : {slides_done}")
    print(f"slides_skipped  : {slides_skipped}")
    print(f"patches_written : {total_patches_written}")
    print("==========================\n")


if __name__ == "__main__":
    main()
