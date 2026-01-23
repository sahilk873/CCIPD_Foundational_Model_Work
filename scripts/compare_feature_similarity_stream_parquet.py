#!/usr/bin/env python3
"""
compare_feature_similarity_stream_parquet.py

Stream-compute similarity metrics between baseline H5 features and augmented H5 features,
and write results as a Parquet dataset partitioned by augmentation:

  out_dir/
    augmentation=affine_L0/
      part-00000.parquet
      part-00001.parquet
    augmentation=hsv_L1/
      part-00000.parquet
    ...

Each output row is one tile, containing:
  slide, augmentation, (optional x,y), label (optional), label_valid, cosine, corr, l2, l1

Key features:
- No huge in-memory concat across slides.
- No extra H5 outputs.
- Aligns by coords if both files have coords; else aligns by index up to min length.
- Handles label length mismatch safely (drops invalid labels, keeps label_valid flag).
- Writes Parquet part-files per augmentation for easy “append” (new part files).

Requirements:
  pip install h5py numpy pandas scikit-learn scipy pyarrow
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterator

import h5py
import numpy as np
import pandas as pd

# Parquet backend
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise SystemExit(
        "pyarrow is required for Parquet output.\n"
        "Test with: python3 -c \"import pyarrow; print(pyarrow.__version__)\"\n"
        "Install with: pip install pyarrow\n"
        f"Original error: {e}"
    )


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

        # labels may be mismatched length in the wild; we load if present even if mismatch
        labels_key = find_dataset_by_name(h5, LABEL_CANDIDATES)
        labels = None
        if labels_key is not None:
            arr = _read_dataset(h5, labels_key).squeeze()
            if arr.ndim == 1 and _is_numeric_array(arr):
                labels = arr

        keys_used = {"coords": coords_key or "", "features": feats_key or "", "labels": labels_key or ""}
        return coords, feats, labels, keys_used


# -------------------------
# Alignment (fast, no giant dicts)
# -------------------------
def coords_to_key(coords: np.ndarray) -> np.ndarray:
    coords = coords.astype(np.int64)
    # pack (x,y) into int64 key
    return (coords[:, 0] << 32) | (coords[:, 1] & ((1 << 32) - 1))


def align_by_coords_fast(
    coords_b: np.ndarray, feats_b: np.ndarray, labels_b: Optional[np.ndarray],
    coords_a: np.ndarray, feats_a: np.ndarray, labels_a: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    kb = coords_to_key(coords_b)
    ka = coords_to_key(coords_a)

    # indices in each array for intersection
    common, ib, ia = np.intersect1d(kb, ka, assume_unique=False, return_indices=True)
    if common.size == 0:
        raise RuntimeError("No overlapping coords between the two files.")

    feats_b2 = feats_b[ib]
    feats_a2 = feats_a[ia]

    lb2 = labels_b[ib] if labels_b is not None and labels_b.shape[0] == feats_b.shape[0] else None
    la2 = labels_a[ia] if labels_a is not None and labels_a.shape[0] == feats_a.shape[0] else None

    x = (common >> 32).astype(np.int64)
    y = (common & ((1 << 32) - 1)).astype(np.int64)
    coords2 = np.stack([x, y], axis=1)
    return feats_b2, feats_a2, lb2, la2, coords2


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


# -------------------------
# Matching baseline vs aug file
# -------------------------
def index_h5_by_stem(root: Path, recursive: bool) -> Dict[str, Path]:
    it = root.rglob("*.h5") if recursive else root.glob("*.h5")
    idx: Dict[str, Path] = {}
    for p in it:
        if p.is_file():
            idx[p.stem] = p
    return idx


def resolve_baseline_for_aug(aug_h5: Path, baseline_idx: Dict[str, Path]) -> Optional[Path]:
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
# Parquet dataset writer (partition by augmentation)
# -------------------------
SCHEMA = pa.schema([
    ("slide", pa.string()),
    ("augmentation", pa.string()),
    ("x", pa.int64()),
    ("y", pa.int64()),
    ("label", pa.int8()),          # -1 when missing/invalid
    ("label_valid", pa.bool_()),
    ("cosine", pa.float32()),
    ("corr", pa.float32()),
    ("l2", pa.float32()),
    ("l1", pa.float32()),
])


class AugPartitionWriter:
    def __init__(self, out_root: Path, rows_per_part: int = 2_000_000):
        self.out_root = out_root
        self.rows_per_part = int(rows_per_part)
        self._buffers: Dict[str, List[pd.DataFrame]] = {}
        self._counts: Dict[str, int] = {}
        self._part_idx: Dict[str, int] = {}

    def _aug_dir(self, aug: str) -> Path:
        d = self.out_root / f"augmentation={aug}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def append_df(self, aug: str, df: pd.DataFrame) -> None:
        self._buffers.setdefault(aug, []).append(df)
        self._counts[aug] = self._counts.get(aug, 0) + len(df)
        if self._counts[aug] >= self.rows_per_part:
            self.flush(aug)

    def flush(self, aug: str) -> None:
        bufs = self._buffers.get(aug, [])
        if not bufs:
            return
        df = pd.concat(bufs, axis=0, ignore_index=True)
        self._buffers[aug] = []
        self._counts[aug] = 0

        # enforce schema columns and dtypes
        for col in ["x", "y"]:
            if col not in df.columns:
                df[col] = -1
        if "label" not in df.columns:
            df["label"] = -1
            df["label_valid"] = False
        if "label_valid" not in df.columns:
            df["label_valid"] = df["label"].astype(int) >= 0

        # reorder
        df = df[["slide", "augmentation", "x", "y", "label", "label_valid", "cosine", "corr", "l2", "l1"]]

        # convert to arrow table with explicit schema
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)

        part_i = self._part_idx.get(aug, 0)
        out_path = self._aug_dir(aug) / f"part-{part_i:05d}.parquet"
        pq.write_table(table, out_path, compression="zstd")  # zstd is great for numeric columns
        self._part_idx[aug] = part_i + 1
        print(f"[OK] Wrote {out_path} (rows={table.num_rows})")

    def close(self) -> None:
        for aug in list(self._buffers.keys()):
            self.flush(aug)


# -------------------------
# Pair compute -> DataFrame (streamable)
# -------------------------
def compute_pair_rows(
    baseline_path: Path,
    aug_path: Path,
    augmentation: str,
    label_precedence: str,
    include_coords: bool,
) -> Tuple[pd.DataFrame, Dict]:
    coords_b, feats_b, labels_b, keys_b = load_h5_triplet(baseline_path)
    coords_a, feats_a, labels_a, keys_a = load_h5_triplet(aug_path)

    aligned_by_coords = (coords_b is not None and coords_a is not None)

    if aligned_by_coords:
        feats_b2, feats_a2, lb2, la2, coords2 = align_by_coords_fast(
            coords_b, feats_b, labels_b, coords_a, feats_a, labels_a
        )
        x = coords2[:, 0]
        y = coords2[:, 1]
    else:
        n = int(min(feats_b.shape[0], feats_a.shape[0]))
        feats_b2 = feats_b[:n]
        feats_a2 = feats_a[:n]
        lb2 = labels_b[:n] if labels_b is not None and labels_b.shape[0] >= n else None
        la2 = labels_a[:n] if labels_a is not None and labels_a.shape[0] >= n else None
        x = np.full((n,), -1, dtype=np.int64)
        y = np.full((n,), -1, dtype=np.int64)

    # choose labels carefully (handle mismatch + invalids)
    labels = None
    if lb2 is not None and la2 is not None:
        labels = lb2 if label_precedence == "baseline" else la2
    elif lb2 is not None:
        labels = lb2
    elif la2 is not None:
        labels = la2

    label_valid = np.zeros((feats_b2.shape[0],), dtype=bool)
    label_out = np.full((feats_b2.shape[0],), -1, dtype=np.int8)
    if labels is not None:
        lab = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy()
        label_valid = np.isfinite(lab)
        lab_int = np.where(label_valid, lab.astype(np.int64), -1)
        # clamp to {0,1} if you expect binary; otherwise keep raw but fit into int8 carefully
        lab_int = np.where(label_valid, lab_int, -1)
        lab_int = np.clip(lab_int, -1, 127)
        label_out = lab_int.astype(np.int8)

    # metrics
    cos = cosine_sim(feats_b2, feats_a2).astype(np.float32)
    cor = corr_sim(feats_b2, feats_a2).astype(np.float32)
    l2v = l2_dist(feats_b2, feats_a2).astype(np.float32)
    l1v = l1_dist(feats_b2, feats_a2).astype(np.float32)

    slide = aug_path.stem
    df = pd.DataFrame({
        "slide": slide,
        "augmentation": augmentation,
        "x": x if include_coords else np.full_like(x, -1),
        "y": y if include_coords else np.full_like(y, -1),
        "label": label_out,
        "label_valid": label_valid,
        "cosine": cos,
        "corr": cor,
        "l2": l2v,
        "l1": l1v,
    })

    meta = {
        "baseline_h5": str(baseline_path),
        "aug_h5": str(aug_path),
        "keys_used": {"baseline": keys_b, "aug": keys_a},
        "aligned_by_coords": bool(aligned_by_coords),
        "n_tiles_used": int(len(df)),
        "has_any_labels": bool(label_valid.any()),
    }
    return df, meta


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, help="Directory of baseline (normal) .h5 feature files")
    ap.add_argument("--aug_root", required=True, help="Root directory with augmentation subfolders (none/, affine_L0/, ...)")
    ap.add_argument("--out_dir", required=True, help="Output dataset directory (Parquet partitioned by augmentation)")
    ap.add_argument("--recursive", action="store_true", help="Recursively search baseline_dir and aug_root for .h5 files")
    ap.add_argument("--label_precedence", choices=["baseline", "aug"], default="baseline",
                    help="Which file's labels to use if both present and disagree (default: baseline)")
    ap.add_argument("--require_labels", action="store_true",
                    help="Error if no valid labels exist for an augmentation partition.")
    ap.add_argument("--rows_per_part", type=int, default=2_000_000,
                    help="Flush parquet part-file after this many rows per augmentation.")
    ap.add_argument("--no_coords", action="store_true", help="Do not store x/y coords in output (smaller files).")
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

    writer = AugPartitionWriter(out_root=out_root, rows_per_part=args.rows_per_part)

    master = []
    for aug_folder in aug_folders:
        aug_name = aug_folder.name

        it = aug_folder.rglob("*.h5") if args.recursive else aug_folder.glob("*.h5")
        aug_files = sorted([p for p in it if p.is_file()])
        if not aug_files:
            continue

        print(f"\n==================== AUGMENTATION: {aug_name} ====================")
        print(f"[INFO] Found {len(aug_files)} augmented files in {aug_folder}")

        matched = missing = failed = 0
        any_valid_labels = False
        meta_examples = []

        for aug_h5 in aug_files:
            base_h5 = resolve_baseline_for_aug(aug_h5, baseline_idx)
            if base_h5 is None:
                missing += 1
                print(f"[WARN] No baseline match for: {aug_h5.name}")
                continue

            try:
                df, meta = compute_pair_rows(
                    baseline_path=base_h5,
                    aug_path=aug_h5,
                    augmentation=aug_name,
                    label_precedence=args.label_precedence,
                    include_coords=(not args.no_coords),
                )
                writer.append_df(aug_name, df)
                matched += 1
                any_valid_labels = any_valid_labels or bool(df["label_valid"].any())
                if len(meta_examples) < 3:
                    meta_examples.append(meta)
            except Exception as e:
                failed += 1
                print(f"[ERROR] Failed pair: baseline={base_h5.name} aug={aug_h5.name} err={e}")

        # flush remaining buffered rows for this augmentation
        writer.flush(aug_name)

        if args.require_labels and not any_valid_labels:
            raise SystemExit(f"[{aug_name}] --require_labels set but no valid labels were found in this partition.")

        # write a small per-augmentation meta json (not heavy)
        meta_path = out_root / f"augmentation={aug_name}" / "meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "augmentation": aug_name,
                "aug_folder": str(aug_folder),
                "baseline_dir": str(baseline_dir),
                "matched_pairs": matched,
                "missing_pairs": missing,
                "failed_pairs": failed,
                "any_valid_labels": bool(any_valid_labels),
                "meta_examples_first_3": meta_examples,
                "schema": [str(x) for x in SCHEMA],
            }, f, indent=2)
        print(f"[OK] Wrote {meta_path}")

        master.append({
            "augmentation": aug_name,
            "matched_pairs": matched,
            "missing_pairs": missing,
            "failed_pairs": failed,
            "any_valid_labels": bool(any_valid_labels),
        })

    writer.close()

    master_df = pd.DataFrame(master).sort_values("augmentation")
    master_csv = out_root / "master_summary.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\n[OK] Wrote {master_csv}")
    print(f"[OK] Parquet dataset root: {out_root}")


if __name__ == "__main__":
    main()
