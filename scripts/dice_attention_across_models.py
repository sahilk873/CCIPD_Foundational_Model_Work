#!/usr/bin/env python3
# dice_attention_across_models.py
#
# You have:
#   /scratch/.../conch_attn_maps/<SLIDE_FOLDER>/*.png
#   /scratch/.../musk_attn_maps/<SLIDE_FOLDER>/*.png
#   /scratch/.../hoptimus_attn_maps/<SLIDE_FOLDER>/*.png
#
# This script compares attention-map PNGs ACROSS MODELS, matching by:
#   - slide folder name (must exist under both model roots)
#   - tile id parsed from filename "tile<number>"
#   - optional --pattern to force same layer/head file selection, e.g. "*_L-1_Havg.png"
#
# Outputs:
#   out_dir/
#     pair_summaries.csv               (one row per slide per model-pair)
#     (optional) tiles_<pair>__<slide>.csv   (tile-level rows per slide per pair)

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
from PIL import Image


# -----------------------------
# Filename parsing / IO
# -----------------------------
_TILE_RE = re.compile(r"tile(\d+)", re.IGNORECASE)

def extract_tile_id(name: str) -> Optional[int]:
    m = _TILE_RE.search(name)
    return int(m.group(1)) if m else None


def load_gray_01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_to(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    H, W = size_hw
    pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((W, H), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


# -----------------------------
# Thresholding / Dice
# -----------------------------
def try_otsu_threshold(arr: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(arr))
    except Exception:
        hist, bin_edges = np.histogram(arr, bins=256, range=(0.0, 1.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
        var12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = int(np.argmax(var12))
        return float(bin_centers[idx])


def parse_thresh_spec(arr: np.ndarray, spec: str) -> float:
    spec = spec.strip().lower()
    if spec == "otsu":
        return try_otsu_threshold(arr)
    if spec.startswith("p") and spec[1:].isdigit():
        q = int(spec[1:])
        q = int(np.clip(q, 0, 100))
        return float(np.percentile(arr, q))
    return float(spec)


def binarize(arr: np.ndarray, spec: str) -> Tuple[np.ndarray, float]:
    thr = parse_thresh_spec(arr, spec)
    return (arr >= thr), thr


def soft_dice(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-8) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    num = 2.0 * float(np.sum(a * b))
    den = float(np.sum(a * a) + np.sum(b * b) + eps)
    return num / den


def hard_dice(a_bin: np.ndarray, b_bin: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-8) -> float:
    if mask is not None:
        a_bin = a_bin[mask]
        b_bin = b_bin[mask]
    inter = float(np.logical_and(a_bin, b_bin).sum())
    size = float(a_bin.sum() + b_bin.sum())
    return (2.0 * inter) / (size + eps)


# -----------------------------
# Directory -> tile map
# -----------------------------
def build_tile_map(dirp: Path, pattern: Optional[str]) -> Dict[int, Path]:
    """
    Map tile_id -> file path.
    If multiple files match same tile_id (e.g., multiple layers/heads), choose lexicographically first.
    Use --pattern to force the exact same layer/head across models.
    """
    files: Iterable[Path] = dirp.glob(pattern) if pattern else dirp.iterdir()

    picks: Dict[int, List[Path]] = {}
    for p in files:
        if not p.is_file():
            continue
        tid = extract_tile_id(p.name)
        if tid is None:
            continue
        picks.setdefault(tid, []).append(p)

    out: Dict[int, Path] = {}
    for tid, paths in picks.items():
        out[tid] = sorted(paths, key=lambda x: x.name)[0]
    return out


def list_slide_dirs(root: Path) -> Dict[str, Path]:
    """
    Return mapping: slide_folder_name -> path
    Only includes direct children that are directories.
    """
    return {p.name: p for p in root.iterdir() if p.is_dir()}


def summarise(xs: List[float]) -> Tuple[float, float, float, int]:
    if not xs:
        return (float("nan"), float("nan"), float("nan"), 0)
    a = np.asarray(xs, dtype=np.float64)
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return (float(a.mean()), float(np.median(a)), std, int(len(a)))


# -----------------------------
# Compare one slide folder for one model pair
# -----------------------------
def compare_slide_pair(
    slide_name: str,
    dirA: Path,
    dirB: Path,
    modelA: str,
    modelB: str,
    pattern: Optional[str],
    do_soft: bool,
    do_hard: bool,
    hard_thresh: str,
    fg_thresh: str,
    limit: int,
) -> Tuple[List[dict], List[float], List[float]]:
    mapA = build_tile_map(dirA, pattern)
    mapB = build_tile_map(dirB, pattern)

    common_ids = sorted(set(mapA.keys()) & set(mapB.keys()))
    if not common_ids:
        return [], [], []

    if limit > 0:
        common_ids = common_ids[:limit]

    rows: List[dict] = []
    soft_scores: List[float] = []
    hard_scores: List[float] = []

    for tid in common_ids:
        pa = mapA[tid]
        pb = mapB[tid]

        A = load_gray_01(pa)
        B = load_gray_01(pb)
        B = resize_to(B, A.shape)

        thrA_fg = parse_thresh_spec(A, fg_thresh)
        thrB_fg = parse_thresh_spec(B, fg_thresh)
        fg_mask = (A >= thrA_fg) | (B >= thrB_fg)

        row = {
            "slide": slide_name,
            "modelA": modelA,
            "modelB": modelB,
            "tile_id": tid,
            "fileA": pa.name,
            "fileB": pb.name,
            "H": int(A.shape[0]),
            "W": int(A.shape[1]),
            "fg_thresh_A": float(thrA_fg),
            "fg_thresh_B": float(thrB_fg),
            "fg_frac": float(fg_mask.mean()),
        }

        if do_soft:
            s = soft_dice(A, B, mask=fg_mask)
            row["soft_dice"] = s
            soft_scores.append(s)

        if do_hard:
            A_bin, thrA = binarize(A, hard_thresh)
            B_bin, thrB = binarize(B, hard_thresh)
            hd = hard_dice(A_bin, B_bin, mask=fg_mask)
            row["hard_dice"] = hd
            row["hard_thr_A"] = float(thrA)
            row["hard_thr_B"] = float(thrB)
            hard_scores.append(hd)

        rows.append(row)

    return rows, soft_scores, hard_scores


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare attention-map PNGs across model roots (conch/musk/hoptimus) using Dice."
    )

    ap.add_argument("--conch-root", type=str, required=True, help="Root folder containing per-slide subfolders for CONCH.")
    ap.add_argument("--musk-root", type=str, required=True, help="Root folder containing per-slide subfolders for MUSK.")
    ap.add_argument("--hoptimus-root", type=str, required=True, help="Root folder containing per-slide subfolders for H-optimus.")

    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for CSVs.")
    ap.add_argument("--pattern", type=str, default="",
                    help="Glob to select the same layer/head across models, e.g. '*_L-1_Havg.png'. "
                         "Strongly recommended if you generated multiple overlays per tile.")
    ap.add_argument("--limit", type=int, default=0, help="Max matched tile IDs per slide (0 = all).")

    ap.add_argument("--soft", action="store_true", help="Compute soft Dice on continuous maps.")
    ap.add_argument("--hard", action="store_true", help="Compute hard Dice on thresholded masks.")
    ap.add_argument("--hard-thresh", type=str, default="otsu",
                    help="Hard Dice threshold: 'otsu', 'pXX', or numeric like 0.35 (applied per image).")
    ap.add_argument("--fg-thresh", type=str, default="p10",
                    help="Foreground mask threshold: 'pXX' or numeric. Foreground = (A>=thrA) OR (B>=thrB).")

    ap.add_argument("--write-tile-csvs", action="store_true",
                    help="If set, write tile-level CSV per slide per model-pair (lots of files).")

    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    if not args.soft and not args.hard:
        raise SystemExit("Select at least one of --soft or --hard.")

    conch_root = Path(args.conch_root)
    musk_root = Path(args.musk_root)
    hopt_root = Path(args.hoptimus_root)
    for r in (conch_root, musk_root, hopt_root):
        if not r.is_dir():
            raise SystemExit(f"Not a directory: {r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern.strip() or None

    # Slide folder name -> path
    conch_slides = list_slide_dirs(conch_root)
    musk_slides = list_slide_dirs(musk_root)
    hopt_slides = list_slide_dirs(hopt_root)

    # Only compare slides that exist under ALL three
    common_slides = sorted(set(conch_slides.keys()) & set(musk_slides.keys()) & set(hopt_slides.keys()))
    if not common_slides:
        raise SystemExit("No common slide subfolder names across conch/musk/hoptimus roots.")

    # Model pair definitions
    pairs = [
        ("conch", "musk", conch_slides, musk_slides),
        ("conch", "hoptimus", conch_slides, hopt_slides),
        ("musk", "hoptimus", musk_slides, hopt_slides),
    ]

    # Summary rows: one row per (slide, pair)
    summary_rows: List[dict] = []

    # Optional: single combined tile-level CSV (instead of many files)
    # (kept simple: we do per-slide per-pair if requested)
    for slide in common_slides:
        for modelA, modelB, dictA, dictB in pairs:
            dirA = dictA[slide]
            dirB = dictB[slide]

            tile_rows, soft_scores, hard_scores = compare_slide_pair(
                slide_name=slide,
                dirA=dirA,
                dirB=dirB,
                modelA=modelA,
                modelB=modelB,
                pattern=pattern,
                do_soft=args.soft,
                do_hard=args.hard,
                hard_thresh=args.hard_thresh,
                fg_thresh=args.fg_thresh,
                limit=args.limit,
            )

            if not tile_rows:
                # slide exists, but no matching tile ids under pattern
                continue

            sm, s_med, sstd, sn = (float("nan"), float("nan"), float("nan"), 0)
            hm, hmed, hstd, hn = (float("nan"), float("nan"), float("nan"), 0)

            if args.soft:
                sm, s_med, sstd, sn = summarise(soft_scores)
            if args.hard:
                hm, hmed, hstd, hn = summarise(hard_scores)

            srow = {
                "slide": slide,
                "modelA": modelA,
                "modelB": modelB,
                "pattern": pattern or "",
                "n_tiles": len(tile_rows),
                "fg_thresh": args.fg_thresh,
                "hard_thresh": args.hard_thresh,
            }
            if args.soft:
                srow.update({"soft_mean": sm, "soft_median": s_med, "soft_std": sstd})
            if args.hard:
                srow.update({"hard_mean": hm, "hard_median": hmed, "hard_std": hstd})

            summary_rows.append(srow)

            if args.write_tile_csvs:
                pair_tag = f"{modelA}_vs_{modelB}"
                tile_csv = out_dir / f"tiles__{pair_tag}__{slide}.csv"
                fieldnames = ["slide", "modelA", "modelB", "tile_id", "fileA", "fileB", "H", "W",
                              "fg_thresh_A", "fg_thresh_B", "fg_frac"]
                if args.soft:
                    fieldnames.append("soft_dice")
                if args.hard:
                    fieldnames += ["hard_dice", "hard_thr_A", "hard_thr_B"]

                with tile_csv.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(tile_rows)

    if not summary_rows:
        raise SystemExit(
            "No comparisons produced results. Likely causes:\n"
            "  - slide folder names don't match across roots\n"
            "  - --pattern doesn't match any PNGs in one or more models\n"
            "  - filenames do not contain 'tile<number>'"
        )

    # Write one summary CSV across everything
    summary_csv = out_dir / "pair_summaries.csv"
    base_fields = ["slide", "modelA", "modelB", "pattern", "n_tiles", "fg_thresh", "hard_thresh"]
    extra_fields: List[str] = []
    if args.soft:
        extra_fields += ["soft_mean", "soft_median", "soft_std"]
    if args.hard:
        extra_fields += ["hard_mean", "hard_median", "hard_std"]

    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Saved summary: {summary_csv}")
    if args.write_tile_csvs:
        print(f"Saved tile-level CSVs into: {out_dir}")
    print(f"Compared {len(set(r['slide'] for r in summary_rows))} slides across model pairs.")


if __name__ == "__main__":
    main()
