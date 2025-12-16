#!/usr/bin/env python3
# dice_attention.py

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def extract_tile_id(name: str) -> Optional[int]:
    """Extract numeric tile ID from filename, e.g. tile17_L-1_Havg.png -> 17."""
    m = re.search(r"tile(\d+)", name)
    return int(m.group(1)) if m else None


def load_gray_01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def resize_to(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    H, W = size_hw
    pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((W, H), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return out


def try_otsu_threshold(arr: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(arr))
    except Exception:
        # Manual Otsu fallback (256 bins)
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
    # numeric
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


def build_tile_map(dirp: Path) -> Dict[int, Path]:
    """
    Map tile_id -> filepath.
    If multiple files match same tile_id, choose lexicographically first.
    """
    picks: Dict[int, List[Path]] = {}
    for p in dirp.iterdir():
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


def auto_out_name(dirA: Path, dirB: Path, fg: str, hard: str) -> str:
    a = dirA.name
    b = dirB.name
    fg_tag = fg.replace(".", "p").replace(" ", "")
    hard_tag = hard.replace(".", "p").replace(" ", "")
    return f"dice_{a}_vs_{b}_fg{fg_tag}_hard{hard_tag}.csv"


def parse_args():
    ap = argparse.ArgumentParser(description="Compute Dice similarity between attention maps in two directories.")
    ap.add_argument("--dirA", type=str, required=True, help="Directory A (reference).")
    ap.add_argument("--dirB", type=str, required=True, help="Directory B (resized to A).")
    ap.add_argument("--out", type=str, default="", help="Output CSV path. If omitted, auto-named in CWD.")
    ap.add_argument("--soft", action="store_true", help="Compute soft Dice on continuous maps.")
    ap.add_argument("--hard", action="store_true", help="Compute hard Dice on thresholded masks.")
    ap.add_argument("--hard-thresh", type=str, default="otsu",
                    help="Hard Dice threshold: 'otsu', 'pXX', or numeric like 0.35 (applied per image).")
    ap.add_argument("--fg-thresh", type=str, default="p10",
                    help="Foreground mask threshold: 'pXX' or numeric. Foreground = (A>=thrA) OR (B>=thrB).")
    ap.add_argument("--limit", type=int, default=0, help="Max number of matched tile IDs (0 = all).")
    return ap.parse_args()


def main():
    args = parse_args()
    dirA = Path(args.dirA)
    dirB = Path(args.dirB)

    if not dirA.is_dir() or not dirB.is_dir():
        raise SystemExit("dirA and dirB must both be directories.")

    mapA = build_tile_map(dirA)
    mapB = build_tile_map(dirB)

    common_ids = sorted(set(mapA.keys()) & set(mapB.keys()))
    if not common_ids:
        raise SystemExit("No matching tile IDs found (filenames must contain 'tile<number>').")

    if args.limit > 0:
        common_ids = common_ids[:args.limit]

    if not args.soft and not args.hard:
        raise SystemExit("Select at least one of --soft or --hard.")

    out_csv = Path(args.out) if args.out.strip() else Path(auto_out_name(dirA, dirB, args.fg_thresh, args.hard_thresh))

    rows: List[dict] = []
    soft_scores: List[float] = []
    hard_scores: List[float] = []

    for tid in common_ids:
        pa = mapA[tid]
        pb = mapB[tid]

        A = load_gray_01(pa)
        B = load_gray_01(pb)
        B = resize_to(B, A.shape)

        # Foreground mask from each image independently, then union
        thrA_fg = parse_thresh_spec(A, args.fg_thresh)
        thrB_fg = parse_thresh_spec(B, args.fg_thresh)
        fg_mask = (A >= thrA_fg) | (B >= thrB_fg)

        row = {
            "tile_id": tid,
            "fileA": pa.name,
            "fileB": pb.name,
            "H": int(A.shape[0]),
            "W": int(A.shape[1]),
            "fg_thresh_A": float(thrA_fg),
            "fg_thresh_B": float(thrB_fg),
            "fg_frac": float(fg_mask.mean()),
        }

        if args.soft:
            s = soft_dice(A, B, mask=fg_mask)
            row["soft_dice"] = s
            soft_scores.append(s)

        if args.hard:
            A_bin, thrA = binarize(A, args.hard_thresh)
            B_bin, thrB = binarize(B, args.hard_thresh)
            hd = hard_dice(A_bin, B_bin, mask=fg_mask)
            row["hard_dice"] = hd
            row["hard_thr_A"] = thrA
            row["hard_thr_B"] = thrB
            hard_scores.append(hd)

        rows.append(row)

    # Write CSV
    fieldnames = ["tile_id", "fileA", "fileB", "H", "W", "fg_thresh_A", "fg_thresh_B", "fg_frac"]
    if args.soft:
        fieldnames.append("soft_dice")
    if args.hard:
        fieldnames += ["hard_dice", "hard_thr_A", "hard_thr_B"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    def summ(xs: List[float]) -> str:
        if not xs:
            return "n/a"
        a = np.asarray(xs, dtype=np.float64)
        std = a.std(ddof=1) if len(a) > 1 else 0.0
        return f"mean={a.mean():.4f}, median={np.median(a):.4f}, std={std:.4f}, n={len(a)}"

    print(f"Saved: {out_csv}")
    if args.soft:
        print("Soft Dice:", summ(soft_scores))
    if args.hard:
        print("Hard Dice:", summ(hard_scores))


if __name__ == "__main__":
    main()
