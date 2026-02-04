import os
import glob
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.patches import Patch

from models import DeepMLP, DeepMLPEnsemble


# ------------------------ IO helpers ------------------------
def load_h5_data(h5_path):
    """Returns (coords, features, labels_or_None) from a TRIDENT-style .h5."""
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        features = f["features"][:]
        labels = f["labels"][:] if "labels" in f else None
    return coords, features, labels


def predict_classes_and_probs(model, features, device):
    model.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds, probs


def find_common_files(hopt_dir, musk_dir, conch_dir):
    hopt = {os.path.basename(p) for p in glob.glob(os.path.join(hopt_dir, "*.h5"))}
    musk = {os.path.basename(p) for p in glob.glob(os.path.join(musk_dir, "*.h5"))}
    conch = {os.path.basename(p) for p in glob.glob(os.path.join(conch_dir, "*.h5"))}
    return sorted(hopt & musk & conch)


def _coords_to_keys(coords):
    """coords: (N,2) or (N,>=2). Keep x,y only. Returns list of tuples."""
    coords = np.asarray(coords)
    return [(int(c[0]), int(c[1])) for c in coords]


def align_by_coords(feat_m, coords_m, feat_h, coords_h, feat_c, coords_c):
    """
    Align three feature matrices by coordinate intersection.
    Returns (feat_m_aligned, feat_h_aligned, feat_c_aligned, coords_common)
    """
    km = _coords_to_keys(coords_m)
    kh = _coords_to_keys(coords_h)
    kc = _coords_to_keys(coords_c)

    common = sorted(set(km) & set(kh) & set(kc))
    if not common:
        raise ValueError("No overlapping coords across the three dirs for this slide.")

    im = {k: i for i, k in enumerate(km)}
    ih = {k: i for i, k in enumerate(kh)}
    ic = {k: i for i, k in enumerate(kc)}

    feat_m2 = feat_m[[im[k] for k in common], :]
    feat_h2 = feat_h[[ih[k] for k in common], :]
    feat_c2 = feat_c[[ic[k] for k in common], :]

    coords_common = np.array(common, dtype=np.int64)
    return feat_m2, feat_h2, feat_c2, coords_common


def subset_by_coords(arr, coords_src, coords_keep):
    """
    Subset arr (N, ...) that corresponds to coords_src down to coords_keep (K,2),
    preserving coords_keep order.
    """
    if arr is None:
        return None
    ks = _coords_to_keys(coords_src)
    idx = {k: i for i, k in enumerate(ks)}
    kk = _coords_to_keys(coords_keep)
    take = [idx[k] for k in kk]
    return arr[take, ...] if arr.ndim > 1 else arr[take]


# ------------------------ Coordinate mode detection ------------------------
def detect_coords_mode(coords, tile_size):
    c = np.asarray(coords, dtype=float)
    if len(c) == 0:
        return "topleft"

    rx = (c[:, 0] - c[:, 0].min()) % tile_size
    ry = (c[:, 1] - c[:, 1].min()) % tile_size

    def dist_to_half(x):
        d = np.abs(x - tile_size / 2.0)
        return np.minimum(d, tile_size - d)

    mean_dist_zero = 0.5 * (np.mean(np.minimum(rx, tile_size - rx)) +
                            np.mean(np.minimum(ry, tile_size - ry)))
    mean_dist_half = 0.5 * (np.mean(dist_to_half(rx)) + np.mean(dist_to_half(ry)))

    return "center" if mean_dist_half + 1e-6 < mean_dist_zero else "topleft"


# ------------------------ Overlay (TN transparent, no gaps, correct origin) ------------------------
def overlay_confusion_map(slide_img_rgb, patch_preds, patch_labels, coords,
                          tile_size, coords_mode="topleft", alpha=0.85, pad_px=6,
                          debug_print=False):

    preds = np.asarray(patch_preds, dtype=np.uint8).ravel()
    gts = np.asarray(patch_labels, dtype=np.uint8).ravel()
    coords = np.asarray(coords, dtype=float)
    assert len(preds) == len(gts) == len(coords), "Mismatch among preds/labels/coords."

    H, W, _ = slide_img_rgb.shape
    out = slide_img_rgb.copy()

    # Interpret coords
    if coords_mode == "center":
        tl = coords - tile_size / 2.0
    else:
        tl = coords.copy()

    # Map coords to grid indices, then to thumbnail pixels
    min_x, min_y = float(np.min(tl[:, 0])), float(np.min(tl[:, 1]))
    col_idx = np.rint((tl[:, 0] - min_x) / tile_size).astype(int)
    row_idx = np.rint((tl[:, 1] - min_y) / tile_size).astype(int)

    n_cols = int(col_idx.max()) + 1
    n_rows = int(row_idx.max()) + 1

    cell_w = W / max(n_cols, 1)
    cell_h = H / max(n_rows, 1)

    if debug_print:
        print(f"[DEBUG] grid: {n_cols}x{n_rows}, image: {W}x{H}, cell: {cell_w:.3f}x{cell_h:.3f}, mode={coords_mode}")

    tp = (preds == 1) & (gts == 1)
    fn = (preds == 0) & (gts == 1)
    fp = (preds == 1) & (gts == 0)

    COLORS = {"TP": (0, 255, 0), "FN": (255, 0, 0), "FP": (0, 0, 255)}

    def draw_filled_alpha(img, x1, y1, x2, y2, color, a, pad=0):
        x1 = max(0, int(np.floor(x1)) - pad)
        y1 = max(0, int(np.floor(y1)) - pad)
        x2 = min(W, int(np.ceil(x2)) + pad)
        y2 = min(H, int(np.ceil(y2)) + pad)
        if x1 >= x2 or y1 >= y2:
            return
        region = img[y1:y2, x1:x2]
        color_patch = np.full_like(region, color, dtype=np.uint8)
        blended = cv2.addWeighted(region, 1 - a, color_patch, a, 0)
        img[y1:y2, x1:x2] = blended

    for i in range(len(col_idx)):
        x1, y1 = col_idx[i] * cell_w, row_idx[i] * cell_h
        x2, y2 = (col_idx[i] + 1) * cell_w, (row_idx[i] + 1) * cell_h

        if tp[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["TP"], alpha, pad=pad_px)
        elif fn[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["FN"], alpha, pad=pad_px)
        elif fp[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["FP"], alpha, pad=pad_px)

    legend = [
        Patch(color=np.array(COLORS["TP"]) / 255.0, label="TP (pred=1, gt=1)"),
        Patch(color=np.array(COLORS["FN"]) / 255.0, label="FN (pred=0, gt=1)"),
        Patch(color=np.array(COLORS["FP"]) / 255.0, label="FP (pred=1, gt=0)"),
    ]
    return out, legend


# ------------------------ Metrics helpers ------------------------
def tpr_and_count(preds, labels):
    y = labels.astype(int).ravel()
    p = preds.astype(int).ravel()
    pos_mask = (y == 1)
    pos = int(pos_mask.sum())
    if pos == 0:
        return None, 0, 0
    tp = int(((p == 1) & pos_mask).sum())
    tpr = tp / pos
    return tpr, tp, pos


# ------------------------ Main ------------------------
def main(args):
    plt.rcParams["font.size"] = 16
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    common_files = find_common_files(args.hopt_dir, args.musk_dir, args.conch_dir)
    if not common_files:
        raise ValueError("No common .h5 files across hoptimus/musk/conch directories.")
    if args.max_overlays is not None:
        common_files = common_files[:min(args.max_overlays, len(common_files))]

    os.makedirs(args.save_dir, exist_ok=True)

    # Infer input dims
    sample = common_files[0]
    _, f_h, _ = load_h5_data(os.path.join(args.hopt_dir, sample))
    _, f_m, _ = load_h5_data(os.path.join(args.musk_dir, sample))
    _, f_c, _ = load_h5_data(os.path.join(args.conch_dir, sample))
    in_hopt, in_musk, in_conch = f_h.shape[1], f_m.shape[1], f_c.shape[1]
    ensemble_in = in_hopt + in_musk + in_conch
    print(f"[INFO] Feature dims — Hopt: {in_hopt}, Musk: {in_musk}, Conch: {in_conch} (Ensemble: {ensemble_in})")
    print("[INFO] Ensemble concat order used here: [Musk, Hopt, Conch] — must match training!")

    # Load models
    hopt_model = DeepMLP(input_dim=in_hopt).to(device)
    hopt_model.load_state_dict(torch.load(args.hopt_model, map_location=device))
    hopt_model.eval()

    ensemble_model = DeepMLPEnsemble(input_dim=ensemble_in).to(device)
    ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    ensemble_model.eval()

    # ---------- PASS 1: rank slides ----------
    ranking = []
    for base in common_files:
        hopt_path = os.path.join(args.hopt_dir, base)
        musk_path = os.path.join(args.musk_dir, base)
        conch_path = os.path.join(args.conch_dir, base)

        coords_h, feat_h, labels = load_h5_data(hopt_path)
        if labels is None:
            continue
        coords_m, feat_m, _ = load_h5_data(musk_path)
        coords_c, feat_c, _ = load_h5_data(conch_path)

        # Align features by coordinate intersection (fixes mismatched N)
        try:
            feat_m_a, feat_h_a, feat_c_a, coords_a = align_by_coords(
                feat_m, coords_m,
                feat_h, coords_h,
                feat_c, coords_c
            )
        except ValueError:
            # No overlap (rare) -> skip
            continue

        # Subset labels to aligned coords (labels correspond to coords_h originally)
        labels_a = subset_by_coords(labels, coords_h, coords_a)

        # Predict on aligned arrays
        hopt_preds, _ = predict_classes_and_probs(hopt_model, feat_h_a, device)
        concat_feats = np.concatenate([feat_m_a, feat_h_a, feat_c_a], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        tpr_h, tp_h, pos = tpr_and_count(hopt_preds, labels_a)
        tpr_e, tp_e, _ = tpr_and_count(ensemble_preds, labels_a)
        if tpr_h is None or tpr_e is None:
            continue

        ranking.append({
            "base": base,
            "tpr_hopt": tpr_h, "tp_hopt": tp_h, "pos": pos,
            "tpr_ens": tpr_e, "tp_ens": tp_e,
            "delta_tpr": (tpr_e - tpr_h)
        })

    if not ranking:
        raise ValueError("No slides with positive GT patches to rank by TP (after alignment).")

    # Rank according to user argument
    if args.rank_by == "tp_ensemble":
        ranking.sort(key=lambda r: r["tpr_ens"], reverse=True)
    elif args.rank_by == "tp_delta":
        ranking.sort(key=lambda r: r["delta_tpr"], reverse=True)
    else:  # tp_count
        ranking.sort(key=lambda r: r["tp_ens"], reverse=True)

    selected = ranking[:min(args.top_k, len(ranking))]

    print(f"\n[INFO] Top slides by {args.rank_by}:")
    for r in selected:
        print(f"  {r['base']}: Ens TPR={r['tpr_ens']:.3f}, Hopt={r['tpr_hopt']:.3f}, Δ={r['delta_tpr']:+.3f}")

    # ---------- PASS 2: render overlays ----------
    for r in selected:
        base = r["base"]
        hopt_path = os.path.join(args.hopt_dir, base)
        musk_path = os.path.join(args.musk_dir, base)
        conch_path = os.path.join(args.conch_dir, base)

        coords_h, feat_h, labels = load_h5_data(hopt_path)
        coords_m, feat_m, _ = load_h5_data(musk_path)
        coords_c, feat_c, _ = load_h5_data(conch_path)
        if labels is None:
            continue

        # Align features and labels
        try:
            feat_m_a, feat_h_a, feat_c_a, coords_a = align_by_coords(
                feat_m, coords_m,
                feat_h, coords_h,
                feat_c, coords_c
            )
        except ValueError:
            print(f"[WARN] {base}: no overlapping coords — skipping overlay.")
            continue

        labels_a = subset_by_coords(labels, coords_h, coords_a)

        # Coord mode detection (use aligned coords)
        coords_mode_final = args.coords_mode
        if args.coords_mode == "auto":
            coords_mode_final = detect_coords_mode(coords_a, args.tile_size)

        # Slide thumbnail path
        slide_jpg = os.path.join(args.slides_dir, os.path.splitext(base)[0] + ".jpg")
        if not os.path.exists(slide_jpg):
            print(f"[ERROR] Missing slide: {slide_jpg}")
            continue

        slide_img = np.array(Image.open(slide_jpg).convert("RGB"))

        # Predict on aligned arrays
        hopt_preds, _ = predict_classes_and_probs(hopt_model, feat_h_a, device)
        concat_feats = np.concatenate([feat_m_a, feat_h_a, feat_c_a], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        ensemble_overlay, legend_e = overlay_confusion_map(
            slide_img, ensemble_preds, labels_a, coords_a,
            args.tile_size, coords_mode_final, args.alpha
        )
        hopt_overlay, _ = overlay_confusion_map(
            slide_img, hopt_preds, labels_a, coords_a,
            args.tile_size, coords_mode_final, args.alpha
        )

        # Keep your original rotation behavior
        ensemble_overlay = np.rot90(ensemble_overlay, k=3)
        hopt_overlay = np.rot90(hopt_overlay, k=3)

        fig = plt.figure(figsize=(22, 11))
        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[40, 2], hspace=0.02, wspace=0.02)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        leg_ax = fig.add_subplot(gs[1, :])

        ax0.imshow(ensemble_overlay)
        ax0.set_title(f"Ensemble (TPR={r['tpr_ens']:.3f}, Δ={r['delta_tpr']:+.3f})", fontsize=22)
        ax0.axis("off")

        ax1.imshow(hopt_overlay)
        ax1.set_title(f"Hopt-only (TPR={r['tpr_hopt']:.3f})", fontsize=22)
        ax1.axis("off")

        leg_ax.axis("off")
        leg_ax.legend(handles=legend_e, loc="center", ncol=3, frameon=True, fontsize=16)

        out_path = os.path.join(args.save_dir, f"{os.path.splitext(base)[0]}_overlay.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved overlay → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hopt_dir", required=True)
    parser.add_argument("--musk_dir", required=True)
    parser.add_argument("--conch_dir", required=True)
    parser.add_argument("--slides_dir", required=True)

    # Accept both names so your older commands still work
    parser.add_argument("--hopt_model", "--musk_model", dest="hopt_model", required=True)
    parser.add_argument("--ensemble_model", required=True)

    parser.add_argument("--tile_size", type=int, required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--rank_by", choices=["tp_count", "tp_ensemble", "tp_delta"], default="tp_delta")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_overlays", type=int, default=None)

    parser.add_argument("--coords_mode", choices=["auto", "topleft", "center"], default="auto")
    parser.add_argument("--alpha", type=float, default=0.85)

    parser.add_argument("--device", default="cuda")  # "cuda" or "cpu"

    args = parser.parse_args()
    main(args)
