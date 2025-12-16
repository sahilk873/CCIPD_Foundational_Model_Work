'''#!/usr/bin/env python3
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

from models import DeepMLP, DeepMLPEnsemble  # your models.py


# ------------------------ IO helpers ------------------------

def load_h5_data(h5_path):
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


def find_common_files(conch_dir, musk_dir, hopt_dir):
    conch = {os.path.basename(p) for p in glob.glob(os.path.join(conch_dir, "*.h5"))}
    musk  = {os.path.basename(p) for p in glob.glob(os.path.join(musk_dir,  "*.h5"))}
    hopt  = {os.path.basename(p) for p in glob.glob(os.path.join(hopt_dir,  "*.h5"))}
    return sorted(conch & musk & hopt)


# ------------------------ Coordinate mode detection ------------------------

def detect_coords_mode(coords, tile_size):
    """
    Heuristically detect whether coords are tile TOP-LEFT or CENTER.
    """
    c = np.asarray(coords, dtype=float)
    if len(c) == 0:
        return "topleft"

    rx = (c[:, 0] - c[:, 0].min()) % tile_size
    ry = (c[:, 1] - c[:, 1].min()) % tile_size

    def dist_to_half(x):
        d = np.abs(x - tile_size/2.0)
        return np.minimum(d, tile_size - d)

    mean_dist_zero = 0.5 * (np.mean(np.minimum(rx, tile_size - rx)) +
                            np.mean(np.minimum(ry, tile_size - ry)))
    mean_dist_half = 0.5 * (np.mean(dist_to_half(rx)) + np.mean(dist_to_half(ry)))

    return "center" if mean_dist_half + 1e-6 < mean_dist_zero else "topleft"


# ------------------------ Overlay (TN transparent, no gaps, correct origin) ------------------------

def overlay_confusion_map(
    slide_img_rgb,            # np.uint8 (H, W, 3), RGB
    patch_preds,              # (N,) 0/1
    patch_labels,             # (N,) 0/1
    coords,                   # (N,2) coords in ORIGINAL slide space
    tile_size,                # original patch size (pixels at level-0)
    coords_mode="topleft",    # "topleft" or "center"
    alpha=0.85,
    pad_px=6,                 # expand each edge by this many pixels in JPG space
    debug_print=False,
):
    """
    TN tiles are fully transparent.
    Tiles are snapped to a discrete grid to guarantee no gaps between cells.
    """
    preds  = np.asarray(patch_preds,  dtype=np.uint8).ravel()
    gts    = np.asarray(patch_labels, dtype=np.uint8).ravel()
    coords = np.asarray(coords,       dtype=float)
    assert len(preds) == len(gts) == len(coords), "Mismatch among preds/labels/coords."

    H, W, _ = slide_img_rgb.shape
    out = slide_img_rgb.copy()

    # Convert coords to TOP-LEFT if they are centers
    if coords_mode == "center":
        tl = coords - tile_size / 2.0
    else:
        tl = coords.copy()

    # Establish a grid by normalizing to the minimum and dividing by tile_size
    min_x, min_y = float(np.min(tl[:, 0])), float(np.min(tl[:, 1]))
    col_idx = np.rint((tl[:, 0] - min_x) / tile_size).astype(int)
    row_idx = np.rint((tl[:, 1] - min_y) / tile_size).astype(int)

    # Total grid size (in cells). +1 because indices are zero-based.
    n_cols = int(col_idx.max()) + 1
    n_rows = int(row_idx.max()) + 1

    # Map the grid to image space using uniform cell sizes that tile perfectly
    cell_w = W / max(n_cols, 1)
    cell_h = H / max(n_rows, 1)

    if debug_print:
        print(f"[DEBUG] grid: {n_cols}x{n_rows}, image: {W}x{H}, cell: {cell_w:.3f}x{cell_h:.3f}, mode={coords_mode}")

    # Categories (TN = transparent)
    tp = (preds == 1) & (gts == 1)
    fn = (preds == 0) & (gts == 1)
    fp = (preds == 1) & (gts == 0)

    COLORS = {
        "TP": (0, 255, 0),
        "FN": (255, 0, 0),
        "FP": (0, 0, 255),
    }

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

    # Draw each tile snapped to the grid -> no seams
    for i in range(len(col_idx)):
        x1 = col_idx[i] * cell_w
        y1 = row_idx[i] * cell_h
        x2 = (col_idx[i] + 1) * cell_w
        y2 = (row_idx[i] + 1) * cell_h

        if tp[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["TP"], alpha, pad=pad_px)
        elif fn[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["FN"], alpha, pad=pad_px)
        elif fp[i]:
            draw_filled_alpha(out, x1, y1, x2, y2, COLORS["FP"], alpha, pad=pad_px)
        # TN: transparent

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
    # Larger default font size
    plt.rcParams["font.size"] = 16

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    common_files = find_common_files(args.conch_dir, args.musk_dir, args.hopt_dir)
    if not common_files:
        raise ValueError("No common .h5 files across conch/musk/hoptimus directories.")
    if args.max_overlays is not None:
        common_files = common_files[:min(args.max_overlays, len(common_files))]

    os.makedirs(args.save_dir, exist_ok=True)

    # Infer input dims
    sample = common_files[0]
    _, f_c, _ = load_h5_data(os.path.join(args.conch_dir, sample))
    _, f_m, _ = load_h5_data(os.path.join(args.musk_dir,  sample))
    _, f_h, _ = load_h5_data(os.path.join(args.hopt_dir,  sample))
    in_conch, in_musk, in_hopt = f_c.shape[1], f_m.shape[1], f_h.shape[1]
    ensemble_in = in_conch + in_musk + in_hopt
    print(f"[INFO] Feature dims — Conch: {in_conch}, Musk: {in_musk}, Hoptimus: {in_hopt} (Ensemble: {ensemble_in})")
    print("[INFO] Ensemble inference order: [Musk, Hoptimus, Conch] — must match training!")

    # Load models
    conch_model = DeepMLP(input_dim=in_conch).to(device)
    conch_model.load_state_dict(torch.load(args.conch_model, map_location=device))
    conch_model.eval()

    ensemble_model = DeepMLPEnsemble(input_dim=ensemble_in).to(device)
    ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    ensemble_model.eval()

    # ---------- PASS 1: rank slides by tumor detection ----------
    ranking = []
    for base in common_files:
        conch_path = os.path.join(args.conch_dir, base)
        musk_path  = os.path.join(args.musk_dir,  base)
        hopt_path  = os.path.join(args.hopt_dir,  base)

        _, feat_c, labels = load_h5_data(conch_path)
        if labels is None:
            print(f"[WARN] {base}: no 'labels' — skip.")
            continue
        _, feat_m, _ = load_h5_data(musk_path)
        _, feat_h, _ = load_h5_data(hopt_path)

        conch_preds, _ = predict_classes_and_probs(conch_model, feat_c, device)
        concat_feats = np.concatenate([feat_m, feat_h, feat_c], axis=1)  # [Musk, Hopt, Conch]
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        tpr_c, tp_c, pos = tpr_and_count(conch_preds, labels)
        tpr_e, tp_e, _   = tpr_and_count(ensemble_preds, labels)
        if tpr_c is None or tpr_e is None:
            print(f"[INFO] {base}: no positive GT patches — skip for TP ranking.")
            continue

        delta_tpr = tpr_e - tpr_c
        ranking.append({
            "base": base,
            "tpr_conch": tpr_c, "tp_conch": tp_c, "pos": pos,
            "tpr_ens": tpr_e,   "tp_ens": tp_e,
            "delta_tpr": delta_tpr
        })

    if not ranking:
        raise ValueError("No slides with positive GT patches to rank by TP.")

    if args.rank_by == "tp_ensemble":
        ranking.sort(key=lambda r: r["tpr_ens"], reverse=True)
    elif args.rank_by == "tp_delta":
        ranking.sort(key=lambda r: r["delta_tpr"], reverse=True)
    else:  # tp_count
        ranking.sort(key=lambda r: r["tp_ens"], reverse=True)

    selected = ranking[:min(args.top_k, len(ranking))]

    print(f"\n[INFO] Top slides by {args.rank_by}:")
    for r in selected:
        print(f"  {r['base']}: TPR ens={r['tpr_ens']:.3f} (TP {r['tp_ens']}/{r['pos']}), "
              f"TPR conch={r['tpr_conch']:.3f} (TP {r['tp_conch']}/{r['pos']}), ΔTPR={r['delta_tpr']:+.3f}")

    # ---------- PASS 2: generate overlays only for selected ----------
    for r in selected:
        base = r["base"]
        conch_path = os.path.join(args.conch_dir, base)
        musk_path  = os.path.join(args.musk_dir,  base)
        hopt_path  = os.path.join(args.hopt_dir,  base)

        coords_c, feat_c, labels = load_h5_data(conch_path)
        coords_m, feat_m, _ = load_h5_data(musk_path)
        coords_h, feat_h, _ = load_h5_data(hopt_path)

        if not (np.array_equal(coords_c, coords_m) and np.array_equal(coords_c, coords_h)):
            print(f"[WARN] {base}: coords mismatch across models — skipping overlay.")
            continue

        # Auto-detect coords_mode on this slide if requested
        coords_mode_final = args.coords_mode
        if args.coords_mode == "auto":
            coords_mode_final = detect_coords_mode(coords_c, args.tile_size)
            print(f"[INFO] {base}: auto-detected coords_mode = {coords_mode_final}")

        slide_jpg = os.path.join(args.slides_dir, os.path.splitext(base)[0] + ".jpg")
        if not os.path.exists(slide_jpg):
            print(f"[WARN] {base}: slide image missing at {slide_jpg} — skipping overlay.")
            continue
        slide_img = np.array(Image.open(slide_jpg).convert("RGB"))

        conch_preds, _ = predict_classes_and_probs(conch_model, feat_c, device)
        concat_feats = np.concatenate([feat_m, feat_h, feat_c], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        ensemble_overlay, legend_e = overlay_confusion_map(
            slide_img_rgb=slide_img,
            patch_preds=ensemble_preds,
            patch_labels=labels,
            coords=coords_c,
            tile_size=args.tile_size,
            coords_mode=coords_mode_final,
            alpha=args.alpha,
            debug_print=args.debug,
        )
        conch_overlay, _ = overlay_confusion_map(
            slide_img_rgb=slide_img,
            patch_preds=conch_preds,
            patch_labels=labels,
            coords=coords_c,
            tile_size=args.tile_size,
            coords_mode=coords_mode_final,
            alpha=args.alpha,
            debug_print=False,
        )

        # ---- NEW: flip both overlays horizontally before plotting ----
        ensemble_overlay = np.rot90(ensemble_overlay, k=3)
        conch_overlay    = np.rot90(conch_overlay, k=3)

        # ---- Plot with a single shared legend and larger text ----
        fig, axs = plt.subplots(1, 2, figsize=(22, 11))  # slightly larger canvas
        axs[0].imshow(ensemble_overlay)
        axs[0].set_title(
            f"Ensemble (TPR={r['tpr_ens']:.3f}, ΔTPR={r['delta_tpr']:+.3f}) vs Ground Truth",
            fontsize=22
        )
        axs[0].axis("off")

        axs[1].imshow(conch_overlay)
        axs[1].set_title(
            f"Conch-only (TPR={r['tpr_conch']:.3f}) vs Ground Truth",
            fontsize=22
        )
        axs[1].axis("off")

        # Single legend for both subplots (use handles from one overlay)
        # --- tight layout with a thin legend row ---
        fig = plt.figure(figsize=(22, 11))
        gs  = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[40, 2], hspace=0.02, wspace=0.02)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        leg_ax = fig.add_subplot(gs[1, :])

        ax0.imshow(ensemble_overlay)
        ax0.set_title(f"Ensemble (TPR={r['tpr_ens']:.3f}, ΔTPR={r['delta_tpr']:+.3f}) vs Ground Truth", fontsize=22)
        ax0.axis("off")

        ax1.imshow(conch_overlay)
        ax1.set_title(f"Conch-only (TPR={r['tpr_conch']:.3f}) vs Ground Truth", fontsize=22)
        ax1.axis("off")

        # Legend axis
        leg_ax.axis("off")
        leg_ax.legend(
            handles=legend_e,
            loc="center",
            ncol=3,
            frameon=True,
            fontsize=16,
            handlelength=1.6,
            borderpad=0.3
        )

        plt.savefig(out_path, dpi=300)     # no bbox_inches="tight"
        plt.close(fig)

        print(f"[INFO] Saved overlay → {out_path}")


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tumor overlay with TN transparent and full-tile fill; ranks slides by ensemble tumor TPR.")
    p.add_argument("--conch_dir",       required=True)
    p.add_argument("--musk_dir",        required=True)
    p.add_argument("--hopt_dir",        required=True)
    p.add_argument("--slides_dir",      required=True)
    p.add_argument("--conch_model",     required=True)
    p.add_argument("--ensemble_model",  required=True)
    p.add_argument("--tile_size",       type=int, default=512, help="Patch size in ORIGINAL pixels.")
    p.add_argument("--coords_mode",     choices=["auto", "topleft", "center"], default="auto",
                   help="Coordinate semantics in H5. 'auto' tries to detect.")
    p.add_argument("--alpha",           type=float, default=0.85, help="Overlay opacity for TP/FN/FP (0..1).")
    p.add_argument("--save_dir",        required=True)
    p.add_argument("--max_overlays",    type=int, default=None)
    p.add_argument("--top_k",           type=int, default=3)
    p.add_argument("--rank_by",         choices=["tp_ensemble", "tp_delta", "tp_count"], default="tp_ensemble")
    p.add_argument("--device",          type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--debug",           action="store_true")
    args = p.parse_args()

    main(args)
'''
'''

#!/usr/bin/env python3
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


def find_common_files(musk_dir, hopt_dir, conch_dir):
    musk  = {os.path.basename(p) for p in glob.glob(os.path.join(musk_dir,  "*.h5"))}
    hopt  = {os.path.basename(p) for p in glob.glob(os.path.join(hopt_dir,  "*.h5"))}
    conch = {os.path.basename(p) for p in glob.glob(os.path.join(conch_dir, "*.h5"))}
    return sorted(musk & hopt & conch)


# ------------------------ Coordinate mode detection ------------------------

def detect_coords_mode(coords, tile_size):
    c = np.asarray(coords, dtype=float)
    if len(c) == 0:
        return "topleft"

    rx = (c[:, 0] - c[:, 0].min()) % tile_size
    ry = (c[:, 1] - c[:, 1].min()) % tile_size

    def dist_to_half(x):
        d = np.abs(x - tile_size/2.0)
        return np.minimum(d, tile_size - d)

    mean_dist_zero = 0.5 * (np.mean(np.minimum(rx, tile_size - rx)) +
                            np.mean(np.minimum(ry, tile_size - ry)))
    mean_dist_half = 0.5 * (np.mean(dist_to_half(rx)) + np.mean(dist_to_half(ry)))

    return "center" if mean_dist_half + 1e-6 < mean_dist_zero else "topleft"


# ------------------------ Overlay (TN transparent, no gaps, correct origin) ------------------------

def overlay_confusion_map(slide_img_rgb, patch_preds, patch_labels, coords,
                          tile_size, coords_mode="topleft", alpha=0.85, pad_px=6,
                          debug_print=False):

    preds  = np.asarray(patch_preds,  dtype=np.uint8).ravel()
    gts    = np.asarray(patch_labels, dtype=np.uint8).ravel()
    coords = np.asarray(coords,       dtype=float)
    assert len(preds) == len(gts) == len(coords), "Mismatch among preds/labels/coords."

    H, W, _ = slide_img_rgb.shape
    out = slide_img_rgb.copy()

    if coords_mode == "center":
        tl = coords - tile_size / 2.0
    else:
        tl = coords.copy()

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

    common_files = find_common_files(args.musk_dir, args.hopt_dir, args.conch_dir)
    if not common_files:
        raise ValueError("No common .h5 files across musk/hoptimus/conch directories.")
    if args.max_overlays is not None:
        common_files = common_files[:min(args.max_overlays, len(common_files))]

    os.makedirs(args.save_dir, exist_ok=True)

    # Infer input dims
    sample = common_files[0]
    _, f_m, _ = load_h5_data(os.path.join(args.musk_dir, sample))
    _, f_h, _ = load_h5_data(os.path.join(args.hopt_dir, sample))
    _, f_c, _ = load_h5_data(os.path.join(args.conch_dir, sample))
    in_musk, in_hopt, in_conch = f_m.shape[1], f_h.shape[1], f_c.shape[1]
    ensemble_in = in_musk + in_hopt + in_conch
    print(f"[INFO] Feature dims — Musk: {in_musk}, Hoptimus: {in_hopt}, Conch: {in_conch} (Ensemble: {ensemble_in})")
    print("[INFO] Ensemble inference order: [Musk, Hoptimus, Conch] — must match training!")

    # Load models
    musk_model = DeepMLP(input_dim=in_musk).to(device)
    musk_model.load_state_dict(torch.load(args.musk_model, map_location=device))
    musk_model.eval()

    ensemble_model = DeepMLPEnsemble(input_dim=ensemble_in).to(device)
    ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    ensemble_model.eval()

    # ---------- PASS 1 ----------
    ranking = []
    for base in common_files:
        musk_path = os.path.join(args.musk_dir, base)
        hopt_path = os.path.join(args.hopt_dir, base)
        conch_path = os.path.join(args.conch_dir, base)

        _, feat_m, labels = load_h5_data(musk_path)
        if labels is None:
            continue
        _, feat_h, _ = load_h5_data(hopt_path)
        _, feat_c, _ = load_h5_data(conch_path)

        musk_preds, _ = predict_classes_and_probs(musk_model, feat_m, device)
        concat_feats = np.concatenate([feat_m, feat_h, feat_c], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        tpr_m, tp_m, pos = tpr_and_count(musk_preds, labels)
        tpr_e, tp_e, _   = tpr_and_count(ensemble_preds, labels)
        if tpr_m is None or tpr_e is None:
            continue

        delta_tpr = tpr_e - tpr_m
        ranking.append({
            "base": base,
            "tpr_musk": tpr_m, "tp_musk": tp_m, "pos": pos,
            "tpr_ens": tpr_e, "tp_ens": tp_e,
            "delta_tpr": delta_tpr
        })

    if not ranking:
        raise ValueError("No slides with positive GT patches to rank by TP.")

    # --- Rank according to user argument ---
    if args.rank_by == "tp_ensemble":
        ranking.sort(key=lambda r: r["tpr_ens"], reverse=True)
    elif args.rank_by == "tp_delta":
        ranking.sort(key=lambda r: r["delta_tpr"], reverse=True)
    else:  # tp_count
        ranking.sort(key=lambda r: r["tp_ens"], reverse=True)

    selected = ranking[:min(args.top_k, len(ranking))]

    print(f"\n[INFO] Top slides by {args.rank_by}:")
    for r in selected:
        print(f"  {r['base']}: Ensemble TPR={r['tpr_ens']:.3f}, Musk TPR={r['tpr_musk']:.3f}, Δ={r['delta_tpr']:+.3f}")

    # ---------- PASS 2 ----------
    for r in selected:
        base = r["base"]
        musk_path = os.path.join(args.musk_dir, base)
        hopt_path = os.path.join(args.hopt_dir, base)
        conch_path = os.path.join(args.conch_dir, base)

        coords_m, feat_m, labels = load_h5_data(musk_path)
        coords_h, feat_h, _ = load_h5_data(hopt_path)
        coords_c, feat_c, _ = load_h5_data(conch_path)

        if not (np.array_equal(coords_m, coords_h) and np.array_equal(coords_m, coords_c)):
            print(f"[WARN] {base}: coords mismatch — skipping overlay.")
            continue

        coords_mode_final = args.coords_mode
        if args.coords_mode == "auto":
            coords_mode_final = detect_coords_mode(coords_m, args.tile_size)

        slide_jpg = os.path.join(args.slides_dir, os.path.splitext(base)[0] + ".jpg")
        if not os.path.exists(slide_jpg):
            continue
        slide_img = np.array(Image.open(slide_jpg).convert("RGB"))

        musk_preds, _ = predict_classes_and_probs(musk_model, feat_m, device)
        concat_feats = np.concatenate([feat_m, feat_h, feat_c], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        ensemble_overlay, legend_e = overlay_confusion_map(slide_img, ensemble_preds, labels, coords_m,
                                                           args.tile_size, coords_mode_final, args.alpha)
        musk_overlay, _ = overlay_confusion_map(slide_img, musk_preds, labels, coords_m,
                                                args.tile_size, coords_mode_final, args.alpha)

        ensemble_overlay = np.rot90(ensemble_overlay, k=3)
        musk_overlay = np.rot90(musk_overlay, k=3)

        fig = plt.figure(figsize=(22, 11))
        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[40, 2], hspace=0.02, wspace=0.02)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        leg_ax = fig.add_subplot(gs[1, :])

        ax0.imshow(ensemble_overlay)
        ax0.set_title(f"Ensemble (TPR={r['tpr_ens']:.3f}, Δ={r['delta_tpr']:+.3f})", fontsize=22)
        ax0.axis("off")

        ax1.imshow(musk_overlay)
        ax1.set_title(f"Musk-only (TPR={r['tpr_musk']:.3f})", fontsize=22)
        ax1.axis("off")

        leg_ax.axis("off")
        leg_ax.legend(handles=legend_e, loc="center", ncol=3, frameon=True, fontsize=16)

        out_path = os.path.join(args.save_dir, f"{os.path.splitext(base)[0]}_overlay.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved overlay → {out_path}")


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overlay script using MUSK as baseline model with ranking control.")
    p.add_argument("--musk_dir", required=True)
    p.add_argument("--hopt_dir", required=True)
    p.add_argument("--conch_dir", required=True)
    p.add_argument("--slides_dir", required=True)
    p.add_argument("--musk_model", required=True)
    p.add_argument("--ensemble_model", required=True)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--coords_mode", choices=["auto", "topleft", "center"], default="auto")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_overlays", type=int, default=None)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--rank_by", choices=["tp_ensemble", "tp_delta", "tp_count"], default="tp_ensemble",
                   help="Ranking criterion for selecting top slides.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    main(args)
'''
#!/usr/bin/env python3
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
    hopt  = {os.path.basename(p) for p in glob.glob(os.path.join(hopt_dir, "*.h5"))}
    musk  = {os.path.basename(p) for p in glob.glob(os.path.join(musk_dir, "*.h5"))}
    conch = {os.path.basename(p) for p in glob.glob(os.path.join(conch_dir, "*.h5"))}
    return sorted(hopt & musk & conch)


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

    if coords_mode == "center":
        tl = coords - tile_size / 2.0
    else:
        tl = coords.copy()

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
    print(f"[INFO] Feature dims — Hoptimus: {in_hopt}, Musk: {in_musk}, Conch: {in_conch} (Ensemble: {ensemble_in})")
    print("[INFO] Ensemble inference order: [Musk, Hoptimus, Conch] — must match training!")

    # Load models
    hopt_model = DeepMLP(input_dim=in_hopt).to(device)
    hopt_model.load_state_dict(torch.load(args.hopt_model, map_location=device))
    hopt_model.eval()

    ensemble_model = DeepMLPEnsemble(input_dim=ensemble_in).to(device)
    ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    ensemble_model.eval()

    # ---------- PASS 1 ----------
    ranking = []
    for base in common_files:
        hopt_path = os.path.join(args.hopt_dir, base)
        musk_path = os.path.join(args.musk_dir, base)
        conch_path = os.path.join(args.conch_dir, base)

        _, feat_h, labels = load_h5_data(hopt_path)
        if labels is None:
            continue
        _, feat_m, _ = load_h5_data(musk_path)
        _, feat_c, _ = load_h5_data(conch_path)

        hopt_preds, _ = predict_classes_and_probs(hopt_model, feat_h, device)
        concat_feats = np.concatenate([feat_m, feat_h, feat_c], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        tpr_h, tp_h, pos = tpr_and_count(hopt_preds, labels)
        tpr_e, tp_e, _ = tpr_and_count(ensemble_preds, labels)
        if tpr_h is None or tpr_e is None:
            continue

        delta_tpr = tpr_e - tpr_h
        ranking.append({
            "base": base,
            "tpr_hopt": tpr_h, "tp_hopt": tp_h, "pos": pos,
            "tpr_ens": tpr_e, "tp_ens": tp_e,
            "delta_tpr": delta_tpr
        })

    if not ranking:
        raise ValueError("No slides with positive GT patches to rank by TP.")

    # --- Rank according to user argument ---
    if args.rank_by == "tp_ensemble":
        ranking.sort(key=lambda r: r["tpr_ens"], reverse=True)
    elif args.rank_by == "tp_delta":
        ranking.sort(key=lambda r: r["delta_tpr"], reverse=True)
    else:  # tp_count
        ranking.sort(key=lambda r: r["tp_ens"], reverse=True)
        

    selected = ranking[:min(args.top_k, len(ranking))]

    print(f"\n[INFO] Top slides by {args.rank_by}:")
    
    for r in selected:
        print(f"  {r['base']}: Ensemble TPR={r['tpr_ens']:.3f}, Hoptimus={r['tpr_hopt']:.3f}, Δ={r['delta_tpr']:+.3f}")

    # ---------- PASS 2 ----------
    for r in selected:
        base = r["base"]
        hopt_path = os.path.join(args.hopt_dir, base)
        musk_path = os.path.join(args.musk_dir, base)
        conch_path = os.path.join(args.conch_dir, base)

        coords_h, feat_h, labels = load_h5_data(hopt_path)
        coords_m, feat_m, _ = load_h5_data(musk_path)
        coords_c, feat_c, _ = load_h5_data(conch_path)

        if not (np.array_equal(coords_h, coords_m) and np.array_equal(coords_h, coords_c)):
            print(f"[WARN] {base}: coords mismatch — skipping overlay.")
            continue

        coords_mode_final = args.coords_mode
        if args.coords_mode == "auto":
            coords_mode_final = detect_coords_mode(coords_h, args.tile_size)

        slide_jpg = os.path.join(args.slides_dir, os.path.splitext(base)[0] + ".jpg")
        if not os.path.exists(slide_jpg):
            print(f"[ERROR] Missing slide: {slide_jpg}")
            continue

        if not (np.array_equal(coords_h, coords_m) and np.array_equal(coords_h, coords_c)):
            print(f"[ERROR] Coord mismatch — Hopt:{coords_h.shape}, Musk:{coords_m.shape}, Conch:{coords_c.shape}")
            continue

        if not os.path.exists(slide_jpg):
            continue
        
        slide_img = np.array(Image.open(slide_jpg).convert("RGB"))

        hopt_preds, _ = predict_classes_and_probs(hopt_model, feat_h, device)
        concat_feats = np.concatenate([feat_m, feat_h, feat_c], axis=1)
        ensemble_preds, _ = predict_classes_and_probs(ensemble_model, concat_feats, device)

        ensemble_overlay, legend_e = overlay_confusion_map(slide_img, ensemble_preds, labels, coords_h,
                                                           args.tile_size, coords_mode_final, args.alpha)
        hopt_overlay, _ = overlay_confusion_map(slide_img, hopt_preds, labels, coords_h,
                                                args.tile_size, coords_mode_final, args.alpha)

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
        ax1.set_title(f"Hoptimus-only (TPR={r['tpr_hopt']:.3f})", fontsize=22)
        ax1.axis("off")

        leg_ax.axis("off")
        leg_ax.legend(handles=legend_e, loc="center", ncol=3, frameon=True, fontsize=16)

        out_path = os.path.join(args.save_dir, f"{os.path.splitext(base)[0]}_overlay.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved overlay → {out_path}")


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overlay script using HOPTIMUS1 as baseline model with ranking control.")
    p.add_argument("--hopt_dir", required=True)
    p.add_argument("--musk_dir", required=True)
    p.add_argument("--conch_dir", required=True)
    p.add_argument("--slides_dir", required=True)
    p.add_argument("--musk_model", required=True)
    p.add_argument("--ensemble_model", required=True)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--coords_mode", choices=["auto", "topleft", "center"], default="auto")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_overlays", type=int, default=None)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--rank_by", choices=["tp_ensemble", "tp_delta", "tp_count"], default="tp_ensemble",
                   help="Ranking criterion for selecting top slides.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    main(args)

#python overlay_top_musk_ranked.py --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_hoptimus1_kich --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_conch_v15_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/visualization --musk_model /scratch/pioneer/users/sxk2517/model_weights/musk_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top_musk --rank_by tp_delta --top_k 3 --device cuda
#python overlay_top.py --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_hoptimus1_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_musk_kich --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_conch_v15_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/visualization --hopt_model /scratch/pioneer/users/sxk2517/model_weights/hoptimus1_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top_hoptimus1 --rank_by tp_delta --top_k 3 --device cuda
