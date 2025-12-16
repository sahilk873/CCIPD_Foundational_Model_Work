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


def hard_vote(preds_list):
    """Majority vote on class labels (len=3 → no ties)."""
    stacked = np.vstack([p.astype(int).ravel() for p in preds_list])
    votes = stacked.sum(axis=0)  # count of 1s
    return (votes >= 2).astype(int)


def soft_vote(probs_list, thresh=0.5):
    """Average probabilities then threshold."""
    stacked = np.vstack([q.ravel() for q in probs_list])
    mean_probs = stacked.mean(axis=0)
    return (mean_probs >= thresh).astype(int), mean_probs


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

    # Output folders
    individuals_dir = os.path.join(args.save_dir, "individuals")
    hopt_dir_out = os.path.join(individuals_dir, "hoptimus")
    musk_dir_out = os.path.join(individuals_dir, "musk")
    conch_dir_out = os.path.join(individuals_dir, "conch")

    ensembles_dir = os.path.join(args.save_dir, "ensembles")
    deep_dir_out = os.path.join(ensembles_dir, "deep_mlp")
    soft_dir_out = os.path.join(ensembles_dir, "soft_vote")
    hard_dir_out = os.path.join(ensembles_dir, "hard_vote") if args.include_hard_vote else None

    topdiff_dir = os.path.join(args.save_dir, "top_diffs")

    for d in [args.save_dir, individuals_dir, hopt_dir_out, musk_dir_out, conch_dir_out,
              ensembles_dir, deep_dir_out, soft_dir_out, topdiff_dir] + ([hard_dir_out] if hard_dir_out else []):
        os.makedirs(d, exist_ok=True)

    # Infer input dims
    sample = common_files[0]
    _, f_h, _ = load_h5_data(os.path.join(args.hopt_dir, sample))
    _, f_m, _ = load_h5_data(os.path.join(args.musk_dir, sample))
    _, f_c, _ = load_h5_data(os.path.join(args.conch_dir, sample))
    in_hopt, in_musk, in_conch = f_h.shape[1], f_m.shape[1], f_c.shape[1]
    ensemble_in = in_hopt + in_musk + in_conch
    print(f"[INFO] Feature dims — Hoptimus: {in_hopt}, Musk: {in_musk}, Conch: {in_conch} (Ensemble: {ensemble_in})")
    print("[INFO] Ensemble inference order: [Conch, Musk, Hoptimus] — must match training!")


    # Load models
    hopt_model = DeepMLP(input_dim=in_hopt).to(device)
    hopt_model.load_state_dict(torch.load(args.hopt_model, map_location=device))
    hopt_model.eval()

    musk_model = DeepMLP(input_dim=in_musk).to(device)
    musk_model.load_state_dict(torch.load(args.musk_model, map_location=device))
    musk_model.eval()

    conch_model = DeepMLP(input_dim=in_conch).to(device)
    conch_model.load_state_dict(torch.load(args.conch_model, map_location=device))
    conch_model.eval()

    ensemble_model = DeepMLPEnsemble(input_dim=ensemble_in).to(device)
    ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    ensemble_model.eval()

    # ---------- PASS 1: compute ranking by max disagreement ----------
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

        # individual preds/probs
        hopt_preds, hopt_probs = predict_classes_and_probs(hopt_model, feat_h, device)
        musk_preds, musk_probs = predict_classes_and_probs(musk_model, feat_m, device)
        conch_preds, conch_probs = predict_classes_and_probs(conch_model, feat_c, device)

        # deep ensemble preds
        concat_feats = np.concatenate([feat_c, feat_m, feat_h], axis=1)
        ens_preds, ens_probs = predict_classes_and_probs(ensemble_model, concat_feats, device)

        # voting ensembles
        hard_preds = hard_vote([hopt_preds, musk_preds, conch_preds])
        soft_preds, _ = soft_vote([hopt_probs, musk_probs, conch_probs], thresh=0.5)

        # TPRs
        tpr_h, tp_h, pos = tpr_and_count(hopt_preds, labels)
        tpr_m, tp_m, _ = tpr_and_count(musk_preds, labels)
        tpr_c, tp_c, _ = tpr_and_count(conch_preds, labels)
        tpr_ens, tp_ens, _ = tpr_and_count(ens_preds, labels)
        tpr_soft, tp_soft, _ = tpr_and_count(soft_preds, labels)
        tpr_hard, tp_hard, _ = tpr_and_count(hard_preds, labels)

        if None in (tpr_h, tpr_m, tpr_c, tpr_ens, tpr_soft, tpr_hard):
            # skip slides without positives
            continue

        # disagreement metric: max over models of max(|model-ens|, |model-soft|)
        deltas = [
            max(abs(tpr_h - tpr_ens), abs(tpr_h - tpr_soft)),
            max(abs(tpr_m - tpr_ens), abs(tpr_m - tpr_soft)),
            max(abs(tpr_c - tpr_ens), abs(tpr_c - tpr_soft)),
        ]
        max_disagree = max(deltas)

        ranking.append({
            "base": base,
            "pos": pos,
            "tprs": {
                "hopt": tpr_h, "musk": tpr_m, "conch": tpr_c,
                "deep": tpr_ens, "soft": tpr_soft, "hard": tpr_hard
            },
            "tps": {
                "hopt": tp_h, "musk": tp_m, "conch": tp_c,
                "deep": tp_ens, "soft": tp_soft, "hard": tp_hard
            },
            "max_delta_vs_ensembles": max_disagree
        })

    if not ranking:
        raise ValueError("No slides with positive GT patches to rank by disagreement / TP.")

    # --- Rank according to argument ---
    if args.rank_by == "tp_ensemble":
        ranking.sort(key=lambda r: r["tprs"]["deep"], reverse=True)
    elif args.rank_by == "tp_delta":
        # For backward-compat: use deep ensemble minus HOPT delta
        ranking.sort(key=lambda r: r["tprs"]["deep"] - r["tprs"]["hopt"], reverse=True)
    elif args.rank_by == "tp_count":
        ranking.sort(key=lambda r: r["tps"]["deep"], reverse=True)
    else:  # "max_delta_vs_ensembles"
        ranking.sort(key=lambda r: r["max_delta_vs_ensembles"], reverse=True)

    selected = ranking[:min(args.top_k, len(ranking))]

    print(f"\n[INFO] Top slides by {args.rank_by}:")
    for r in selected:
        t = r["tprs"]
        print(f"  {r['base']}: "
              f"maxΔ={r['max_delta_vs_ensembles']:.3f} | "
              f"deep={t['deep']:.3f}, soft={t['soft']:.3f}, hard={t['hard']:.3f} | "
              f"hopt={t['hopt']:.3f}, musk={t['musk']:.3f}, conch={t['conch']:.3f}")

    # ---------- PASS 2: save overlays ----------
    for r in selected:
        base = r["base"]
        stem = os.path.splitext(base)[0]

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

        slide_jpg = os.path.join(args.slides_dir, stem + ".jpg")
        if not os.path.exists(slide_jpg):
            print(f"[ERROR] Missing slide: {slide_jpg}")
            continue

        slide_img = np.array(Image.open(slide_jpg).convert("RGB"))

        # Predictions
        hopt_preds, hopt_probs = predict_classes_and_probs(hopt_model, feat_h, device)
        musk_preds, musk_probs = predict_classes_and_probs(musk_model, feat_m, device)
        conch_preds, conch_probs = predict_classes_and_probs(conch_model, feat_c, device)

        concat_feats = np.concatenate([feat_c, feat_m, feat_h], axis=1)
        ens_preds, ens_probs = predict_classes_and_probs(ensemble_model, concat_feats, device)

        hard_preds = hard_vote([hopt_preds, musk_preds, conch_preds])
        soft_preds, _ = soft_vote([hopt_probs, musk_probs, conch_probs], thresh=0.5)

        # Overlays for each
        overlay_items = [
            ("Hoptimus", hopt_preds, hopt_dir_out, r["tprs"]["hopt"]),
            ("Musk",     musk_preds, musk_dir_out, r["tprs"]["musk"]),
            ("Conch",    conch_preds, conch_dir_out, r["tprs"]["conch"]),
            ("Ensemble (Deep MLP)", ens_preds, deep_dir_out, r["tprs"]["deep"]),
            ("Ensemble (Soft Vote)", soft_preds, soft_dir_out, r["tprs"]["soft"])
        ]
        if args.include_hard_vote:
            overlay_items.append(("Ensemble (Hard Vote)", hard_preds, hard_dir_out, r["tprs"]["hard"]))

        # Save individual overlays into their respective folders
        per_map_images = {}
        legend_any = None
        for title, preds_arr, outdir, tpr_val in overlay_items:
            ov, legend = overlay_confusion_map(
                slide_img, preds_arr, labels, coords_h,
                args.tile_size, coords_mode_final, args.alpha, pad_px=args.pad_px, debug_print=args.debug
            )
            ov = np.rot90(ov, k=3)
            out_path = os.path.join(outdir, f"{stem}_overlay.png")
            Image.fromarray(ov).save(out_path, quality=95)
            legend_any = legend  # reuse legend
            per_map_images[title] = (ov, tpr_val, out_path)
            print(f"[INFO] Saved overlay → {out_path}")

        # Composite comparison for top disagreement slides
        # Layout: up to 6 panels (3 individual + deep + soft + optional hard)
        titles_in_order = ["Ensemble (Deep MLP)", "Ensemble (Soft Vote)", "Hoptimus", "Musk", "Conch"]
        if args.include_hard_vote:
            titles_in_order.insert(1, "Ensemble (Hard Vote)")

        n_panels = len(titles_in_order)
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols)) + 1  # +1 for legend row

        fig = plt.figure(figsize=(ncols * 7.5, (nrows - 1) * 6.5 + 2.0))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=[6]*(nrows-1) + [1], hspace=0.03, wspace=0.03)

        # draw images
        idx = 0
        for ti in titles_in_order:
            ov_img, tpr_val, _ = per_map_images[ti]
            rr = idx // ncols
            cc = idx % ncols
            ax = fig.add_subplot(gs[rr, cc])
            ax.imshow(ov_img)
            ax.set_title(f"{ti} (TPR={tpr_val:.3f})", fontsize=18)
            ax.axis("off")
            idx += 1

        # legend
        leg_ax = fig.add_subplot(gs[nrows-1, :])
        leg_ax.axis("off")
        leg_ax.legend(handles=legend_any, loc="center", ncol=3, frameon=True, fontsize=14)

        comp_out = os.path.join(topdiff_dir, f"{stem}_comparison.png")
        plt.savefig(comp_out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved comparison → {comp_out}")


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate overlays for individual models, deep ensemble, and vote ensembles; rank by max disagreement.")
    # feature/slide roots
    p.add_argument("--hopt_dir", required=True)
    p.add_argument("--musk_dir", required=True)
    p.add_argument("--conch_dir", required=True)
    p.add_argument("--slides_dir", required=True)

    # weights
    p.add_argument("--hopt_model", required=True)
    p.add_argument("--musk_model", required=True)
    p.add_argument("--conch_model", required=True)
    p.add_argument("--ensemble_model", required=True)

    # viz / geometry
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--coords_mode", choices=["auto", "topleft", "center"], default="auto")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--pad_px", type=int, default=6, help="Padding (in px) to slightly expand each colored cell for seamless overlays.")

    # output, selection, device
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_overlays", type=int, default=150)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--rank_by", choices=["tp_ensemble", "tp_delta", "tp_count", "max_delta_vs_ensembles"],
                   default="max_delta_vs_ensembles",
                   help="Ranking criterion. Default: slides where individuals disagree most vs. deep+soft ensembles.")
    p.add_argument("--include_hard_vote", action="store_true", help="Also generate/save hard-vote ensemble overlays.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    main(args)

#python total_overlay.py --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --hopt_model /scratch/pioneer/users/sxk2517/model_weights/hoptimus_base.pth --musk_model /scratch/pioneer/users/sxk2517/model_weights/musk_base.pth --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_20x.pth --ensemble_model  /scratch/pioneer/users/sxk2517/model_weights/fused_all.pth --save_dir /scratch/pioneer/users/sxk2517/overlays_total --max_overlays 150