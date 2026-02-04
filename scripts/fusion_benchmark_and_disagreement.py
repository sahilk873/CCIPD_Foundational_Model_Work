#!/usr/bin/env python3
"""
Benchmark (Section 1) + Disagreement analysis (Section 3): fusion vs single-res vs majority vote
on the same aligned test set, plus accuracy on disagreement/agreement patches.

Uses FusedFeatureDataset for aligned patches across 5x/10x/20x. Same 80/20 train/val split
logic as fusion training (with fixed seed for reproducibility). Evaluates:
  - Fusion model (full fused features)
  - Single-resolution MLPs (5x-only, 10x-only, 20x-only) on their block
  - Majority vote of the three single-res predictions

Then runs disagreement analysis: among patches where the three single-res models disagree,
does fusion (or majority vote) get the correct label more often?

Usage:
  python scripts/fusion_benchmark_and_disagreement.py \
    --dirs trident_processed/5x_512px_0px_overlap/features_conch_v15 \
            trident_processed/10x_1024px_0px_overlap/features_conch_v15 \
            trident_processed/20x_2048px_0px_overlap/features_conch_v15 \
    --fusion_model fused_conch_all.pth \
    --single_res_models conch_5x.pth conch_10x.pth conch_20x.pth \
    --seed 42 \
    --out_dir fusion_benchmark_conch
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Same dataset and model as fusion training
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from fusion_model import FusedFeatureDataset, DeepMLP


def get_block_dims(dirs):
    """Infer feature dimension per dir from first .h5 in each directory."""
    block_dims = []
    for d in dirs:
        files = [f for f in os.listdir(d) if f.endswith(".h5")]
        if not files:
            raise ValueError(f"No .h5 files in {d}")
        path = os.path.join(d, sorted(files)[0])
        with h5py.File(path, "r") as f:
            if "features" not in f:
                raise ValueError(f"{path}: missing 'features'")
            block_dims.append(int(f["features"].shape[1]))
    return block_dims


def get_val_set(dataset, val_frac=0.2, seed=42):
    """Reproducible 80/20 train/val split (same logic as fusion_model.py)."""
    n = len(dataset)
    val_size = int(val_frac * n)
    train_size = n - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    return val_set


def evaluate_fusion(model, features, labels, device, batch_size=256):
    """Run fusion model on (features, labels); return y_true, y_pred, y_probs."""
    model.eval()
    y_true = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    y_pred_list, y_probs_list = [], []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            X = features[i : i + batch_size].to(device)
            logits = model(X)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_list.extend(batch_preds)
            y_probs_list.extend(batch_probs)
    return y_true, np.array(y_pred_list), np.array(y_probs_list)


def evaluate_single_res(model, block_features, device, batch_size=256):
    """Run single-res MLP on one block; return preds, probs."""
    model.eval()
    preds_list, probs_list = [], []
    with torch.no_grad():
        for i in range(0, len(block_features), batch_size):
            X = block_features[i : i + batch_size].to(device)
            logits = model(X)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.extend(batch_preds)
            probs_list.extend(batch_probs)
    return np.array(preds_list), np.array(probs_list)


def majority_vote(pred_5x, pred_10x, pred_20x):
    """Per-patch majority vote (0 or 1). Tie-break: round(mean) -> 1 if sum>=2."""
    stack = np.stack([pred_5x, pred_10x, pred_20x], axis=1)
    return (np.sum(stack, axis=1) >= 2).astype(np.int64)


def compute_metrics(y_true, y_pred, y_probs=None, name=""):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs) if y_probs is not None and len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "sensitivity": recall, "f1": f1, "auc": auc}


def run_benchmark(
    dirs,
    fusion_model_path,
    single_res_paths,
    seed=42,
    batch_size=256,
    out_dir="fusion_benchmark",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load aligned fused dataset
    print("[INFO] Loading FusedFeatureDataset...")
    dataset = FusedFeatureDataset(dirs)
    block_dims = get_block_dims(dirs)
    total_dim = sum(block_dims)
    assert dataset.features.shape[1] == total_dim, (dataset.features.shape[1], total_dim)

    # Val set (same split as fusion training with seed)
    val_set = get_val_set(dataset, val_frac=0.2, seed=seed)
    val_indices = val_set.indices
    val_features = dataset.features[val_indices]
    val_labels = dataset.labels[val_indices]
    n_val = len(val_labels)
    print(f"[INFO] Val set size: {n_val}")
    if n_val == 0:
        raise ValueError("Validation set is empty (dataset too small or split produced 0 samples).")

    # Slice val features into blocks
    offset = 0
    blocks = []
    for d in block_dims:
        blocks.append(val_features[:, offset : offset + d])
        offset += d

    # Load fusion model
    print("[INFO] Loading fusion model...")
    fusion = DeepMLP(input_dim=total_dim).to(device)
    ckpt = torch.load(fusion_model_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    fusion.load_state_dict(state)
    fusion.eval()

    # Fusion predictions
    y_true, pred_fusion, probs_fusion = evaluate_fusion(
        fusion, val_features, val_labels, device, batch_size=batch_size
    )
    y_true = np.asarray(y_true)

    # Load single-res models and run on their block
    if len(single_res_paths) != 3:
        raise ValueError("--single_res_models must provide exactly 3 paths (5x, 10x, 20x).")
    pred_5x, pred_10x, pred_20x = None, None, None
    for i, (path, block_feats) in enumerate(zip(single_res_paths, blocks)):
        dim = block_dims[i]
        model = DeepMLP(input_dim=dim).to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        preds, _ = evaluate_single_res(model, block_feats, device, batch_size=batch_size)
        if i == 0:
            pred_5x = preds
        elif i == 1:
            pred_10x = preds
        else:
            pred_20x = preds

    # Majority vote
    pred_majority = majority_vote(pred_5x, pred_10x, pred_20x)

    # Sanity check: all prediction arrays match val set size
    assert len(pred_fusion) == n_val, (len(pred_fusion), n_val)
    assert len(pred_5x) == len(pred_10x) == len(pred_20x) == n_val, (len(pred_5x), len(pred_10x), len(pred_20x), n_val)
    assert len(pred_majority) == n_val, (len(pred_majority), n_val)

    # ---------- Benchmark: metrics for each model ----------
    res_names = ["5x", "10x", "20x"]
    res_preds = [pred_5x, pred_10x, pred_20x]
    all_metrics = []

    for name, preds in [("fusion", pred_fusion), ("majority_vote", pred_majority)] + list(
        zip(res_names, res_preds)
    ):
        probs = probs_fusion if name == "fusion" else None
        m = compute_metrics(y_true, preds, probs, name=name)
        m["model"] = name
        all_metrics.append(m)

    # Print benchmark table
    print("\n" + "=" * 60)
    print("BENCHMARK: Fusion vs single-resolution vs majority vote")
    print("=" * 60)
    print(f"{'Model':<16} {'Accuracy':>10} {'Sensitivity':>12} {'F1':>10} {'AUC':>10}")
    print("-" * 60)
    for m in all_metrics:
        auc_str = f"{m['auc']:.4f}" if np.isfinite(m["auc"]) else "n/a"
        print(
            f"{m['model']:<16} {m['accuracy']:>10.4f} {m['sensitivity']:>12.4f} {m['f1']:>10.4f} {auc_str:>10}"
        )
    print("=" * 60)

    # ---------- Disagreement analysis ----------
    agree = (pred_5x == pred_10x) & (pred_10x == pred_20x)
    disagree = ~agree
    n_agree = int(np.sum(agree))
    n_disagree = int(np.sum(disagree))

    acc_fusion_dis = acc_maj_dis = acc_fusion_agree = acc_maj_agree = None
    correct_5x = correct_10x = correct_20x = 0

    print("\n" + "=" * 60)
    print("DISAGREEMENT: Patches where 5x, 10x, 20x predictions differ")
    print("=" * 60)
    pct_agree = 100 * n_agree / n_val if n_val > 0 else 0.0
    pct_disagree = 100 * n_disagree / n_val if n_val > 0 else 0.0
    print(f"  Agreement patches:   {n_agree} ({pct_agree:.1f}%)")
    print(f"  Disagreement patches: {n_disagree} ({pct_disagree:.1f}%)")

    # Accuracy on disagreement subset
    if n_disagree > 0:
        y_d = y_true[disagree]
        fus_d = pred_fusion[disagree]
        maj_d = pred_majority[disagree]
        acc_fusion_dis = accuracy_score(y_d, fus_d)
        acc_maj_dis = accuracy_score(y_d, maj_d)
        print(f"\n  On DISAGREEMENT patches:")
        print(f"    Fusion accuracy:       {acc_fusion_dis:.4f}")
        print(f"    Majority vote accuracy: {acc_maj_dis:.4f}")
        if acc_fusion_dis > acc_maj_dis:
            print("    -> Fusion resolves disagreements better than majority vote.")
        elif acc_fusion_dis < acc_maj_dis:
            print("    -> Majority vote is better on disagreement patches.")
        else:
            print("    -> Fusion and majority vote tie on disagreement patches.")
        correct_5x = (pred_5x[disagree] == y_true[disagree]).sum()
        correct_10x = (pred_10x[disagree] == y_true[disagree]).sum()
        correct_20x = (pred_20x[disagree] == y_true[disagree]).sum()
        print(f"\n  When single-res DISAGREED, who was correct (count):")
        print(f"    5x correct:  {correct_5x} / {n_disagree}")
        print(f"    10x correct: {correct_10x} / {n_disagree}")
        print(f"    20x correct: {correct_20x} / {n_disagree}")

    # Accuracy on agreement subset
    if n_agree > 0:
        y_a = y_true[agree]
        fus_a = pred_fusion[agree]
        maj_a = pred_majority[agree]
        acc_fusion_agree = accuracy_score(y_a, fus_a)
        acc_maj_agree = accuracy_score(y_a, maj_a)
        print(f"\n  On AGREEMENT patches (all three single-res agree):")
        print(f"    Fusion accuracy:       {acc_fusion_agree:.4f}")
        print(f"    Majority vote accuracy: {acc_maj_agree:.4f}")

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)

    # Benchmark table CSV
    pd_all = pd.DataFrame(all_metrics)
    bench_path = os.path.join(out_dir, "benchmark_metrics.csv")
    pd_all.to_csv(bench_path, index=False)
    print(f"\n[INFO] Benchmark metrics saved to {bench_path}")

    # Disagreement summary
    disagree_rows = [
        {"subset": "all", "n": n_val, "fusion_accuracy": float(accuracy_score(y_true, pred_fusion))},
        {"subset": "agreement", "n": n_agree, "fusion_accuracy": float(acc_fusion_agree) if acc_fusion_agree is not None else None, "majority_vote_accuracy": float(acc_maj_agree) if acc_maj_agree is not None else None},
        {"subset": "disagreement", "n": n_disagree, "fusion_accuracy": float(acc_fusion_dis) if acc_fusion_dis is not None else None, "majority_vote_accuracy": float(acc_maj_dis) if acc_maj_dis is not None else None},
    ]
    if n_disagree > 0:
        disagree_rows.append({"subset": "disagreement_5x_correct", "n": int(correct_5x)})
        disagree_rows.append({"subset": "disagreement_10x_correct", "n": int(correct_10x)})
        disagree_rows.append({"subset": "disagreement_20x_correct", "n": int(correct_20x)})
    disagree_path = os.path.join(out_dir, "disagreement_summary.csv")
    pd.DataFrame(disagree_rows).to_csv(disagree_path, index=False)
    print(f"[INFO] Disagreement summary saved to {disagree_path}")

    return {
        "metrics": all_metrics,
        "n_val": n_val,
        "n_agree": int(n_agree),
        "n_disagree": int(n_disagree),
        "pred_fusion": pred_fusion,
        "pred_majority": pred_majority,
        "pred_5x": pred_5x,
        "pred_10x": pred_10x,
        "pred_20x": pred_20x,
        "y_true": y_true,
        "agree": agree,
        "disagree": disagree,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fusion vs single-res vs majority vote and run disagreement analysis."
    )
    parser.add_argument(
        "--dirs",
        nargs=3,
        required=True,
        metavar="DIR",
        help="Exactly 3 H5 directories (5x, 10x, 20x) in same order as fusion training",
    )
    parser.add_argument("--fusion_model", required=True, help="Path to fusion checkpoint (.pth)")
    parser.add_argument(
        "--single_res_models",
        nargs=3,
        required=True,
        metavar="PTH",
        help="Paths to 5x, 10x, 20x single-resolution MLP checkpoints (.pth)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", default="fusion_benchmark", help="Output directory for CSVs")

    args = parser.parse_args()

    for d in args.dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    if not os.path.isfile(args.fusion_model):
        raise FileNotFoundError(f"Fusion model not found: {args.fusion_model}")
    for p in args.single_res_models:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Single-res model not found: {p}")

    run_benchmark(
        dirs=list(args.dirs),
        fusion_model_path=args.fusion_model,
        single_res_paths=list(args.single_res_models),
        seed=args.seed,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
