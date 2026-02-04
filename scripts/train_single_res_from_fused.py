#!/usr/bin/env python3
"""
Train 5x-only, 10x-only, 20x-only MLPs on the same aligned train set as fusion
(same FusedFeatureDataset and 80/20 split with seed). Use the resulting checkpoints
with fusion_benchmark_and_disagreement.py for a fair comparison.

Usage:
  python scripts/train_single_res_from_fused.py \
    --dirs trident_processed/5x_512px_0px_overlap/features_conch_v15 \
            trident_processed/10x_1024px_0px_overlap/features_conch_v15 \
            trident_processed/20x_2048px_0px_overlap/features_conch_v15 \
    --seed 42 --save_prefix conch --epochs 20
  # Writes conch_5x.pth, conch_10x.pth, conch_20x.pth
"""

import os
import sys
import argparse
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from fusion_model import FusedFeatureDataset, DeepMLP


def get_block_dims(dirs):
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


def main():
    parser = argparse.ArgumentParser(description="Train 5x/10x/20x MLPs on aligned fused train set.")
    parser.add_argument("--dirs", nargs=3, required=True, metavar="DIR", help="Three H5 dirs (5x, 10x, 20x)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_prefix", default="conch", help="Prefix: <prefix>_5x.pth, ...")
    parser.add_argument("--save_paths", nargs=3, default=None, metavar="PTH", help="Explicit paths (overrides save_prefix)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading FusedFeatureDataset...")
    dataset = FusedFeatureDataset(args.dirs)
    block_dims = get_block_dims(args.dirs)
    total_dim = sum(block_dims)
    assert dataset.features.shape[1] == total_dim

    val_frac = 0.2
    val_size = int(val_frac * len(dataset))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, _ = random_split(dataset, [train_size, val_size], generator=generator)
    train_indices = train_set.indices

    train_labels = dataset.labels[train_indices]
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    save_paths = args.save_paths
    if save_paths is None:
        save_paths = [f"{args.save_prefix}_5x.pth", f"{args.save_prefix}_10x.pth", f"{args.save_prefix}_20x.pth"]

    offset = 0
    for i, (block_dim, save_path) in enumerate(zip(block_dims, save_paths)):
        block_feats = dataset.features[train_indices, offset : offset + block_dim]
        block_labels = dataset.labels[train_indices]
        train_dataset = torch.utils.data.TensorDataset(block_feats, block_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)

        model = DeepMLP(input_dim=block_dim).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Block {i} Epoch {epoch+1}/{args.epochs} Loss: {total_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Saved {save_path}")
        offset += block_dim

    print("Done. Use these checkpoints with fusion_benchmark_and_disagreement.py --single_res_models ...")


if __name__ == "__main__":
    main()
