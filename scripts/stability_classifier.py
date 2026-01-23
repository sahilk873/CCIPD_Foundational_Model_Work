#!/usr/bin/env python3
# cv_iqr_mean_std_mlp_classifier.py
# ------------------------------------------------------------
# Train an MLP on concatenated patch-level features from a single
# directory of .h5 files.
#
# Expected keys in each .h5 (stability builder output):
#   required: label
#   optional: features (baseline), cv, iqr, mean, std, coords
#
# You can choose which feature blocks to use via --use.
# Example:
#   --use cv iqr mean std              (stability-only)
#   --use features                     (baseline-only)
#   --use features cv iqr mean std     (fused)
#
# Defaults:
#   --use cv iqr mean std
#
# Model:
#   LayerNorm MLP (no BatchNorm)
# Loss:
#   class-weighted CrossEntropyLoss (no WeightedRandomSampler)
#
# PATCH:
# - compute_metrics now returns a dict (still prints by default)
# - after loading the best checkpoint, we re-evaluate on val and
#   can write a JSON summary via --out_json
# ------------------------------------------------------------

import os
import json
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ALLOWED_BLOCKS = ("features", "cv", "iqr", "mean", "std")


class PatchConcatDataset(Dataset):
    def __init__(self, h5_dir: str, use_blocks: list[str]):
        use_blocks = list(use_blocks)
        if not use_blocks:
            raise ValueError("use_blocks is empty. Provide at least one block to use.")
        bad = [b for b in use_blocks if b not in ALLOWED_BLOCKS]
        if bad:
            raise ValueError(f"Unknown blocks {bad}. Allowed: {list(ALLOWED_BLOCKS)}")

        files = [f for f in os.listdir(h5_dir) if f.lower().endswith((".h5", ".hdf5"))]
        if not files:
            raise ValueError(f"No .h5/.hdf5 files found in: {h5_dir}")

        feats_list, labels_list = [], []
        self.block_dims: dict[str, int] = {}
        self.use_blocks = use_blocks

        for fname in sorted(files):
            path = os.path.join(h5_dir, fname)
            try:
                with h5py.File(path, "r") as f:
                    if "label" not in f:
                        print(f"[WARN] Skipping {fname}: missing key ['label']")
                        continue

                    # ---- labels ----
                    y = f["label"][:]
                    if y.ndim == 2 and y.shape[1] == 1:
                        y = y.reshape(-1)
                    elif y.ndim == 2 and y.shape[1] == 2:
                        y = np.argmax(y, axis=1)
                    elif y.ndim != 1:
                        raise ValueError(f"Unsupported label shape {y.shape}")

                    uniq = np.unique(y)
                    if not np.all(np.isin(uniq, [0, 1])):
                        raise ValueError(f"Non-binary labels {uniq.tolist()} (expected 0/1)")

                    N = len(y)

                    # ---- blocks ----
                    blocks = []
                    for key in self.use_blocks:
                        if key not in f:
                            raise ValueError(f"Missing requested block '{key}'")
                        arr = f[key][:]

                        # normalize shapes:
                        # - expected: (N, D)
                        # - allow: (N,) -> reshape (N,1)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        if arr.ndim != 2:
                            raise ValueError(f"{key} has shape {arr.shape}, expected 2D (N, D)")

                        if arr.shape[0] != N:
                            raise ValueError(f"Row mismatch: {key} has N={arr.shape[0]} but label has N={N}")

                        D = int(arr.shape[1])
                        if key not in self.block_dims:
                            self.block_dims[key] = D
                        else:
                            if D != self.block_dims[key]:
                                raise ValueError(
                                    f"Dim mismatch for '{key}': {D} != expected {self.block_dims[key]}"
                                )

                        blocks.append(arr.astype(np.float32, copy=False))

                    X = np.concatenate(blocks, axis=1)  # (N, sum(D_block))

                    feats_list.append(X.astype(np.float32, copy=False))
                    labels_list.append(y.astype(np.int64, copy=False))

            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e!r}")

        if not feats_list:
            raise ValueError("No usable .h5 files loaded. Check keys/shapes/labels.")

        X_all = np.concatenate(feats_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)

        self.features = torch.tensor(X_all, dtype=torch.float32)
        self.labels = torch.tensor(y_all, dtype=torch.long)
        self.input_dim = int(self.features.shape[1])

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DeepMLP_LayerNorm(nn.Module):
    """
    Linear -> LayerNorm -> ReLU -> Dropout
    """

    def __init__(
        self,
        input_dim: int,
        h1: int | None = None,
        h2: int | None = None,
        h3: int | None = None,
        p1: float = 0.35,
        p2: float = 0.30,
        p3: float = 0.25,
    ):
        super().__init__()

        if h1 is None:
            h1 = max(512, input_dim // 2)
        if h2 is None:
            h2 = max(256, h1 // 2)
        if h3 is None:
            h3 = max(128, h2 // 2)

        self.fc1 = nn.Linear(input_dim, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.drop1 = nn.Dropout(p1)

        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.drop2 = nn.Dropout(p2)

        self.fc3 = nn.Linear(h2, h3)
        self.ln3 = nn.LayerNorm(h3)
        self.drop3 = nn.Dropout(p3)

        self.fc4 = nn.Linear(h3, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.relu(self.ln1(self.fc1(x))))
        x = self.drop2(self.relu(self.ln2(self.fc2(x))))
        x = self.drop3(self.relu(self.ln3(self.fc3(x))))
        return self.fc4(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running += float(loss.item())
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    sen = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    # Force a 2x2 confusion matrix even if a split has only one class
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    pos_rate = float((y_pred == 1).mean()) if len(y_pred) else float("nan")

    report_txt = classification_report(y_true, y_pred, digits=4, zero_division=0)

    out: Dict[str, Any] = {
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "f1": f1,
        "auroc": auc,
        "pred_pos_rate": pos_rate,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(len(y_true)),
        "report": report_txt,
    }

    if verbose:
        print("  · Accuracy:    {:.4f}".format(acc))
        print("  · Sensitivity: {:.4f}".format(sen))
        print("  · Specificity: {:.4f}".format(spe))
        print("  · F1-Score:    {:.4f}".format(f1))
        print("  · AUROC:       {:.4f}".format(auc))
        print("  · Pred+ rate:  {:.4f}".format(pos_rate))
        print(report_txt)

    return out


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main(
    h5_dir: str,
    use_blocks: list[str],
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 1337,
    out_json: Optional[str] = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = PatchConcatDataset(h5_dir, use_blocks=use_blocks)
    print(f"[INFO] Using blocks: {dataset.use_blocks}")
    print(f"[INFO] Block dims: {dataset.block_dims}")
    print(f"[INFO] Loaded {len(dataset)} patches | input_dim={dataset.input_dim}")

    val_size = max(1, int(val_frac * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # ----- class weights from TRAIN split -----
    train_labels = torch.tensor([int(train_set[i][1]) for i in range(len(train_set))], dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=2).float()
    print(f"[INFO] Train class counts: {class_counts.tolist()}")

    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights * (class_weights.numel() / class_weights.sum())
    print(f"[INFO] Class weights (CE): {class_weights.tolist()}")
    # -----------------------------------------

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeepMLP_LayerNorm(dataset.input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_epoch = None
    best_state = None

    print("\n========== TRAINING ==========")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)

        print(f"\n[Epoch {epoch:02d}/{epochs}] Loss: {loss:.4f}")

        m = compute_metrics(y_true, y_pred, y_prob, verbose=True)
        cur_f1 = float(m["f1"])

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print("\n[WARN] No best_state captured (unexpected). Exiting.")
        return

    # Reload best checkpoint and re-evaluate on val, so metrics match the actual final model
    model.load_state_dict(best_state)
    print(f"\n[INFO] Loaded best model (val F1 = {best_f1:.4f} @ epoch {best_epoch})")

    print("\n========== BEST CHECKPOINT: VAL RE-EVAL ==========")
    y_true, y_pred, y_prob = evaluate(model, val_loader, device)
    best_val_metrics = compute_metrics(y_true, y_pred, y_prob, verbose=True)

    if out_json:
        payload: Dict[str, Any] = {
            "h5_dir": h5_dir,
            "use_blocks": list(dataset.use_blocks),
            "block_dims": dict(dataset.block_dims),
            "input_dim": int(dataset.input_dim),
            "n_total": int(len(dataset)),
            "n_train": int(len(train_set)),
            "n_val": int(len(val_set)),
            "seed": int(seed),
            "val_frac": float(val_frac),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "train_class_counts": [float(x) for x in class_counts.tolist()],
            "ce_class_weights": [float(x) for x in class_weights.tolist()],
            "selection_metric": "val_f1",
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_metrics": best_val_metrics,
        }
        _write_json(out_json, payload)
        print(f"[INFO] Wrote metrics JSON: {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir", required=True, help="Directory with .h5 files (stability output files).")
    parser.add_argument(
        "--use",
        nargs="+",
        default=["cv", "iqr", "mean", "std"],
        help=f"Which blocks to concatenate. Options: {list(ALLOWED_BLOCKS)}. "
             "Examples: --use features  OR  --use cv iqr mean std  OR  --use features cv iqr mean std",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--out_json",
        default=None,
        help="If set, write a JSON summary of BEST CHECKPOINT val metrics after reloading best weights.",
    )
    args = parser.parse_args()

    main(
        args.h5_dir,
        use_blocks=args.use,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        out_json=args.out_json,
    )
