#!/usr/bin/env python3
# cv_iqr_mean_std_lstm_classifier.py
# ------------------------------------------------------------
# Train an LSTM-based "block fusion" classifier on patch-level
# features from a directory of .h5 files.
#
# Instead of concatenating blocks along feature dimension, we:
#   (1) Project each block into a shared token space (d_model)
#   (2) Treat blocks as a short sequence of tokens (T = #blocks)
#   (3) Feed sequence through an LSTM to get a fused embedding
#   (4) Feed embedding into a downstream classifier head
#
# Expected keys in each .h5 (stability builder output):
#   required: label
#   optional: features (baseline), cv, iqr, mean, std, coords
#
# You choose which blocks to use via --use (order matters; it is
# the LSTM "sequence" order).
#
# Loss:
#   class-weighted CrossEntropyLoss (no WeightedRandomSampler)
#
# Output:
#   prints val metrics each epoch; selects best by val F1; reloads
#   best weights and re-evaluates; optionally writes JSON summary.
# ------------------------------------------------------------

import os
import json
from typing import Any, Dict, Optional, List, Tuple

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


# -------------------------
# Dataset: blocks as tokens
# -------------------------
class BlockSequenceDataset(Dataset):
    """
    Loads patch-level blocks but returns them as a dict per sample:
      x = { block_name: tensor(D_block,) }
      y = tensor() long
    """

    def __init__(self, h5_dir: str, use_blocks: List[str]):
        use_blocks = list(use_blocks)
        if not use_blocks:
            raise ValueError("use_blocks is empty. Provide at least one block to use.")
        bad = [b for b in use_blocks if b not in ALLOWED_BLOCKS]
        if bad:
            raise ValueError(f"Unknown blocks {bad}. Allowed: {list(ALLOWED_BLOCKS)}")

        files = [f for f in os.listdir(h5_dir) if f.lower().endswith((".h5", ".hdf5"))]
        if not files:
            raise ValueError(f"No .h5/.hdf5 files found in: {h5_dir}")

        self.use_blocks = use_blocks
        self.block_dims: Dict[str, int] = {}

        blocks_accum: Dict[str, List[np.ndarray]] = {k: [] for k in use_blocks}
        labels_list: List[np.ndarray] = []

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
                    for key in self.use_blocks:
                        if key not in f:
                            raise ValueError(f"Missing requested block '{key}' in {fname}")
                        arr = f[key][:]

                        # normalize to (N, D)
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

                        blocks_accum[key].append(arr.astype(np.float32, copy=False))

                    labels_list.append(y.astype(np.int64, copy=False))

            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e!r}")

        if not labels_list:
            raise ValueError("No usable .h5 files loaded. Check keys/shapes/labels.")

        # Concatenate across files
        y_all = np.concatenate(labels_list, axis=0)
        self.labels = torch.tensor(y_all, dtype=torch.long)

        self.blocks: Dict[str, torch.Tensor] = {}
        for key in self.use_blocks:
            Xk = np.concatenate(blocks_accum[key], axis=0)
            self.blocks[key] = torch.tensor(Xk, dtype=torch.float32)

        self.n = int(self.labels.shape[0])

        # sanity
        for key in self.use_blocks:
            if int(self.blocks[key].shape[0]) != self.n:
                raise ValueError(f"Internal error: block {key} N mismatch")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x = {k: self.blocks[k][idx] for k in self.use_blocks}  # each is (D_k,)
        y = self.labels[idx]
        return x, y


def block_collate_fn(batch):
    """
    batch: list of (x_dict, y)
    returns:
      X_dict: {k: (B, D_k)}
      y: (B,)
    """
    xs, ys = zip(*batch)
    keys = xs[0].keys()
    out_x = {}
    for k in keys:
        out_x[k] = torch.stack([x[k] for x in xs], dim=0)
    out_y = torch.stack(list(ys), dim=0)
    return out_x, out_y


# -------------------------
# Model: projection -> LSTM
# -------------------------
class BlockLSTMClassifier(nn.Module):
    """
    For each block k with dim Dk:
      token_k = LN(Wk x_k + bk) -> ReLU -> Dropout -> (B, d_model)
    Stack tokens in order:
      seq = (B, T, d_model)
    LSTM(seq) -> final hidden -> head -> logits (B,2)
    """

    def __init__(
        self,
        use_blocks: List[str],
        block_dims: Dict[str, int],
        d_model: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        token_dropout: float = 0.10,
        head_hidden: int = 256,
        head_dropout: float = 0.25,
    ):
        super().__init__()
        self.use_blocks = list(use_blocks)
        self.block_dims = dict(block_dims)
        self.d_model = int(d_model)

        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        # Per-block projection to shared token dim
        self.proj = nn.ModuleDict()
        self.proj_ln = nn.ModuleDict()
        for k in self.use_blocks:
            dk = int(self.block_dims[k])
            self.proj[k] = nn.Linear(dk, self.d_model)
            self.proj_ln[k] = nn.LayerNorm(self.d_model)

        self.token_dropout = nn.Dropout(float(token_dropout))
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=0.0 if int(lstm_layers) == 1 else 0.10,
        )

        lstm_out_dim = int(lstm_hidden) * self.num_directions

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, int(head_hidden)),
            nn.LayerNorm(int(head_hidden)),
            nn.ReLU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(head_hidden), 2),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for k in self.use_blocks:
            xk = x_dict[k]  # (B, Dk)
            tk = self.proj_ln[k](self.proj[k](xk))  # (B, d_model)
            tk = self.token_dropout(self.relu(tk))
            tokens.append(tk)

        # (B, T, d_model)
        seq = torch.stack(tokens, dim=1)

        # LSTM
        _, (h_n, _) = self.lstm(seq)
        # h_n: (num_layers * num_directions, B, hidden)

        if self.bidirectional:
            # last layer forward/backward are last two entries
            h_f = h_n[-2]  # (B, hidden)
            h_b = h_n[-1]  # (B, hidden)
            emb = torch.cat([h_f, h_b], dim=1)  # (B, 2*hidden)
        else:
            emb = h_n[-1]  # (B, hidden)

        return self.head(emb)


# -------------------------
# Train / Eval helpers
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for X_dict, y in loader:
        X_dict = {k: v.to(device) for k, v in X_dict.items()}
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X_dict)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for X_dict, y in loader:
        X_dict = {k: v.to(device) for k, v in X_dict.items()}
        logits = model(X_dict)
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


# -------------------------
# Main
# -------------------------
def main(
    h5_dir: str,
    use_blocks: List[str],
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 1337,
    out_json: Optional[str] = None,
    d_model: int = 256,
    lstm_hidden: int = 256,
    lstm_layers: int = 1,
    bidirectional: bool = False,
    token_dropout: float = 0.10,
    head_hidden: int = 256,
    head_dropout: float = 0.25,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = BlockSequenceDataset(h5_dir, use_blocks=use_blocks)
    print(f"[INFO] Using blocks (sequence order): {dataset.use_blocks}")
    print(f"[INFO] Block dims: {dataset.block_dims}")
    print(f"[INFO] Loaded {len(dataset)} patches")

    val_size = max(1, int(val_frac * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # ----- class weights from TRAIN split -----
    # NOTE: train_set is a Subset, so we query labels via __getitem__
    train_labels = torch.tensor([int(train_set[i][1]) for i in range(len(train_set))], dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=2).float()
    print(f"[INFO] Train class counts: {class_counts.tolist()}")

    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights * (class_weights.numel() / class_weights.sum())
    print(f"[INFO] Class weights (CE): {class_weights.tolist()}")
    # -----------------------------------------

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=block_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=block_collate_fn,
    )

    model = BlockLSTMClassifier(
        use_blocks=dataset.use_blocks,
        block_dims=dataset.block_dims,
        d_model=d_model,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        token_dropout=token_dropout,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    ).to(device)

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
            "model": {
                "type": "BlockLSTMClassifier",
                "d_model": int(d_model),
                "lstm_hidden": int(lstm_hidden),
                "lstm_layers": int(lstm_layers),
                "bidirectional": bool(bidirectional),
                "token_dropout": float(token_dropout),
                "head_hidden": int(head_hidden),
                "head_dropout": float(head_dropout),
            },
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
        help=f"Which blocks to use as the token sequence (order matters). Options: {list(ALLOWED_BLOCKS)}. "
             "Examples: --use features  OR  --use cv iqr mean std  OR  --use features cv iqr mean std",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)

    # Model hyperparams
    parser.add_argument("--d_model", type=int, default=256, help="Shared token dimension after per-block projection.")
    parser.add_argument("--lstm_hidden", type=int, default=256, help="LSTM hidden size.")
    parser.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM.")
    parser.add_argument("--token_dropout", type=float, default=0.10, help="Dropout applied to projected tokens.")
    parser.add_argument("--head_hidden", type=int, default=256, help="Hidden size for the classifier head MLP.")
    parser.add_argument("--head_dropout", type=float, default=0.25, help="Dropout in the classifier head.")

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
        d_model=args.d_model,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        token_dropout=args.token_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    )
