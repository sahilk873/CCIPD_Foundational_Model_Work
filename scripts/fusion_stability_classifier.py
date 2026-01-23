#!/usr/bin/env python3
# fusion_stability_classifier.py
# ------------------------------------------------------------
# Train a classifier on patch-level feature blocks from a directory of .h5 files.
#
# Expected keys in each .h5:
#   required: label
#   optional blocks: features, cv, iqr, mean, std
#
# Choose which blocks to include via --use.
#
# Fusion modes (choose via --fusion):
#   1) concat     : concatenate selected blocks, then a LayerNorm MLP (baseline)
#   2) multihead  : per-block encoder heads -> concat(latents) -> fusion MLP
#   3) gated      : per-block encoder heads -> temperature-softmax gates -> weighted sum -> classifier
#
# This version makes the learnable MLPs (heads, fusion, gates, classifier) "deep + wide"
# using the SAME depth/width heuristic as your cv_iqr_mean_std_mlp_classifier.py:
#   h1 = max(512, input_dim//2), h2 = max(256, h1//2), h3 = max(128, h2//2)
# with LayerNorm + ReLU + Dropout on hidden layers.
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
# Dataset: returns per-block tensors (not concatenated)
# -------------------------
class PatchBlockDataset(Dataset):
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
        self._X_blocks: Dict[str, List[np.ndarray]] = {k: [] for k in self.use_blocks}
        self._y_list: List[np.ndarray] = []

        for fname in sorted(files):
            path = os.path.join(h5_dir, fname)
            try:
                with h5py.File(path, "r") as f:
                    if "label" not in f:
                        print(f"[WARN] Skipping {fname}: missing key ['label']")
                        continue

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

                    N = int(len(y))

                    loaded_blocks: Dict[str, np.ndarray] = {}
                    for key in self.use_blocks:
                        if key not in f:
                            raise ValueError(f"Missing requested block '{key}'")
                        arr = f[key][:]

                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        if arr.ndim != 2:
                            raise ValueError(f"{key} has shape {arr.shape}, expected 2D (N, D)")
                        if int(arr.shape[0]) != N:
                            raise ValueError(f"Row mismatch: {key} has N={arr.shape[0]} but label has N={N}")

                        D = int(arr.shape[1])
                        if key not in self.block_dims:
                            self.block_dims[key] = D
                        else:
                            if D != self.block_dims[key]:
                                raise ValueError(f"Dim mismatch for '{key}': {D} != expected {self.block_dims[key]}")

                        loaded_blocks[key] = arr.astype(np.float32, copy=False)

                    for key in self.use_blocks:
                        self._X_blocks[key].append(loaded_blocks[key])
                    self._y_list.append(y.astype(np.int64, copy=False))

            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e!r}")

        if not self._y_list:
            raise ValueError("No usable .h5 files loaded. Check keys/shapes/labels.")

        self.y = np.concatenate(self._y_list, axis=0)
        self.X_blocks: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in self._X_blocks.items()}

        n_all = int(len(self.y))
        for k in self.use_blocks:
            if int(self.X_blocks[k].shape[0]) != n_all:
                raise RuntimeError(f"Internal error: block {k} rows != labels rows")

        self.n = n_all

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        xb = {k: torch.tensor(self.X_blocks[k][idx], dtype=torch.float32) for k in self.use_blocks}
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return xb, y


# -------------------------
# Deep+Wide building block (same heuristic as your other script)
# -------------------------
class DeepMLP_LayerNorm(nn.Module):
    """
    Linear -> LayerNorm -> ReLU -> Dropout  (x3) -> Linear(out)
    Width heuristic:
      h1 = max(512, in//2), h2 = max(256, h1//2), h3 = max(128, h2//2)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        h1: Optional[int] = None,
        h2: Optional[int] = None,
        h3: Optional[int] = None,
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

        self.fc4 = nn.Linear(h3, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.relu(self.ln1(self.fc1(x))))
        x = self.drop2(self.relu(self.ln2(self.fc2(x))))
        x = self.drop3(self.relu(self.ln3(self.fc3(x))))
        return self.fc4(x)


# -------------------------
# Models
# -------------------------
class BlockEncoder(nn.Module):
    """
    Encodes one block (D_in) -> z_dim with the same deep+wide heuristic.
    """
    def __init__(self, in_dim: int, z_dim: int, p: float = 0.25):
        super().__init__()
        # output is z_dim; we add an LN+ReLU after the projection for stability
        self.mlp = DeepMLP_LayerNorm(input_dim=in_dim, output_dim=z_dim, p1=p, p2=p, p3=p)
        self.out_ln = nn.LayerNorm(z_dim)
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        return self.out_relu(self.out_ln(z))


class MultiHeadFusion(nn.Module):
    """
    Per-block encoders -> concat latents -> deep+wide fusion MLP -> logits
    """
    def __init__(
        self,
        block_dims: Dict[str, int],
        use_blocks: List[str],
        z_dim: int = 128,
        p: float = 0.25,
    ):
        super().__init__()
        self.use_blocks = list(use_blocks)
        self.encoders = nn.ModuleDict({k: BlockEncoder(block_dims[k], z_dim=z_dim, p=p) for k in self.use_blocks})

        fused_in = int(z_dim * len(self.use_blocks))
        self.fusion = DeepMLP_LayerNorm(input_dim=fused_in, output_dim=2, p1=p, p2=p, p3=p)

    def forward(self, xb: Dict[str, torch.Tensor]) -> torch.Tensor:
        zs = [self.encoders[k](xb[k]) for k in self.use_blocks]
        z = torch.cat(zs, dim=1)
        return self.fusion(z)


class GatedFusion(nn.Module):
    """
    Per-block encoders -> temperature-softmax gates -> weighted sum of latents -> deep+wide classifier

    Logs:
      - last_gate_weights: (B, K) detached tensor, set every forward call
    """
    def __init__(
        self,
        block_dims: Dict[str, int],
        use_blocks: List[str],
        z_dim: int = 128,
        p: float = 0.25,
        init_temp: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.use_blocks = list(use_blocks)
        self.K = len(self.use_blocks)
        self.z_dim = int(z_dim)
        self.eps = float(eps)

        self.encoders = nn.ModuleDict({k: BlockEncoder(block_dims[k], z_dim=z_dim, p=p) for k in self.use_blocks})

        # gate input: concat of per-block latents (K*z_dim) -> logits (K)
        self.gate = DeepMLP_LayerNorm(input_dim=self.K * self.z_dim, output_dim=self.K, p1=p, p2=p, p3=p)

        # classifier on gated latent z_dim -> logits (2)
        self.classifier = DeepMLP_LayerNorm(input_dim=self.z_dim, output_dim=2, p1=p, p2=p, p3=p)

        self.temperature = float(init_temp)
        self.last_gate_weights: Optional[torch.Tensor] = None

    def set_temperature(self, T: float) -> None:
        self.temperature = float(max(self.eps, T))

    def forward(self, xb: Dict[str, torch.Tensor]) -> torch.Tensor:
        zs = [self.encoders[k](xb[k]) for k in self.use_blocks]          # list of (B, z)
        zcat = torch.cat(zs, dim=1)                                      # (B, K*z)

        gate_logits = self.gate(zcat)                                    # (B, K)
        T = float(max(self.eps, self.temperature))
        w = torch.softmax(gate_logits / T, dim=1)                        # (B, K)
        self.last_gate_weights = w.detach()

        zstack = torch.stack(zs, dim=1)                                  # (B, K, z)
        z = (w.unsqueeze(-1) * zstack).sum(dim=1)                        # (B, z)

        return self.classifier(z)


class ConcatWrapper(nn.Module):
    """
    Accepts dict blocks, concatenates in a fixed order, then a deep+wide LayerNorm MLP.
    """
    def __init__(self, block_dims: Dict[str, int], use_blocks: List[str], p: float = 0.25):
        super().__init__()
        self.use_blocks = list(use_blocks)
        input_dim = int(sum(block_dims[k] for k in self.use_blocks))
        self.model = DeepMLP_LayerNorm(input_dim=input_dim, output_dim=2, p1=p, p2=p, p3=p)

    def forward(self, xb: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([xb[k] for k in self.use_blocks], dim=1)
        return self.model(x)


# -------------------------
# Temperature schedule helper (gated)
# -------------------------
def gate_temperature_at_epoch(
    epoch: int,
    epochs: int,
    t_start: float,
    t_end: float,
    schedule: str = "linear",
) -> float:
    e = max(1, int(epoch))
    E = max(1, int(epochs))
    p = (e - 1) / max(1, (E - 1))  # 0..1

    schedule = schedule.lower().strip()
    if schedule == "linear":
        return float(t_start + (t_end - t_start) * p)

    if schedule == "cosine":
        import math
        return float(t_end + (t_start - t_end) * (0.5 * (1 + math.cos(math.pi * p))))

    if schedule == "exp":
        import math
        if t_start <= 0 or t_end <= 0:
            return float(t_end)
        if E == 1:
            return float(t_end)
        ratio = (t_end / t_start) ** (1 / (E - 1))
        return float(t_start * (ratio ** (e - 1)))

    raise ValueError("Unknown temperature schedule. Use: linear, cosine, exp")


# -------------------------
# Train / Eval
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0

    gate_sum = None
    gate_count = 0

    for xb, y in loader:
        xb = {k: v.to(device) for k, v in xb.items()}
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item())

        if hasattr(model, "last_gate_weights") and getattr(model, "last_gate_weights") is not None:
            w = model.last_gate_weights  # (B, K)
            w_mean = w.mean(dim=0)       # (K,)
            gate_sum = w_mean.clone() if gate_sum is None else (gate_sum + w_mean)
            gate_count += 1

    avg_loss = running / max(1, len(loader))

    avg_gate = None
    if gate_sum is not None and gate_count > 0:
        avg_gate = (gate_sum / gate_count).detach().cpu().numpy()

    return avg_loss, avg_gate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for xb, y in loader:
        xb = {k: v.to(device) for k, v in xb.items()}
        logits = model(xb)
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


def build_model(
    fusion: str,
    block_dims: Dict[str, int],
    use_blocks: List[str],
    z_dim: int,
    dropout: float,
    t_start: float,
) -> nn.Module:
    fusion = fusion.lower().strip()
    if fusion == "concat":
        return ConcatWrapper(block_dims=block_dims, use_blocks=use_blocks, p=dropout)
    if fusion == "multihead":
        return MultiHeadFusion(block_dims=block_dims, use_blocks=use_blocks, z_dim=z_dim, p=dropout)
    if fusion == "gated":
        return GatedFusion(
            block_dims=block_dims,
            use_blocks=use_blocks,
            z_dim=z_dim,
            p=dropout,
            init_temp=t_start,
        )
    raise ValueError("Unknown --fusion. Choose from: concat, multihead, gated")


def main(
    h5_dir: str,
    use_blocks: List[str],
    fusion: str,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 1337,
    out_json: Optional[str] = None,
    z_dim: int = 128,
    dropout: float = 0.25,
    temp_schedule: str = "linear",
    t_start: float = 2.0,
    t_end: float = 1.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = PatchBlockDataset(h5_dir, use_blocks=use_blocks)
    print(f"[INFO] Using blocks: {dataset.use_blocks}")
    print(f"[INFO] Block dims: {dataset.block_dims}")
    print(f"[INFO] Loaded {len(dataset)} patches")
    print(f"[INFO] Fusion mode: {fusion}")

    val_size = max(1, int(val_frac * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # class weights from TRAIN split
    train_labels = torch.tensor([int(train_set[i][1]) for i in range(len(train_set))], dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=2).float()
    print(f"[INFO] Train class counts: {class_counts.tolist()}")

    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights * (class_weights.numel() / class_weights.sum())
    print(f"[INFO] Class weights (CE): {class_weights.tolist()}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(
        fusion=fusion,
        block_dims=dataset.block_dims,
        use_blocks=dataset.use_blocks,
        z_dim=z_dim,
        dropout=dropout,
        t_start=t_start,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_epoch = None
    best_state = None

    gate_history: List[Dict[str, Any]] = []

    print("\n========== TRAINING ==========")
    for epoch in range(1, epochs + 1):
        cur_temp = None
        if fusion.lower() == "gated" and hasattr(model, "set_temperature"):
            cur_temp = gate_temperature_at_epoch(
                epoch=epoch,
                epochs=epochs,
                t_start=t_start,
                t_end=t_end,
                schedule=temp_schedule,
            )
            model.set_temperature(cur_temp)

        loss, avg_gate = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)

        print(f"\n[Epoch {epoch:02d}/{epochs}] Loss: {loss:.4f}")

        if fusion.lower() == "gated" and avg_gate is not None:
            pairs = list(zip(dataset.use_blocks, [float(x) for x in avg_gate.tolist()]))
            pairs_str = " | ".join([f"{k}:{v:.3f}" for k, v in pairs])
            if cur_temp is not None:
                print(f"  · Gate T: {cur_temp:.3f} | Avg gates: {pairs_str}")
            else:
                print(f"  · Avg gates: {pairs_str}")

            gate_history.append({
                "epoch": int(epoch),
                "temperature": float(cur_temp) if cur_temp is not None else None,
                "avg_gate": {k: float(v) for k, v in zip(dataset.use_blocks, avg_gate.tolist())},
            })

        m = compute_metrics(y_true, y_pred, y_prob, verbose=True)
        cur_f1 = float(m["f1"])

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print("\n[WARN] No best_state captured (unexpected). Exiting.")
        return

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
            "fusion": str(fusion),
            "model_hparams": {
                "z_dim": int(z_dim),
                "dropout": float(dropout),
            },
            "best_val_metrics": best_val_metrics,
        }

        if fusion.lower() == "gated":
            payload["gate_temperature"] = {
                "schedule": str(temp_schedule),
                "t_start": float(t_start),
                "t_end": float(t_end),
            }
            payload["gate_history"] = gate_history

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
        help=f"Which blocks to use. Options: {list(ALLOWED_BLOCKS)}. "
             "Examples: --use features  OR  --use cv iqr mean std  OR  --use features cv iqr mean std",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "multihead", "gated"],
        help="Fusion strategy: concat, multihead, gated",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_json", default=None, help="Write a JSON summary of best-checkpoint val metrics.")

    # Main hyperparams
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dim for each block head (multihead/gated).")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout used throughout.")

    # Gated temperature scheduling
    parser.add_argument(
        "--temp_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "exp"],
        help="Gate temperature schedule (gated fusion only).",
    )
    parser.add_argument("--t_start", type=float, default=2.0, help="Starting temperature (gated fusion only).")
    parser.add_argument("--t_end", type=float, default=1.0, help="Ending temperature (gated fusion only).")

    args = parser.parse_args()

    main(
        h5_dir=args.h5_dir,
        use_blocks=args.use,
        fusion=args.fusion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        out_json=args.out_json,
        z_dim=args.z_dim,
        dropout=args.dropout,
        temp_schedule=args.temp_schedule,
        t_start=args.t_start,
        t_end=args.t_end,
    )
