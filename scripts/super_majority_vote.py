import os
import h5py
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ============ Helpers ============

def make_keys(length: int, coords: np.ndarray | None, fname_stem: str) -> np.ndarray:
    """
    Returns an array of stable keys for each row:
      - If coords provided, key = f"{fname}|{int(x)},{int(y)}"
      - Else key = f"{fname}|idx:{i}"
    """
    if coords is not None and coords.ndim >= 2 and coords.shape[1] >= 2:
        xs = np.rint(coords[:length, 0]).astype(int)
        ys = np.rint(coords[:length, 1]).astype(int)
        return np.array([f"{fname_stem}|{x},{y}" for x, y in zip(xs, ys)], dtype=object)
    else:
        return np.array([f"{fname_stem}|idx:{i}" for i in range(length)], dtype=object)

def stratified_split(y: np.ndarray, train_frac=0.8, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)

    def split(idx):
        n = len(idx)
        ntr = int(train_frac * n)
        nva = int(val_frac * n)
        return idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]

    tr0, va0, te0 = split(idx0)
    tr1, va1, te1 = split(idx1)

    train = np.concatenate([tr0, tr1]); rng.shuffle(train)
    val   = np.concatenate([va0, va1]); rng.shuffle(val)
    test  = np.concatenate([te0, te1]); rng.shuffle(test)
    return train, val, test

# ============ Dataset/Model ============

class FeatureDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        assert feats.shape[0] == len(labels)
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ===========================================
# MLP Model Definition
# ===========================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.model(x)


# ============ Alignment (drop misaligned only) ============

def load_aligned_across_dirs(feature_dirs: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align by (filename, coords) if available; else by (filename, row-index).
    Drop keys not present in ALL dirs (i.e., misaligned). Keep everything else.
    Labels are taken by per-sample majority (no drops for label disagreement).
    Returns:
      feats_per_dir: list of (N, D_j)
      y:            (N,)
    """
    assert len(feature_dirs) == 9, "Expect 9 dirs (3 models × 3 magnifications)."

    base = feature_dirs[0]
    file_names = [f for f in os.listdir(base) if f.endswith(".h5")]

    per_dir_chunks: List[list[np.ndarray]] = [[] for _ in range(9)]
    y_chunks: list[np.ndarray] = []

    files_used = 0
    total_aligned = 0
    total_dropped = 0

    for fname in file_names:
        # All dirs must contain this file; otherwise it's not "misaligned", it's missing — skip the file.
        paths = [os.path.join(d, fname) for d in feature_dirs]
        if not all(os.path.exists(p) for p in paths):
            continue

        try:
            # Load each dir: feats, labels, coords, keys
            per_dir = []
            key_sets = []
            key2row_per_dir: List[Dict[str, int]] = []

            for p in paths:
                with h5py.File(p, "r") as f:
                    if "features" not in f or "labels" not in f:
                        raise RuntimeError(f"{p} missing features/labels")
                    feats = np.asarray(f["features"][:])
                    labels = np.asarray(f["labels"][:]).astype(int).flatten()
                    coords = np.asarray(f["coords"][:]) if "coords" in f else None

                    m = min(len(feats), len(labels))
                    feats, labels = feats[:m], labels[:m]
                    keys = make_keys(m, coords, os.path.splitext(fname)[0])
                    per_dir.append({"feats": feats, "labels": labels, "keys": keys})
                    key_sets.append(set(keys))
                    key2row_per_dir.append({k: i for i, k in enumerate(keys)})

            # Intersection = aligned patches; everything else is "misaligned" and dropped
            aligned_keys = sorted(set.intersection(*key_sets))
            if len(aligned_keys) == 0:
                continue

            # Build arrays in aligned order
            feats_blocks = []
            labels_blocks = []
            for d in range(9):
                rows = [key2row_per_dir[d][k] for k in aligned_keys]
                feats_blocks.append(per_dir[d]["feats"][rows])
                labels_blocks.append(per_dir[d]["labels"][rows])

            # Majority label per sample (0/1)
            label_mat = np.stack(labels_blocks, axis=0)  # (9, Nfile)
            votes_for_one = label_mat.sum(axis=0)
            y_file = (votes_for_one >= 5).astype(int)

            for d in range(9):
                per_dir_chunks[d].append(feats_blocks[d])
            y_chunks.append(y_file)

            files_used += 1
            # approximate dropped count per file = (min available per-dir) - aligned
            min_len = min(len(s) for s in key_sets)
            dropped = max(min_len - len(aligned_keys), 0)
            kept = len(aligned_keys)
            total_aligned += kept
            total_dropped += dropped
            if dropped > 0:
                print(f"[INFO] {fname}: kept (aligned) {kept}, dropped (misaligned) ~{dropped}")

        except Exception as e:
            warnings.warn(f"[WARN] skipping {fname}: {e}")

    if files_used == 0 or len(y_chunks) == 0:
        raise ValueError("No aligned patches found across all 9 dirs.")

    feats_per_dir = [np.concatenate(chs, axis=0) for chs in per_dir_chunks]
    y = np.concatenate(y_chunks, axis=0)

    print(f"[INFO] Files used: {files_used}")
    print(f"[INFO] Total aligned patches kept: {total_aligned:,} | misaligned dropped (approx): {total_dropped:,}")
    print(f"[INFO] Final dataset size: {len(y):,}")
    return feats_per_dir, y

# ============ Train / Eval / Soft Vote ============

def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)

    best_acc = -1
    best_state = None
    for ep in range(epochs):
        model.train()
        run = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            run += loss.item()

        model.eval()
        vpred, vtrue = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = torch.argmax(model(X), dim=1)
                vpred.extend(pred.cpu().numpy()); vtrue.extend(y.cpu().numpy())
        vacc = accuracy_score(vtrue, vpred)
        print(f"Epoch {ep+1:02d}: train_loss={run/len(train_loader):.4f} | val_acc={vacc:.4f}")
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_proba(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (p1, y_true) for the loader."""
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = torch.softmax(model(X), dim=1)[:, 1].cpu().numpy()
            probs.extend(p); y_true.extend(y.numpy())
    return np.array(probs), np.array(y_true)

def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    auc  = roc_auc_score(y_true, y_score)
    print("\n=== Soft Majority Vote (mean prob across 9 models) ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    return acc, sens, spec, f1, auc

# ============ Orchestrate ============

def main(feature_dirs: List[str], epochs=20, batch_size=256, lr=1e-4, seed=42):
    assert len(feature_dirs) == 9, "Pass 9 dirs (3 models × 3 mags)."
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Align (drop misaligned only)
    feats_per_dir, y = load_aligned_across_dirs(feature_dirs)

    # 2) Shared stratified split
    tr, va, te = stratified_split(y, train_frac=0.8, val_frac=0.1, seed=seed)

    # 3) Train one model per dir; collect test probs
    test_probs_all = []
    y_true_ref = None

    for i, feats in enumerate(feats_per_dir):
        print(f"\n=== Training model {i+1}/9 (D={feats.shape[1]}) ===")
        ds = FeatureDataset(feats, y)
        tl = DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=True)
        vl = DataLoader(Subset(ds, va), batch_size=batch_size, shuffle=False)
        te_loader = DataLoader(Subset(ds, te), batch_size=batch_size, shuffle=False)

        model = DeepMLP(input_dim=feats.shape[1])
        model = train_model(model, tl, vl, device, epochs=epochs, lr=lr)

        p1, y_true = predict_proba(model, te_loader, device)
        test_probs_all.append(p1)
        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true_ref, y_true):
            warnings.warn("[WARN] test label order mismatch across models; using first as reference.")

    '''# 4) Soft vote: mean probability across 9 models
    probs_stack = np.stack(test_probs_all, axis=0)     # (9, N_test)
    mean_probs  = probs_stack.mean(axis=0)             # (N_test,)
    compute_metrics(y_true_ref, mean_probs)'''

    probs_stack = np.stack(test_probs_all, axis=0)             # (9, N_test)
    binary_preds = (probs_stack >= 0.5).astype(int)            # (9, N_test)
    votes_for_one = binary_preds.sum(axis=0)                   # (# of models voting 1 per sample)
    hard_preds = (votes_for_one >= 5).astype(int)              # majority threshold = 5/9

    # Compute metrics using hard predictions (no probability weighting)
    acc = accuracy_score(y_true_ref, hard_preds)
    sens = recall_score(y_true_ref, hard_preds)
    f1   = f1_score(y_true_ref, hard_preds)
    cm   = confusion_matrix(y_true_ref, hard_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n=== Hard Majority Vote (≥5 of 9 models) ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1:          {f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Align patches across 9 dirs (drop misaligned only) and soft-vote (mean prob) 9 MLPs."
    )
    parser.add_argument("--dirs", nargs=9, required=True,
                        help="Nine feature dirs (e.g., CONCH/MUSK/Hoptimus × 5x/10x/20x).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dirs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)


