import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import List, Tuple


# ===========================================
# Dataset Class
# ===========================================
class FeatureDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        # Ensure proper shapes
        labels = np.asarray(labels).reshape(-1)
        assert len(feats) == len(labels), "Features and labels must have the same number of rows."
        self.X = torch.tensor(np.asarray(feats), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.y):
            raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.y)}")
        return self.X[idx], self.y[idx]


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


# ===========================================
# Load Features from Directory
# ===========================================
def load_features_from_dir(h5_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    all_feats, all_labels = [], []
    for fname in os.listdir(h5_dir):
        if not fname.endswith(".h5"):
            continue
        fpath = os.path.join(h5_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                if "features" in f and "labels" in f:
                    feats = f["features"][:]
                    labels = f["labels"][:]
                    feats = np.asarray(feats)
                    labels = np.asarray(labels).reshape(-1)

                    # Truncate to shared min length within file if needed
                    m = min(len(feats), len(labels))
                    if m == 0:
                        continue
                    if m != len(feats) or m != len(labels):
                        print(f"⚠️ Truncating {fname} to {m} due to feature/label length mismatch")

                    all_feats.append(feats[:m])
                    all_labels.append(labels[:m])
                else:
                    print(f"⚠️ {fname} missing 'features' or 'labels'; skipping")
        except Exception as e:
            print(f"❌ Error reading {fpath}: {e}")

    if not all_feats:
        raise ValueError(f"No valid .h5 files found in {h5_dir}")

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(int)
    return feats, labels


# ===========================================
# Training Function
# ===========================================
def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/max(1,len(train_loader)):.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    return model


# ===========================================
# Evaluate Model
# ===========================================
def evaluate_model(model, test_loader, device):
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            probs.extend(prob)
            labels.extend(y.numpy())
    return np.array(preds), np.array(probs), np.array(labels)


# ===========================================
# Metrics
# ===========================================
def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix/specificity (guard single-class case)
    labels_present = np.unique(np.concatenate([y_true, y_pred]))
    if set(labels_present.tolist()) == {0, 1}:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        cm = confusion_matrix(y_true, y_pred)
        specificity = float("nan")

    # AUC (guard single-class)
    try:
        auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    except ValueError:
        auc = float("nan")

    print("\n=== Ensemble Performance ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Sensitivity:  {recall:.4f}")
    print(f"Specificity:  {specificity if specificity==specificity else float('nan'):.4f}")
    print(f"F1:           {f1:.4f}")
    print(f"AUC:          {auc if auc==auc else float('nan'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return acc, recall, specificity, f1, auc


# ===========================================
# Main
# ===========================================
def main(feature_dirs: List[str]):
    assert len(feature_dirs) == 3, "--dirs must have exactly three directories"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load all dirs first
    loaded = []
    lengths = []
    for d in feature_dirs:
        feats, labels = load_features_from_dir(d)
        loaded.append((feats, labels))
        lengths.append(len(labels))
        print(f"[INFO] Loaded {len(labels):,} samples from {d}")

    # Use a shared split based on the smallest available length across dirs
    min_len = min(lengths)
    if len(set(lengths)) != 1:
        print(f"⚠️ Length mismatch across dirs {lengths}; using common prefix length min_len={min_len}")
    # Truncate each directory’s arrays to min_len (keeps alignment by order)
    loaded = [(f[:min_len], l[:min_len]) for (f, l) in loaded]

    # Create shared indices & split once
    indices = np.arange(min_len)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    n_train = int(0.8 * min_len)
    n_val = int(0.1 * min_len)
    n_test = min_len - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    preds_all, probs_all, labels_all = [], [], []

    # Training per directory (per model)
    for i, (h5_dir, (feats, labels)) in enumerate(zip(feature_dirs, loaded)):
        print(f"\n=== Training Model {i+1} on {h5_dir} ===")

        dataset = FeatureDataset(feats, labels)
        train_set = Subset(dataset, train_idx.tolist())
        val_set = Subset(dataset, val_idx.tolist())
        test_set = Subset(dataset, test_idx.tolist())

        # Dataloaders
        pin_mem = (device.type == "cuda")
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=pin_mem, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, pin_memory=pin_mem, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=pin_mem, num_workers=2)

        input_dim = feats.shape[1]
        model = DeepMLP(input_dim=input_dim)
        model = train_model(model, train_loader, val_loader, device, epochs=20)

        preds, probs, labels_np = evaluate_model(model, test_loader, device)
        preds_all.append(preds)
        probs_all.append(probs)
        labels_all.append(labels_np)

    # Shared labels (all models used the same test split)
    y_true = labels_all[0]

    # Soft Voting Ensemble
    probs_stack = np.stack(probs_all, axis=1)  # shape: [N_test, n_models]
    mean_probs = probs_stack.mean(axis=1)
    y_pred_ens = (mean_probs >= 0.5).astype(int)

    compute_metrics(y_true, y_pred_ens, mean_probs)


# ===========================================
# CLI
# ===========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs=3, required=True, help="Three feature directories for model training")
    args = parser.parse_args()

    # Convert to list if needed
    dirs = list(args.dirs) if isinstance(args.dirs, (list, tuple)) else [args.dirs]
    main(dirs)
