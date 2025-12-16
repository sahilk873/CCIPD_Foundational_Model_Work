import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================
# Dataset Class
# ============================
'''class FusedFeatureDataset(Dataset):
    def __init__(self, dirs):
        """
        dirs: list of directories, each containing .h5 files with 'features' and 'labels' arrays.
        """
        features_list, labels_list = [], []
        base_dir = dirs[0]
        file_names = [f for f in os.listdir(base_dir) if f.endswith(".h5")]

        for fname in file_names:
            fused_features = []
            label_ref = None
            skip = False

            for d in dirs:
                full_path = os.path.join(d, fname)
                if not os.path.exists(full_path):
                    skip = True
                    break
                try:
                    with h5py.File(full_path, "r") as f:
                        if "features" not in f or "labels" not in f:
                            print(f"⚠️ Skipping {fname} in {d}: missing 'features' or 'labels'")
                            skip = True
                            break
                        feats = f["features"][:]
                        labels = f["labels"][:]
                        if label_ref is None:
                            label_ref = labels
                        elif not np.array_equal(label_ref, labels):
                            print(f"⚠️ Label mismatch for {fname} across dirs; skipping.")
                            skip = True
                            break
                        fused_features.append(feats)
                except Exception as e:
                    print(f"❌ Error reading {fname} in {d}: {e}")
                    skip = True
                    break

            if skip or len(fused_features) != len(dirs):
                continue

            combined = np.concatenate(fused_features, axis=1)
            features_list.append(combined)
            labels_list.append(label_ref)

        if not features_list:
            raise ValueError("No valid aligned .h5 files found across all directories.")

        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        print(f"[INFO] Loaded fused features from {len(features_list)} files")
        print(f"[INFO] Total patches: {len(all_features):,}")

        self.features = torch.tensor(all_features, dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
'''

class FusedFeatureDataset(Dataset):
    def __init__(self, dirs):
        """
        dirs: list of directories, each containing .h5 files with 'features' and 'labels' arrays.
        Keeps only patches whose labels match across all dirs (drops mismatched patches only).
        """
        features_list, labels_list = [], []
        base_dir = dirs[0]
        file_names = [f for f in os.listdir(base_dir) if f.endswith(".h5")]

        for fname in file_names:
            all_feats, all_labels = [], []
            valid = True

            # Load all dirs' features/labels
            for d in dirs:
                full_path = os.path.join(d, fname)
                if not os.path.exists(full_path):
                    print(f"⚠️ {fname} missing in {d}, skipping.")
                    valid = False
                    break
                try:
                    with h5py.File(full_path, "r") as f:
                        if "features" not in f or "labels" not in f:
                            print(f"⚠️ {fname} in {d}: missing 'features' or 'labels'.")
                            valid = False
                            break

                        feats = np.asarray(f["features"][:])
                        labels = np.asarray(f["labels"][:]).flatten()

                        # Ensure matching array lengths
                        min_len = min(len(feats), len(labels))
                        feats, labels = feats[:min_len], labels[:min_len]

                        all_feats.append(feats)
                        all_labels.append(labels)

                except Exception as e:
                    print(f"❌ Error reading {fname} in {d}: {e}")
                    valid = False
                    break

            if not valid or len(all_feats) != len(dirs):
                continue

            # 🔧 Align all label arrays to the smallest common length
            min_common_len = min(len(l) for l in all_labels)
            all_feats = [f[:min_common_len] for f in all_feats]
            all_labels = [l[:min_common_len] for l in all_labels]

            # Now stack labels safely
            label_matrix = np.stack(all_labels, axis=0)

            # ✅ Keep only patches with consistent labels
            consistent_mask = np.all(label_matrix == label_matrix[0, :], axis=0)
            num_kept = np.sum(consistent_mask)
            num_dropped = len(consistent_mask) - num_kept

            if num_dropped > 0:
                print(f"⚠️ {fname}: dropped {num_dropped}/{len(consistent_mask)} inconsistent patches")

            if num_kept == 0:
                print(f"⚠️ {fname}: all patches inconsistent, skipping.")
                continue

            # Apply mask + concatenate features along feature dimension
            fused_patches = [f[consistent_mask] for f in all_feats]
            fused_features = np.concatenate(fused_patches, axis=1)
            fused_labels = all_labels[0][consistent_mask]

            features_list.append(fused_features)
            labels_list.append(fused_labels)

        if not features_list:
            raise ValueError("No valid aligned patches found across directories.")

        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        print(f"[INFO] Loaded fused features from {len(features_list)} files")
        print(f"[INFO] Total patches after filtering: {len(all_features):,}")

        self.features = torch.tensor(all_features, dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



# ============================
# Model Definition
# ============================
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


# ============================
# Training and Evaluation
# ============================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    return y_true, y_pred, y_probs


def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print("\n[Eval Metrics]")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC Score:   {auc:.4f}")
    print(classification_report(y_true, y_pred))
    return acc, recall, specificity, f1, auc


# ============================
# Main Pipeline
# ============================
def main(dirs, epochs=20, batch_size=128, lr=1e-3, save_path="fused_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = FusedFeatureDataset(dirs)
    input_dim = dataset.features.shape[1]

    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=False)

    model = DeepMLP(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_stats = {}

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"\n[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f}")

        y_true, y_pred, y_probs = evaluate(model, val_loader, device)
        acc, recall, spec, f1, auc = compute_metrics(y_true, y_pred, y_probs)

        if acc > best_acc:
            best_acc = acc
            best_stats = {"Accuracy": acc, "Recall": recall, "Specificity": spec, "F1": f1, "AUC": auc}
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model (Acc={best_acc:.4f}) → {save_path}")

    print("\n=== FINAL BEST PERFORMANCE ===")
    for k, v in best_stats.items():
        print(f"{k}: {v:.4f}")


# ============================
# CLI Entrypoint
# ============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="List of directories containing .h5 feature files from each FM"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="fused_model.pth")
    args = parser.parse_args()

    main(args.dirs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
