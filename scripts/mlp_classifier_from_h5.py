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


class PatchFeatureDataset(Dataset):
    def __init__(self, h5_dir):
        features_list, labels_list = [], []

        for fname in os.listdir(h5_dir):
            if not fname.endswith(".h5"):
                continue
            full_path = os.path.join(h5_dir, fname)
            try:
                with h5py.File(full_path, "r") as f:
                    if "features" in f and "labels" in f:
                        features = f["features"][:]
                        labels = f["labels"][:]
                        if len(features) == len(labels):  # sanity check
                            features_list.append(features)
                            labels_list.append(labels)
                        else:
                            print(f"⚠️ Skipping {fname}: mismatched features/labels shape")
                    else:
                        print(f"⚠️ Skipping {fname}: missing 'features' or 'labels'")
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")

        if not features_list:
            raise ValueError("No valid .h5 files found in the directory.")

        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        print(f"[INFO] Total number of patches: {len(all_features):,}")

        self.features = torch.tensor(all_features, dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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
    recall = recall_score(y_true, y_pred)  # Sensitivity
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

    return acc


def main(h5_dir, epochs=20, batch_size=128, lr=1e-3, save_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = PatchFeatureDataset(h5_dir)
    input_dim = dataset.features.shape[1]

    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
        # Count TRAIN labels
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    train_tumor = (train_labels == 1).sum().item()
    train_normal = (train_labels == 0).sum().item()

    print(f"\n[TRAIN SET] Total: {len(train_labels):,}")
    print(f"  Tumor patches:     {train_tumor:,}")
    print(f"  Non-tumor patches: {train_normal:,}")

    # Count VAL labels
    val_labels = torch.tensor([val_set[i][1] for i in range(len(val_set))])
    val_tumor = (val_labels == 1).sum().item()
    val_normal = (val_labels == 0).sum().item()

    print(f"\n[VAL SET] Total: {len(val_labels):,}")
    print(f"  Tumor patches:     {val_tumor:,}")
    print(f"  Non-tumor patches: {val_normal:,}")


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

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"\n[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f}")

        y_true, y_pred, y_probs = evaluate(model, val_loader, device)
        acc = compute_metrics(y_true, y_pred, y_probs)

        # Save model if it has the best accuracy so far
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model with accuracy {best_acc:.4f} to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir", required=True, help="Directory of H5 files with 'features' and 'labels'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="File to save the best model weights")
    args = parser.parse_args()

    main(args.h5_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
