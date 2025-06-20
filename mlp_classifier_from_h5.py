'''import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score


class PatchFeatureDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.features = f["features"][:]
            self.labels = f["labels"][:]

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary classification (output logits for 2 classes)
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
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    return accuracy_score(y_true, y_pred), classification_report(y_true, y_pred)


def main(h5_path, epochs=10, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = PatchFeatureDataset(h5_path)
    input_dim = dataset.features.shape[1]
    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = MLPClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, report = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")
        print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MLP on h5-extracted patch features.")
    parser.add_argument("--h5", type=str, required=True, help="Path to .h5 file containing 'features' and 'labels'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    main(args.h5, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
'''
'''
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, accuracy_score


class PatchFeatureDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.features = f["features"][:]
            self.labels = f["labels"][:]
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # binary classification
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
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    return accuracy_score(y_true, y_pred), classification_report(y_true, y_pred)


def main(h5_path, epochs=10, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = PatchFeatureDataset(h5_path)
    input_dim = dataset.features.shape[1]

    # Split dataset
    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    # Get labels from train set only
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    class_counts = torch.bincount(train_labels)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = MLPClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, report = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")
        print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MLP on h5-extracted patch features with tumor upsampling.")
    parser.add_argument("--h5", type=str, required=True, help="Path to .h5 file with features and labels.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args.h5, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
'''

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
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.features = f["features"][:]
            self.labels = f["labels"][:]
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

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


def main(h5_path, epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = PatchFeatureDataset(h5_path)
    input_dim = dataset.features.shape[1]

    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    # Compute class weights for sampler and loss
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = DeepMLP(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"\n[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f}")

        y_true, y_pred, y_probs = evaluate(model, val_loader, device)
        compute_metrics(y_true, y_pred, y_probs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to h5 file with 'features' and 'labels'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args.h5, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
