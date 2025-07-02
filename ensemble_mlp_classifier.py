#!/usr/bin/env python3
# ensemble_mlp_classifier_with_shap.py
# ------------------------------------------------------------
# Train an MLP on concatenated patch-level features from Musk,
# Hoptimus and Conchv15 .h5 files, and apply SHAP analysis.
# ------------------------------------------------------------

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
from typing import Tuple


class EnsemblePatchDataset(Dataset):
    def __init__(self, dir_musk: str, dir_hopt: str, dir_conch: str):
        filenames = set(os.listdir(dir_musk)) & set(os.listdir(dir_hopt)) & set(os.listdir(dir_conch))
        features_list, labels_list = [], []
        dims_set: Tuple[int, int, int] | None = None

        for fname in filenames:
            try:
                with h5py.File(os.path.join(dir_musk, fname), "r") as f1, \
                     h5py.File(os.path.join(dir_hopt, fname), "r") as f2, \
                     h5py.File(os.path.join(dir_conch, fname), "r") as f3:

                    if not all(set(("features", "labels", "coords")).issubset(f.keys()) for f in (f1, f2, f3)):
                        continue

                    coords1, coords2, coords3 = f1["coords"][:], f2["coords"][:], f3["coords"][:]
                    coords_set = set(map(tuple, coords1)) & set(map(tuple, coords2)) & set(map(tuple, coords3))
                    if not coords_set:
                        continue

                    sorted_coords = sorted(coords_set)

                    def index_lookup(coords, targets):
                        return [np.where((coords == c).all(axis=1))[0][0] for c in targets]

                    idx1 = index_lookup(coords1, sorted_coords)
                    idx2 = index_lookup(coords2, sorted_coords)
                    idx3 = index_lookup(coords3, sorted_coords)

                    f1_feat, f2_feat, f3_feat = f1["features"][idx1], f2["features"][idx2], f3["features"][idx3]
                    labels = f1["labels"][idx1]

                    if not (len(f1_feat) == len(f2_feat) == len(f3_feat) == len(labels)):
                        continue

                    if dims_set is None:
                        dims_set = (f1_feat.shape[1], f2_feat.shape[1], f3_feat.shape[1])

                    features_list.append(np.concatenate([f1_feat, f2_feat, f3_feat], axis=1))
                    labels_list.append(labels)

            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e!r}")

        if not features_list:
            raise ValueError("No overlapping .h5 files with matching coords found.")

        self.features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(labels_list, axis=0), dtype=torch.long)
        self.dims = dims_set

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(hidden_dim // 2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        return self.fc4(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp)

    print("  · Accuracy:    {:.4f}".format(acc))
    print("  · Sensitivity: {:.4f}".format(sen))
    print("  · Specificity: {:.4f}".format(spe))
    print("  · F1-Score:    {:.4f}".format(f1))
    print("  · AUROC:       {:.4f}".format(auc))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def run_shap_analysis(model, dataset):
    print("\n========== SHAP ANALYSIS ==========")

    # Force CPU and eval mode
    model.eval().cpu()

    dim_musk, dim_hopt, dim_conch = dataset.dims
    background = dataset.features[:500]
    data_to_explain = dataset.features[500:700]

    def model_predict(x_numpy):
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32).cpu()
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()

    explainer = shap.KernelExplainer(model_predict, background.numpy())
    shap_values = explainer.shap_values(data_to_explain.numpy(), nsamples=100)

    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    musk_mean = np.abs(shap_vals[:, :dim_musk]).mean()
    hopt_mean = np.abs(shap_vals[:, dim_musk:dim_musk+dim_hopt]).mean()
    conch_mean = np.abs(shap_vals[:, dim_musk+dim_hopt:]).mean()
    total = musk_mean + hopt_mean + conch_mean

    print(f"  · Musk Importance:     {musk_mean / total:.2%}")
    print(f"  · Hoptimus Importance: {hopt_mean / total:.2%}")
    print(f"  · Conch Importance:    {conch_mean / total:.2%}")

    shap.summary_plot(shap_vals, data_to_explain.numpy(), show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

    shap.plots.bar(shap.Explanation(
        values=shap_vals,
        base_values=np.zeros(len(shap_vals)),
        data=data_to_explain.numpy()
    ))



def main(musk_dir, hopt_dir, conch_dir, epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = EnsemblePatchDataset(musk_dir, hopt_dir, conch_dir)
    input_dim = dataset.features.shape[1]
    dim_musk, dim_hopt, dim_conch = dataset.dims
    print(f"[INFO] Loaded {len(dataset)} patches (Musk dim {dim_musk}, Hopt dim {dim_hopt}, Conch dim {dim_conch})")

    val_size = max(1, int(0.2 * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_labels = torch.stack([y for _, y in train_set])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / torch.clamp(class_counts.float(), min=1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = DeepMLP(input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_state = 0.0, None
    print("\n========== TRAINING ==========")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch:02d}/{epochs}]  Loss: {train_loss:.4f}")
        current_f1 = f1_score(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1, best_state = current_f1, model.state_dict().copy()
        compute_metrics(y_true, y_pred, y_prob)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[INFO] Loaded best model (F1 = {best_f1:.4f})")

    run_shap_analysis(model, dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--musk",  required=True, help="Directory of Musk .h5 files")
    parser.add_argument("--hopt",  required=True, help="Directory of Hoptimus .h5 files")
    parser.add_argument("--conch", required=True, help="Directory of Conchv15 .h5 files")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()

    main(args.musk, args.hopt, args.conch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
