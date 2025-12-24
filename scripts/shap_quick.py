#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import h5py
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# Dataset
# ============================================================
class EnsemblePatchDataset(Dataset):
    def __init__(self, dir_musk=None, dir_hopt=None, dir_conch=None, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            self.load_cached(cache_path)
            print(f"[INFO] Loaded dataset from cache: {cache_path}")
            return

        filenames = set(os.listdir(dir_musk)) & set(os.listdir(dir_hopt)) & set(os.listdir(dir_conch))
        features_list, labels_list = [], []
        dims_set = None

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

                    f1_feat = f1["features"][idx1]
                    f2_feat = f2["features"][idx2]
                    f3_feat = f3["features"][idx3]
                    labels = f1["labels"][idx1]

                    if dims_set is None:
                        dims_set = (f1_feat.shape[1], f2_feat.shape[1], f3_feat.shape[1])

                    features_list.append(np.concatenate([f1_feat, f2_feat, f3_feat], axis=1))
                    labels_list.append(labels)

            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e!r}")

        if not features_list:
            raise RuntimeError("No valid data found.")

        self.features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(labels_list, axis=0), dtype=torch.long)
        self.dims = dims_set

        if cache_path:
            self.save_cached(cache_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def save_cached(self, path):
        torch.save(
            {"features": self.features, "labels": self.labels, "dims": self.dims},
            path
        )

    def load_cached(self, path):
        data = torch.load(path, map_location="cpu")
        self.features = data["features"]
        self.labels = data["labels"]
        self.dims = data["dims"]


# ============================================================
# Model
# ============================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
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
        return self.net(x)


# ============================================================
# Training / Evaluation
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp)

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sen:.4f}")
    print(f"  Specificity: {spe:.4f}")
    print(f"  F1-score:    {f1:.4f}")
    print(f"  AUROC:       {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


# ============================================================
# SHAP Analysis (with saved plots)
# ============================================================
def run_shap_analysis(model, dataset, cache_dir="shap_cache"):
    print("\n========== SHAP ANALYSIS ==========")
    os.makedirs(cache_dir, exist_ok=True)

    # ---------- Normalize ----------
    scaler_path = os.path.join(cache_dir, "scaler.pkl")
    Xnorm_path = os.path.join(cache_dir, "X_normalized.npy")

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_norm = np.load(Xnorm_path)
    else:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(dataset.features.numpy())
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        np.save(Xnorm_path, X_norm)

    # ---------- Background ----------
    bg_path = os.path.join(cache_dir, "background.npy")
    expl_path = os.path.join(cache_dir, "to_explain.npy")

    if os.path.exists(bg_path):
        background = np.load(bg_path)
        to_explain = np.load(expl_path)
    else:
        km = shap.kmeans(X_norm, 50)
        background = km.data
        to_explain = X_norm[500:550]
        np.save(bg_path, background)
        np.save(expl_path, to_explain)

    # ---------- SHAP ----------
    model = model.eval().cpu()

    def model_predict(x):
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    shap_path = os.path.join(cache_dir, "shap_values.npy")
    if os.path.exists(shap_path):
        shap_vals = np.load(shap_path)
    else:
        explainer = shap.KernelExplainer(model_predict, background)
        out = explainer.shap_values(to_explain, nsamples=7000)
        shap_vals = out[1] if isinstance(out, list) else out
        np.save(shap_path, shap_vals)

    abs_shap = np.abs(shap_vals)

    # ---------- SHAP summary plots ----------
    plt.figure()
    shap.summary_plot(shap_vals, to_explain, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(cache_dir, "shap_summary_violin.png"), dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_vals, to_explain, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(cache_dir, "shap_summary_bar.png"), dpi=300)
    plt.close()

    # ---------- Block-level ----------
    dim_musk, dim_hopt, dim_conch = dataset.dims
    m = abs_shap[:, :dim_musk].mean()
    h = abs_shap[:, dim_musk:dim_musk + dim_hopt].mean()
    c = abs_shap[:, dim_musk + dim_hopt:].mean()
    total = m + h + c

    print("\nBlock-level SHAP contribution:")
    print(f"  MUSK:         {m / total:.2%}")
    print(f"  H-Hoptimus-1: {h / total:.2%}")
    print(f"  CONCHv1_5:    {c / total:.2%}")

    # ---------- Top-20 individual features ----------
    # ---------- Top-20 individual features (robust) ----------
    topk = 20

    # Ensure abs_shap is feature-level: (N, D)
    # If it is interaction-shaped (N, D, D), collapse to main effects
    if isinstance(abs_shap, np.ndarray) and abs_shap.ndim == 3:
        abs_shap_main = abs_shap.mean(axis=2)   # (N, D)
    else:
        abs_shap_main = abs_shap                # (N, D)

    # Compute global importance per feature: (D,)
    feat_imp = np.asarray(abs_shap_main).mean(axis=0).reshape(-1)

    # Guard against pathological cases
    D = feat_imp.shape[0]
    topk_eff = min(topk, D)

    # Indices of top-k most important features
    idx = np.argsort(feat_imp)[-topk_eff:][::-1]
    idx = [int(i) for i in idx]  # force Python ints (prevents ambiguous comparisons)

    vals = feat_imp[idx]

    # Identify which foundation model each feature belongs to
    b0 = int(dim_musk)
    b1 = int(dim_musk + dim_hopt)

    sources = []
    labels = []
    for i in idx:
        if i < b0:
            sources.append("MUSK")
            labels.append(f"MUSK[{i}]")
        elif i < b1:
            sources.append("H-Hoptimus-1")
            labels.append(f"HOPT[{i - b0}]")
        else:
            sources.append("CONCHv1_5")
            labels.append(f"CONCH[{i - b1}]")

    # Colors per FM
    colors = {
        "MUSK": "#1f77b4",
        "H-Hoptimus-1": "#ff7f0e",
        "CONCHv1_5": "#2ca02c",
    }

    # Plot
    plt.figure(figsize=(12, 7))
    y = np.arange(topk_eff)

    plt.barh(y, vals, color=[colors[s] for s in sources])
    plt.yticks(y, labels)
    plt.xlabel("Mean |SHAP value| (global importance)")
    plt.title(f"Top {topk_eff} Individual Feature Importances (color = foundation model)")
    plt.gca().invert_yaxis()

    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(cache_dir, f"top{topk_eff}_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()




# ============================================================
# Main
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = EnsemblePatchDataset(
        args.musk, args.hopt, args.conch, cache_path=args.cached_data
    )

    model = DeepMLP(dataset.features.shape[1]).to(device)

    if args.load_model and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        print("[INFO] Loaded model weights")

    if args.shap_only:
        run_shap_analysis(model, dataset)
        return

    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_labels = torch.tensor([y for _, y in train_set])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / torch.clamp(class_counts.float(), min=1)
    sampler = WeightedRandomSampler(class_weights[train_labels], len(train_labels), replacement=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state = 0.0, None

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)
        f1 = f1_score(y_true, y_pred)

        print(f"\nEpoch {epoch+1:02d} | Train loss: {loss:.4f}")
        compute_metrics(y_true, y_pred, y_prob)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    if args.save_model:
        torch.save(model.state_dict(), args.model_path)

    run_shap_analysis(model, dataset)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musk", required=True)
    parser.add_argument("--hopt", required=True)
    parser.add_argument("--conch", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cached_data", default="cached_dataset.pt")
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--shap_only", action="store_true")

    main(parser.parse_args())





