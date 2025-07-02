#!/usr/bin/env python3
import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # make sure this is imported


# ------------------------------
# Dataset Class
# ------------------------------
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
            raise ValueError("No valid data found.")

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
        torch.save({'features': self.features, 'labels': self.labels, 'dims': self.dims}, path)

    def load_cached(self, path):
        data = torch.load(path)
        self.features = data['features']
        self.labels = data['labels']
        self.dims = data['dims']


# ------------------------------
# Model Definition
# ------------------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
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


# ------------------------------
# Training & Evaluation
# ------------------------------
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
    print(f"  · Accuracy:    {acc:.4f}")
    print(f"  · Sensitivity: {sen:.4f}")
    print(f"  · Specificity: {spe:.4f}")
    print(f"  · F1-Score:    {f1:.4f}")
    print(f"  · AUROC:       {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


# ------------------------------
# SHAP Analysis
# ------------------------------
'''def run_shap_analysis(model, dataset):
    print("\n========== SHAP ANALYSIS ==========")
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
    shap_values = explainer.shap_values(data_to_explain.numpy(), nsamples=7000)

    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    musk_mean = np.abs(shap_vals[:, :dim_musk]).mean()
    hopt_mean = np.abs(shap_vals[:, dim_musk:dim_musk+dim_hopt]).mean()
    conch_mean = np.abs(shap_vals[:, dim_musk+dim_hopt:]).mean()
    total = musk_mean + hopt_mean + conch_mean

    print(f"  · Musk Importance:     {musk_mean / total:.2%}")
    print(f"  · Hoptimus Importance: {hopt_mean / total:.2%}")
    print(f"  · Conch Importance:    {conch_mean / total:.2%}")

    shap.summary_plot(shap_vals, data_to_explain.numpy())  # violin plot
    shap.summary_plot(shap_vals, data_to_explain.numpy(), plot_type="bar")  # bar plot'''


def run_shap_analysis(model, dataset, cache_dir="shap_cache"):
    """
    Run SHAP analysis with caching of expensive preprocessing steps.
    Args:
        model: trained PyTorch model
        dataset: EnsemblePatchDataset instance with:
            - features: torch.Tensor (N, D)
            - dims: tuple (dim_musk, dim_hopt, dim_conch)
        cache_dir: directory for cached intermediates
    """
    print("\n========== SHAP ANALYSIS ==========")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Normalize and cache features
    scaler_path = os.path.join(cache_dir, "scaler.pkl")
    Xnorm_path  = os.path.join(cache_dir, "X_normalized.npy")
    if os.path.exists(scaler_path) and os.path.exists(Xnorm_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_norm = np.load(Xnorm_path)
        print("[INFO] Loaded normalized features from cache.")
    else:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(dataset.features.numpy())
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        np.save(Xnorm_path, X_norm)
        print("[INFO] Cached normalized features.")

    # 2. Cache background & explain sets
    bg_path, expl_path = os.path.join(cache_dir, "background.npy"), os.path.join(cache_dir, "to_explain.npy")
    if os.path.exists(bg_path) and os.path.exists(expl_path):
        background = np.load(bg_path)
        to_explain = np.load(expl_path)
        print("[INFO] Loaded SHAP background and explain data from cache.")
    else:
        km = shap.kmeans(X_norm, 50)
        background = km.data if hasattr(km, 'data') else km
        to_explain = X_norm[500:550]
        np.save(bg_path, background)
        np.save(expl_path, to_explain)
        print("[INFO] Cached SHAP background and explain data.")

    # 3. Compute & cache SHAP values
    model = model.eval().cpu()
    def model_predict(x_np):
        with torch.no_grad():
            logits = model(torch.tensor(x_np, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    shap_path = os.path.join(cache_dir, "shap_values.npy")
    if os.path.exists(shap_path):
        shap_vals = np.load(shap_path)
        print("[INFO] Loaded SHAP values from cache.")
    else:
        explainer = shap.KernelExplainer(model_predict, background)
        out = explainer.shap_values(to_explain, nsamples=7000)
        shap_vals = out[1] if isinstance(out, list) else out
        np.save(shap_path, shap_vals)
        print("[INFO] Cached SHAP values.")

    # 4. Block-level contributions
    dim_musk, dim_hopt, dim_conch = dataset.dims
    abs_shap = np.abs(shap_vals)
    m_mean = abs_shap[:, :dim_musk].mean()
    h_mean = abs_shap[:, dim_musk:dim_musk+dim_hopt].mean()
    c_mean = abs_shap[:, dim_musk+dim_hopt:].mean()
    total = m_mean + h_mean + c_mean
    print("\nBlock-level SHAP contributions:")
    print(f"  · MUSK:         {m_mean/total:.2%}")
    print(f"  · H-Hoptimus-1: {h_mean/total:.2%}")
    print(f"  · CONCHv1_5:    {c_mean/total:.2%}")

    # 5. Top-10 feature origins (pure Python ints)
    feat_imp = abs_shap.mean(axis=0)
    top10 = np.argsort(feat_imp)[-10:][::-1].astype(int).tolist()
    counts = {'MUSK':0, 'H-Hoptimus-1':0, 'CONCHv1_5':0}
    b0, b1 = dim_musk, dim_musk+dim_hopt
    for idx in top10:
        if idx < b0:
            counts['MUSK'] += 1
        elif idx < b1:
            counts['H-Hoptimus-1'] += 1
        else:
            counts['CONCHv1_5'] += 1

    print("\nTop-10 feature origins:")
    for name, cnt in counts.items():
        print(f"  · {name}: {cnt} features ({cnt/10*100:.1f}%)")

    # 6. SHAP summary bar plot
    shap.summary_plot(shap_vals, to_explain, plot_type="bar")



# ------------------------------
# Main Execution
# ------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.cached_data and os.path.exists(args.cached_data):
        dataset = EnsemblePatchDataset(cache_path=args.cached_data)
    else:
        dataset = EnsemblePatchDataset(args.musk, args.hopt, args.conch, cache_path=args.cached_data)

    input_dim = dataset.features.shape[1]

    model = DeepMLP(input_dim)
    if args.load_model and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        print(f"[INFO] Loaded model from {args.model_path}")
    model.to(device)

    if args.shap_only:
        run_shap_analysis(model, dataset)
        return

    val_size = max(1, int(0.2 * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_labels = torch.stack([y for _, y in train_set])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / torch.clamp(class_counts.float(), min=1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch:02d}]  Train Loss: {train_loss:.4f}")
        current_f1 = f1_score(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1, best_state = current_f1, model.state_dict().copy()
        compute_metrics(y_true, y_pred, y_prob)

    if best_state:
        model.load_state_dict(best_state)
        print(f"[INFO] Best model loaded (F1 = {best_f1:.4f})")
        if args.save_model:
            torch.save(model.state_dict(), args.model_path)
            print(f"[INFO] Saved model to {args.model_path}")

    run_shap_analysis(model, dataset)


# ------------------------------
# Argument Parser
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musk", help="Directory of Musk .h5 files")
    parser.add_argument("--hopt", help="Directory of Hoptimus .h5 files")
    parser.add_argument("--conch", help="Directory of Conchv15 .h5 files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cached_data", default="cached_dataset.pt")
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--shap_only", action="store_true")
    args = parser.parse_args()
    main(args)
