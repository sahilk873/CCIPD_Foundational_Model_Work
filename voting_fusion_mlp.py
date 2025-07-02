import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


class PatchFeatureDataset(Dataset):
    def __init__(self, h5_dir):
        features_list, labels_list = [], []

        for fname in os.listdir(h5_dir):
            if fname.endswith(".h5"):
                try:
                    with h5py.File(os.path.join(h5_dir, fname), "r") as f:
                        if "features" in f and "labels" in f:
                            feats, labs = f["features"][:], f["labels"][:]
                            if len(feats) == len(labs):
                                features_list.append(feats)
                                labels_list.append(labs)
                except:
                    continue

        self.features = torch.tensor(np.concatenate(features_list), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(labels_list), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, device, epochs=20, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return model


def predict_probs(model, val_loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            logits = model(X)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
    return np.array(probs)


def evaluate_predictions(y_true, y_probs):
    y_pred = (y_probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    print("\n[Ensemble Eval]")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC Score:   {auc:.4f}")
    print(classification_report(y_true, y_pred))


def main(musk_dir, hopt_dir, conch_dir, epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    datasets = [PatchFeatureDataset(d) for d in [musk_dir, hopt_dir, conch_dir]]
    n_total = len(datasets[0])
    val_size = int(0.2 * n_total)
    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    models, val_loaders, all_probs = [], [], []
    for ds in datasets:
        train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size)

        model = MLP(input_dim=ds.features.shape[1]).to(device)
        model = train_model(model, train_loader, device, epochs, lr)
        probs = predict_probs(model, val_loader, device)

        models.append(model)
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    y_true = datasets[0].labels[val_indices].numpy()
    evaluate_predictions(y_true, avg_probs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--musk", required=True)
    parser.add_argument("--hopt", required=True)
    parser.add_argument("--conch", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args.musk, args.hopt, args.conch, args.epochs, args.batch_size, args.lr)
