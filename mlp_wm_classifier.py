import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Path to TRIDENT-extracted .h5 files
H5_DIR = "/scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_WM"

class T4vsTCDataset(Dataset):
    def __init__(self, h5_dir):
        self.features = []
        self.labels = []

        for fname in os.listdir(h5_dir):
            if not fname.endswith(".h5"):
                continue

            label = 1 if "T1" in fname else 0 if "T4" in fname else None
            if label is None:
                continue

            path = os.path.join(h5_dir, fname)
            with h5py.File(path, "r") as f:
                feats = f["features"][:]
                labs = np.full((feats.shape[0],), label)
                self.features.append(feats)
                self.labels.append(labs)

        self.features = torch.tensor(np.concatenate(self.features), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
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

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def compute_class_weights(labels_tensor):
    counts = Counter(labels_tensor.tolist())
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(2)]
    return torch.tensor(weights, dtype=torch.float32)

def main():
    dataset = T4vsTCDataset(H5_DIR)
    input_dim = dataset[0][0].shape[0]

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)

    class_weights = compute_class_weights(dataset.labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["TC", "T4"]))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

if __name__ == "__main__":
    main()

'''import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# ---- CONFIG ----
#H5_DIR = "/scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_WM"
H5_DIR = "/scratch/users/sxk2517/trident_processed/20x_256px_0px_overlap/features_musk"
BATCH_SIZE = 512
NUM_CLASSES = 6
EPOCHS = 70
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ---- CLASS LABELS ----
CLASS_LABELS = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "TC": 5
}
CLASS_NAMES = ["T0", "T1", "T2", "T3", "T4", "TC"]

# ---- DATASET ----
class MulticlassDataset(Dataset):
    def __init__(self, h5_dir):
        self.features = []
        self.labels = []

        for fname in os.listdir(h5_dir):
            if not fname.endswith(".h5"):
                continue

            label = None
            for key, value in CLASS_LABELS.items():
                if key in fname:
                    label = value
                    break
            if label is None:
                continue

            path = os.path.join(h5_dir, fname)
            with h5py.File(path, "r") as f:
                feats = f["features"][:]
                labs = np.full((feats.shape[0],), label)
                self.features.append(feats)
                self.labels.append(labs)

        self.features = torch.tensor(np.concatenate(self.features), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ---- MODEL ----
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, NUM_CLASSES)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ---- UTIL ----
def compute_class_weights(labels_tensor):
    counts = Counter(labels_tensor.tolist())
    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)  # prob of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ---- MAIN ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MulticlassDataset(H5_DIR)
    input_dim = dataset[0][0].shape[0]

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    indices = np.arange(len(dataset))
    y = dataset.labels.numpy()
    train_idx, val_idx = next(splitter.split(indices, y))
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MLP(input_dim).to(device)
    class_weights = compute_class_weights(dataset.labels).to(device)
    loss_fn = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == "__main__":
    main()
'''