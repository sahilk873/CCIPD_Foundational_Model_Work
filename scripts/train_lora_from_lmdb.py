import os, math, random, pickle, lmdb
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.checkpoint import checkpoint_sequential
from torch.amp import autocast, GradScaler
from collections import Counter


from peft import LoraConfig, get_peft_model
from lora_conch_loader import load_conch_v15  # your working CONCH v1.5 loader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ---------------- Utilities ----------------

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- From your MLP script ---
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def resolve_base(model):
    """Return the underlying base model if this is a PEFT-wrapped module; else the model itself."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def extract_embeddings(encoder, dataloader, device, amp_dtype, ckpt_tail_start, ckpt_chunks):
    """Run encoder on a loader and collect all embeddings + labels as tensors."""
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device=="cuda")):
                base = resolve_base(encoder)
                z = encode_with_checkpoint(base, xb,
                                           tail_start=ckpt_tail_start,
                                           chunks=ckpt_chunks,
                                           use_ckpt=False)  # no need to checkpoint for forward-only
            feats.append(z.cpu())
            labels.append(yb.cpu())
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    return feats, labels


def train_mlp_once(encoder, train_loader, val_loader, device, amp_dtype, cfg, class_weights):
    """Extract embeddings, then train an MLP classifier for 30 epochs."""
    # 1. Extract embeddings from LoRA encoder
    tr_feats, tr_labels = extract_embeddings(encoder, train_loader, device, amp_dtype,
                                             cfg.ckpt_tail_start, cfg.ckpt_chunks)
    va_feats, va_labels = extract_embeddings(encoder, val_loader, device, amp_dtype,
                                             cfg.ckpt_tail_start, cfg.ckpt_chunks)

    # 2. Wrap into loaders
    train_emb_loader = DataLoader(TensorDataset(tr_feats, tr_labels),
                                  batch_size=cfg.batch_size, shuffle=True)
    val_emb_loader   = DataLoader(TensorDataset(va_feats, va_labels),
                                  batch_size=cfg.batch_size)

    # 3. Define MLP
    input_dim = tr_feats.shape[1]
    mlp = DeepMLP(input_dim=input_dim, num_classes=cfg.num_classes).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 4. Train for 30 epochs
    for e in range(30):
        mlp.train()
        tot_loss = 0
        for xb, yb in train_emb_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = mlp(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"[MLP epoch {e+1}/5] Loss: {tot_loss/len(train_emb_loader):.4f}")

        # quick val acc
        mlp.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_emb_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = mlp(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        print(f"Val Acc: {correct/total:.4f}")

    return mlp



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


# ---------------- LMDB Dataset ----------------
class PatchLMDB(Dataset):
    """
    Streams patches from a single LMDB.
    - Opens the LMDB environment lazily per worker for best IO performance.
    - Expects keys: "00000000", "00000001", ...
    - Values: pickle.dumps({"image_bytes","image_shape","image_dtype","label","slide_id"(opt)})
    """
    def __init__(self, lmdb_path: str, transform, limit_items: Optional[int] = None):
        super().__init__()
        if not os.path.isdir(lmdb_path):
            raise ValueError(f"LMDB directory not found: {lmdb_path}")
        self.lmdb_path = lmdb_path
        self.transform = transform

        # Determine dataset length once (open/close env in main proc)
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, max_readers=4096)
        with env.begin(write=False) as txn:
            stat = env.stat()
            total = stat.get("entries", 0)
        env.close()

        if total <= 0:
            raise ValueError(f"LMDB at {lmdb_path} appears empty (entries={total}).")

        if limit_items is not None and limit_items > 0:
            self.N = min(limit_items, total)
            if self.N < total:
                print(f"[INFO] Using only {self.N} items (of {total}) from LMDB for this run.")
        else:
            self.N = total

        # Worker-local handles (created lazily)
        self._env = None
        self._txn = None

    def _ensure_open(self):
        # Lazily open per worker process
        if self._env is None or self._txn is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, max_readers=4096)
            self._txn = self._env.begin(write=False)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        self._ensure_open()
        key = f"{idx:08d}".encode("ascii")
        raw = self._txn.get(key)
        if raw is None:
            alt_key = f"{idx % self.N:08d}".encode("ascii")  # defensive
            raw = self._txn.get(alt_key)
            if raw is None:
                raise KeyError(f"Key not found in LMDB: {key!r}")

        val = pickle.loads(raw)
        img_shape = tuple(val["image_shape"])
        img_dtype = np.dtype(val["image_dtype"])
        img = np.frombuffer(val["image_bytes"], dtype=img_dtype).reshape(img_shape)
        y = int(val["label"])

        # To PIL → transform (fix Pillow 'mode' deprecation and ensure RGB)
        from PIL import Image
        pil = Image.fromarray(img)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        x = self.transform(pil)
        return x, torch.tensor(y, dtype=torch.long)


# ---------------- Config ----------------
@dataclass
class TrainCfg:
    lmdb_path: str
    out_dir: str = "out_conch_lora"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    num_workers: int = 8
    val_frac: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    num_classes: int = 2
    limit_items: Optional[int] = None  # for quick debug runs

    # Activation checkpointing controls
    ckpt_tail_start: int = 18  # start checkpointing from this transformer block index
    ckpt_chunks: int = 6       # number of chunks passed to checkpoint_sequential


# ---------------- Helpers: checkpointed encoder forward ----------------
def _as_model_root(obj):
    # Handle PEFT wrappers or containers that stash the real module under .model or .module
    m = getattr(obj, "model", None) or getattr(obj, "module", None) or obj
    return m

def _list_blocks(module):
    b = getattr(module, "blocks", None)
    if b is None:
        return None
    # nn.Sequential or similar
    try:
        return list(b) if hasattr(b, "__iter__") else list(b.children())
    except Exception:
        return list(b.children())

def _find_trunk_blocks(base) -> Optional[Tuple[torch.nn.Module, list]]:
    """
    Try common CONCH/VisionTransformer layouts, then fall back to a recursive search
    for a submodule that has both .patch_embed and .blocks.
    """
    m = _as_model_root(base)

    # Fast path: expected CONCH path
    trunk = getattr(m, "trunk", None)
    if trunk is not None:
        blocks = _list_blocks(trunk)
        if blocks:
            return trunk, blocks

    # Some builds put the visual tower under .visual or similar
    for attr in ("visual", "backbone", "encoder"):
        sub = getattr(m, attr, None)
        if sub is not None:
            b = getattr(sub, "trunk", None)
            if b is not None:
                blocks = _list_blocks(b)
                if blocks:
                    return b, blocks
            blocks = _list_blocks(sub)
            if blocks and hasattr(sub, "patch_embed"):
                return sub, blocks

    # Last resort: recursive search
    for _, mod in m.named_modules():
        if hasattr(mod, "patch_embed") and hasattr(mod, "blocks"):
            blocks = _list_blocks(mod)
            if blocks:
                return mod, blocks

    return None

def encode_with_checkpoint(base, x: torch.Tensor,
                           tail_start: int = 18, chunks: int = 6,
                           use_ckpt: bool = True) -> torch.Tensor:
    """
    Re-implements forward_features + contrast head with tail checkpointing.
    """
    m = _as_model_root(base)
    found = _find_trunk_blocks(m)

    # If we still can't find the expected structure, do the plain forward on the resolved root.
    if (not use_ckpt) or (found is None):
        return m(x)

    trunk, blocks = found
    num_blocks = len(blocks)
    tail_start = max(0, min(tail_start, num_blocks))

    # --- Mirror VisionTransformer.forward_features ---
    x = trunk.patch_embed(x)
    if hasattr(trunk, "pos_drop"):   x = trunk.pos_drop(x)
    if hasattr(trunk, "patch_drop"): x = trunk.patch_drop(x)
    if hasattr(trunk, "norm_pre"):   x = trunk.norm_pre(x)

    # Head blocks (no checkpoint)
    for i in range(tail_start):
        x = blocks[i](x)

    # Tail with checkpoint
    if tail_start < num_blocks:
        tail = torch.nn.Sequential(*blocks[tail_start:])
        x = checkpoint_sequential(tail, chunks, x)

    x = trunk.norm(x)
    if hasattr(trunk, "fc_norm") and not isinstance(trunk.fc_norm, torch.nn.Identity):
        x = trunk.fc_norm(x)
    if hasattr(trunk, "head_drop") and not isinstance(trunk.head_drop, torch.nn.Identity):
        x = trunk.head_drop(x)

    # Contrastive head on the top-level model (CONCHVisionTower-like)
    model = _as_model_root(base)
    x = model.attn_pool_contrast(x)
    x = model.ln_contrast(x)
    if x.dim() == 3:
        x = x.mean(dim=1)
    return x

# ---------------- Loaders ----------------
def make_loaders(ds: Dataset, cfg: TrainCfg):
    N = len(ds)
    idxs = list(range(N))
    random.shuffle(idxs)
    v = int(math.ceil(cfg.val_frac * N))
    val_idx = idxs[:v]
    trn_idx = idxs[v:] if v < N else idxs

    train_loader = DataLoader(
        Subset(ds, trn_idx),
        batch_size=cfg.batch_size,
        sampler=RandomSampler(Subset(ds, trn_idx)),
        num_workers=cfg.num_workers,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=cfg.batch_size,
        sampler=SequentialSampler(Subset(ds, val_idx)),
        num_workers=cfg.num_workers,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader
'''

# ---------------- Main ----------------
def main(
    lmdb_path: str,
    out_dir: str = "out_conch_lora",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 8,
    val_frac: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_classes: int = 2,
    seed: int = 42,
    limit_items: Optional[int] = None,
    ckpt_tail_start: int = 18,
    ckpt_chunks: int = 6,
):
    cfg = TrainCfg(
        lmdb_path=lmdb_path, out_dir=out_dir, epochs=epochs, batch_size=batch_size, lr=lr,
        num_workers=num_workers, val_frac=val_frac, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, num_classes=num_classes, seed=seed, limit_items=limit_items,
        ckpt_tail_start=ckpt_tail_start, ckpt_chunks=ckpt_chunks,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # Device / AMP policy
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[INFO] CUDA available: {use_cuda}")
    if use_cuda:
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load encoder + transform
    enc, tfm, _ = load_conch_v15(device=device)
    enc = enc.to(device)
    enc.eval()

    # Dataset & loaders (LMDB)
    ds = PatchLMDB(cfg.lmdb_path, transform=tfm, limit_items=cfg.limit_items)
    train_loader, val_loader = make_loaders(ds, cfg)
    all_labels = []
    for i in range(len(ds)):
        _, y = ds[i]
        all_labels.append(int(y))

    counts = Counter(all_labels)
    num_classes = cfg.num_classes
    class_counts = torch.tensor([counts.get(c, 0) for c in range(num_classes)], dtype=torch.float)

    # Inverse frequency weighting
    class_weights = class_counts.sum() / (num_classes * class_counts)
    print("[INFO] Class counts:", class_counts.tolist())
    print("[INFO] Class weights:", class_weights.tolist())



    # Inject LoRA adapters (skip Conv2d 'proj' to avoid fp16 bias issues)
    target_modules = ["qkv", "fc1", "fc2", "to_q", "to_kv", "to_out"]
    lcfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target_modules, bias="none", task_type="FEATURE_EXTRACTION"
    )
    for p in enc.parameters():
        p.requires_grad_(False)
    enc = get_peft_model(enc, lcfg).to(device)

    # Choose AMP dtype: prefer bf16 on L40S/Hopper; else fp16
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    # Align encoder & LoRA params to AMP dtype to avoid hidden upcasts
    enc = enc.to(dtype=amp_dtype)
    for n, p in enc.named_parameters():
        if "lora_" in n and p.dtype != amp_dtype:
            p.data = p.data.to(amp_dtype)

    enc.train()

    # Probe embedding dim (no autocast for shape probe)
    # --- Probe embedding dim (match model dtype!) ---
    with torch.no_grad():
        xb0, _ = next(iter(train_loader))
        xb0 = xb0.to(device)

        # enc was cast to amp_dtype earlier (bf16 on L40S, else fp16)
        if use_cuda:
            xb0 = xb0.to(dtype=amp_dtype)
            # keep compute consistent with the rest of training
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                z0 = encode_with_checkpoint(
                    enc.get_base_model(), xb0,
                    tail_start=cfg.ckpt_tail_start,
                    chunks=cfg.ckpt_chunks,
                    use_ckpt=False,   # no need to checkpoint for a single forward
                )
        else:
            xb0 = xb0.float()
            z0 = encode_with_checkpoint(
                enc.get_base_model(), xb0,
                tail_start=cfg.ckpt_tail_start,
                chunks=cfg.ckpt_chunks,
                use_ckpt=False,
            )

    emb_dim = z0.shape[-1]
    print(f"[INFO] Embedding dim = {emb_dim}")


    # FP32 classifier head
    hidden_dim = emb_dim // 2   # you can tune this (e.g., 512)
    head = DeepMLP(emb_dim, hidden_dim=hidden_dim, num_classes=cfg.num_classes).to(device)


    opt = torch.optim.AdamW(
        [p for p in enc.parameters() if p.requires_grad] + list(head.parameters()),
        lr=cfg.lr,
    )
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

    scaler = GradScaler(enabled=(use_cuda and amp_dtype == torch.float16))

    # Training loop
    for e in range(1, cfg.epochs + 1):
        enc.train(); head.train()
        tr_loss, tr_acc, tr_n = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                base = enc.get_base_model()
                z = encode_with_checkpoint(base, xb,
                                           tail_start=cfg.ckpt_tail_start,
                                           chunks=cfg.ckpt_chunks,
                                           use_ckpt=True)
                logits = head(z.float())  # head stays fp32
                loss = crit(logits, yb)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            tr_loss += loss.item() * xb.size(0)
            tr_acc  += (logits.argmax(1) == yb).float().sum().item()
            tr_n    += xb.size(0)

        # Validation
        enc.eval(); head.eval()
        va_loss, va_acc, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device)
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                    base = enc.get_base_model()
                    z = encode_with_checkpoint(base, xb,
                                               tail_start=cfg.ckpt_tail_start,
                                               chunks=cfg.ckpt_chunks,
                                               use_ckpt=True)
                    logits = head(z.float())
                    loss = crit(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_acc  += (logits.argmax(1) == yb).float().sum().item()
                va_n    += xb.size(0)

        print(f"[{e:02d}/{cfg.epochs}] "
              f"train Loss {tr_loss/max(1,tr_n):.4f} Acc {tr_acc/max(1,tr_n):.4f} | "
              f"val Loss {va_loss/max(1,va_n):.4f} Acc {va_acc/max(1,va_n):.4f}")
        # Every 5 epochs: train & validate MLP on embeddings
        if e % 30 == 0 and e != 0:
            print(f"\n[INFO] Training MLP on LoRA embeddings after epoch {e}")
            mlp = train_mlp_once(enc, train_loader, val_loader, device, amp_dtype, cfg, class_weights)
            torch.save(mlp.state_dict(), os.path.join(cfg.out_dir, f"mlp_after_{e}.pt"))

            # ---- MLP validation on embeddings ----
            mlp.eval()
            y_true, y_pred, y_probs = [], [], []
            with torch.no_grad():
                va_feats, va_labels = extract_embeddings(enc, val_loader, device, amp_dtype,
                                                        cfg.ckpt_tail_start, cfg.ckpt_chunks)
                val_emb_loader = DataLoader(TensorDataset(va_feats, va_labels),
                                            batch_size=cfg.batch_size)

                for xb, yb in val_emb_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = mlp(xb)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = logits.argmax(1).cpu().numpy()
                    y_probs.extend(probs)
                    y_pred.extend(preds)
                    y_true.extend(yb.cpu().numpy())

            # ---- Compute metrics ----
            acc = accuracy_score(y_true, y_pred)
            sensitivity = recall_score(y_true, y_pred)  # recall = sensitivity
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_probs)

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)

            print(f"\n[MLP after {e} epochs — Validation Metrics]")
            print(f"Accuracy:    {acc:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"F1 Score:    {f1:.4f}")
            print(f"AUC Score:   {auc:.4f}")
            print(classification_report(y_true, y_pred))


    # Save adapters + head + meta
    enc.save_pretrained(cfg.out_dir)
    torch.save(head.state_dict(), os.path.join(cfg.out_dir, "head.pt"))
    torch.save({"emb_dim": emb_dim, "num_classes": cfg.num_classes,
                "ckpt_tail_start": cfg.ckpt_tail_start, "ckpt_chunks": cfg.ckpt_chunks,
                "amp_dtype": str(amp_dtype)},
               os.path.join(cfg.out_dir, "meta.pt"))
    print(f"[DONE] Saved LoRA adapters and head → {cfg.out_dir}")

'''


# ---------------- Stage 1: Train MLP on frozen embeddings ----------------
# ---------------- Stage 1: Train MLP on frozen embeddings ----------------
def main(
    lmdb_path: str,
    out_dir: str = "out_conch_lora",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_workers: int = 16,
    val_frac: float = 0.2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_classes: int = 2,
    seed: int = 42,
    limit_items: Optional[int] = None,
    ckpt_tail_start: int = 12,
    ckpt_chunks: int = 6,
):
    # ---- basic cfg / device ----
    cfg = TrainCfg(
        lmdb_path=lmdb_path, out_dir=out_dir, epochs=epochs, batch_size=batch_size, lr=lr,
        num_workers=num_workers, val_frac=val_frac, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, num_classes=num_classes, seed=seed, limit_items=limit_items,
        ckpt_tail_start=ckpt_tail_start, ckpt_chunks=ckpt_chunks,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[INFO] CUDA available: {use_cuda}")
    if use_cuda:
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    # ---- load encoder & data ----
    enc, tfm, _ = load_conch_v15(device=device)
    enc = enc.to(device).eval()

    ds = PatchLMDB(cfg.lmdb_path, transform=tfm, limit_items=cfg.limit_items)
    train_loader, val_loader = make_loaders(ds, cfg)

    # ---- Fixed class weights ----
    class_weights = torch.tensor([0.7093909382820129, 1.6939390897750854], dtype=torch.float).to(device)

    # ---- Stage 1: train MLP on frozen embeddings ----
    print("\n[Stage 1] Training MLP on frozen encoder embeddings...")
    for p in enc.parameters():
        p.requires_grad_(False)

    tr_feats, tr_labels = extract_embeddings(enc, train_loader, device, amp_dtype,
                                             cfg.ckpt_tail_start, cfg.ckpt_chunks)
    va_feats, va_labels = extract_embeddings(enc, val_loader, device, amp_dtype,
                                             cfg.ckpt_tail_start, cfg.ckpt_chunks)
    emb_dim = tr_feats.shape[1]

    mlp = DeepMLP(input_dim=emb_dim, hidden_dim=1024, num_classes=cfg.num_classes).to(device)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    crit_mlp = nn.CrossEntropyLoss(weight=class_weights)

    train_emb_loader = DataLoader(TensorDataset(tr_feats, tr_labels),
                                  batch_size=cfg.batch_size, shuffle=True)
    val_emb_loader   = DataLoader(TensorDataset(va_feats, va_labels),
                                  batch_size=cfg.batch_size)

    for e in range(30):  # small, fast convergence
        mlp.train(); running = 0.0
        for xb, yb in train_emb_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_mlp.zero_grad()
            logits = mlp(xb)
            loss = crit_mlp(logits, yb)
            loss.backward()
            opt_mlp.step()
            running += loss.item()

        # validation stats
        mlp.eval(); y_true, y_prob, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_emb_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = mlp(xb)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(1)
                y_true.extend(yb.cpu().numpy()); y_prob.extend(probs.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp)
        print(f"\n[Stage1][Epoch {e+1}] Loss {running/len(train_emb_loader):.4f}")
        print(f"  Accuracy: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    torch.save(mlp.state_dict(), os.path.join(cfg.out_dir, "mlp_stage1.pt"))
    print("[Stage 1] Saved trained MLP head.")

    # ---- Stage 2: inject LoRA, freeze MLP, train LoRA-only ----
    print("\n[Stage 2] Fine-tuning LoRA adapters with MLP fixed...")
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    target_modules = ["qkv", "fc1", "fc2", "to_q", "to_kv", "to_out"]
    lcfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target_modules, bias="none", task_type="FEATURE_EXTRACTION"
    )

    # --- Load a clean encoder for comparison ---
    enc_clean, _, _ = load_conch_v15(device=device)
    enc_clean.eval()

    # --- LoRA-wrapped encoder ---
    enc_lora = get_peft_model(load_conch_v15(device=device)[0], lcfg).to(device).eval()

    # --- Test embeddings on one batch ---
    xb, _ = next(iter(train_loader))
    xb = xb.to(device)

    with torch.no_grad():
        z_clean = enc_clean(xb)
        z_lora  = enc_lora(xb)
        diff = (z_clean - z_lora).abs().max().item()

    print(f"[DEBUG] Max abs diff between clean vs LoRA-wrapped encoder embeddings (before training): {diff:.6e}")

    # --- Now proceed with actual enc = get_peft_model(enc, lcfg) and zero-init ---
    enc = get_peft_model(enc, lcfg).to(device)
    enc = enc.to(dtype=amp_dtype)

    # Force zero init
    for name, module in enc.named_modules():
        if hasattr(module, "lora_A"):
            torch.nn.init.zeros_(module.lora_A.weight)
        if hasattr(module, "lora_B"):
            torch.nn.init.zeros_(module.lora_B.weight)
    enc = get_peft_model(enc, lcfg).to(device)
    enc = enc.to(dtype=amp_dtype)

    # unfreeze only LoRA params
    for n, p in enc.named_parameters():
        p.requires_grad_(("lora_" in n))

    trainable = [p for p in enc.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable LoRA parameters found. Check target_modules.")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(use_cuda and amp_dtype == torch.float16))

    # --- weighted loss for Stage 2 ---
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_auc = -1.0
    for e in range(1, cfg.epochs + 1):
        enc.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                z = encode_with_checkpoint(
                                            resolve_base(enc), xb,
                                            tail_start=cfg.ckpt_tail_start,
                                            chunks=cfg.ckpt_chunks,
                                            use_ckpt=True
                                        )
                logits = mlp(z.float())  # head stays fp32
                loss = crit(logits, yb)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tr_loss += loss.item()

        # validation
        enc.eval(); y_true, y_prob, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                    z = encode_with_checkpoint(resolve_base(enc), xb,
                                               tail_start=cfg.ckpt_tail_start,
                                               chunks=cfg.ckpt_chunks,
                                               use_ckpt=True)
                    logits = mlp(z.float())
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = logits.argmax(1)
                y_true.extend(yb.cpu().numpy()); y_prob.extend(probs.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp)
        print(f"[Stage2][Epoch {e}] train_loss {tr_loss/len(train_loader):.4f} | "
              f"Val Acc {acc:.4f} Sens {sens:.4f} Spec {spec:.4f} F1 {f1:.4f} AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({"enc": enc.state_dict(), "mlp": mlp.state_dict(), "auc": auc, "epoch": e},
                       os.path.join(cfg.out_dir, "best_stage2.pt"))

    enc.save_pretrained(cfg.out_dir)
    torch.save({"emb_dim": emb_dim, "num_classes": cfg.num_classes,
                "ckpt_tail_start": cfg.ckpt_tail_start, "ckpt_chunks": cfg.ckpt_chunks,
                "amp_dtype": str(amp_dtype)}, os.path.join(cfg.out_dir, "meta.pt"))
    print(f"[DONE] Saved LoRA adapters and head → {cfg.out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train LoRA on CONCH v1.5 using an LMDB of patches with AMP + activation checkpointing.")
    ap.add_argument("--lmdb_path", required=True, help="Path to LMDB directory (from converter)")
    ap.add_argument("--out_dir", default="out_conch_lora")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_items", type=int, default=None, help="Use only first N items from LMDB (for quick tests)")
    ap.add_argument("--ckpt_tail_start", type=int, default=12, help="Start checkpointing from this ViT block index (tail)")
    ap.add_argument("--ckpt_chunks", type=int, default=6, help="Number of chunks for checkpoint_sequential")
    args = ap.parse_args()

    main(
        lmdb_path=args.lmdb_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_classes=args.num_classes,
        seed=args.seed,
        limit_items=args.limit_items,
        ckpt_tail_start=args.ckpt_tail_start,
        ckpt_chunks=args.ckpt_chunks,
    )
