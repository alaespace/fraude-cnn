"""
train.py  — v2 (fix biais "tout suspect")
==========================================
Corrections appliquées :
  1. Ratio augmentation rééquilibré : normal x2, anomalie x2 (50/50)
  2. Poids de classe égaux (pas de surpondération anomalie)
  3. LR plus bas : 5e-5
  4. Patience augmentée : 6
  5. Gèle seulement layer1/layer2, libère layer3/layer4/fc
  6. Seuil API abaissé à 0.70 (dans main.py)
"""

import csv, json, time, argparse, random
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("❌ pip install torch torchvision Pillow numpy tqdm")
    exit(1)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
LOGS_DIR      = Path("logs")
METADATA      = PROCESSED_DIR / "metadata_processed.csv"
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 2
RANDOM_SEED   = 42
DEVICE        = torch.device("cpu")

DEFAULT_EPOCHS   = 25
DEFAULT_BATCH    = 16
DEFAULT_LR       = 5e-5      # ← plus bas pour éviter l'overfitting
DEFAULT_PATIENCE = 6         # ← plus de patience

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"🖥️  Device : {DEVICE}")


class InvoiceDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        row   = self.records[idx]
        label = int(row["label"])
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", IMAGE_SIZE, (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(split):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_split(split):
    if not METADATA.exists():
        print(f"❌ {METADATA} introuvable — lance preprocessing.py d'abord")
        exit(1)
    records = []
    with open(METADATA, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split and Path(row["image_path"]).exists():
                records.append(row)
    return records


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # ── Gèle seulement les 2 premières couches ──────────────────────────────
    # Libère layer3, layer4, fc → le modèle apprend mieux
    for name, param in model.named_parameters():
        if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False

    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, NUM_CLASSES),
    )
    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    total_loss = correct = total = 0
    tp = fp = tn = fn = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out   = model(imgs)
            loss  = criterion(out, labels)
            preds = out.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            for p, l in zip(preds, labels):
                p, l = p.item(), l.item()
                if l==1 and p==1: tp+=1
                elif l==0 and p==1: fp+=1
                elif l==0 and p==0: tn+=1
                elif l==1 and p==0: fn+=1

    acc  = correct / total if total else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return {
        "loss":acc, "accuracy":acc,
        "precision":prec, "recall":rec, "f1":f1,
        "tp":tp,"fp":fp,"tn":tn,"fn":fn,
        "loss": total_loss/total,
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch":epoch, "model_state":model.state_dict(),
        "optim_state":optimizer.state_dict(),
        "metrics":metrics, "timestamp":datetime.now().isoformat(),
    }, path)


def train(epochs, batch_size, lr, patience):
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("\n" + "="*55)
    print("  ENTRAÎNEMENT v2 — Fix biais 'tout suspect'")
    print("="*55)
    print(f"  Epochs    : {epochs}")
    print(f"  Batch     : {batch_size}")
    print(f"  LR        : {lr}  (↓ réduit vs v1)")
    print(f"  Patience  : {patience}  (↑ augmenté vs v1)")
    print("="*55 + "\n")

    train_records = load_split("train")
    val_records   = load_split("val")

    # ── Rééquilibrage : même nombre normal/anomalie ─────────────────────────
    normals   = [r for r in train_records if int(r["label"]) == 0]
    anomalies = [r for r in train_records if int(r["label"]) == 1]
    n = min(len(normals), len(anomalies))
    random.shuffle(normals); random.shuffle(anomalies)
    balanced = normals[:n] + anomalies[:n]
    random.shuffle(balanced)

    print(f"📊 Train rééquilibré : {n} normales + {n} anomalies = {len(balanced)}")
    print(f"📊 Val               : {len(val_records)} images\n")

    train_ds = InvoiceDataset(balanced,    transform=get_transforms("train"))
    val_ds   = InvoiceDataset(val_records, transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print("🧠 Chargement ResNet-50 pré-entraîné ImageNet...")
    model = build_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Paramètres entraînés : {trainable:,}\n")

    # ── Poids de classe ÉGAUX (pas de surpondération) ───────────────────────
    criterion = nn.CrossEntropyLoss()   # ← pas de weight= ici
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history      = []
    best_f1      = 0.0
    patience_cnt = 0
    best_path    = MODELS_DIR / "best_model.pt"
    last_path    = MODELS_DIR / "last_model.pt"

    print("🚀 Démarrage entraînement...\n")

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = correct = total = 0
        t0 = time.time()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:02d}/{epochs}",
                    unit="batch", ncols=70)

        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds       = out.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            pbar.set_postfix({"loss":f"{loss.item():.4f}",
                              "acc":f"{correct/total:.3f}"})

        val_m      = evaluate(model, val_loader)
        train_acc  = correct / total
        train_loss = train_loss / total
        elapsed    = time.time() - t0

        scheduler.step(val_m["f1"])

        print(f"\n  Epoch {epoch:02d}/{epochs} ({elapsed:.0f}s)")
        print(f"  Train → loss:{train_loss:.4f}  acc:{train_acc:.4f}")
        print(f"  Val   → loss:{val_m['loss']:.4f}  acc:{val_m['accuracy']:.4f}"
              f"  F1:{val_m['f1']:.4f}  Recall:{val_m['recall']:.4f}")
        print(f"  TP:{val_m['tp']} FP:{val_m['fp']} TN:{val_m['tn']} FN:{val_m['fn']}")

        if val_m["f1"] > best_f1:
            best_f1      = val_m["f1"]
            patience_cnt = 0
            save_checkpoint(model, optimizer, epoch, val_m, best_path)
            print(f"  💾 Meilleur modèle (F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            print(f"  ⏳ Patience : {patience_cnt}/{patience}")

        history.append({
            "epoch":epoch,
            "train_loss":round(train_loss,6),
            "train_acc":round(train_acc,6),
            "val_loss":round(val_m["loss"],6),
            "val_acc":round(val_m["accuracy"],6),
            "val_f1":round(val_m["f1"],6),
            "val_precision":round(val_m["precision"],6),
            "val_recall":round(val_m["recall"],6),
        })
        print()

        if patience_cnt >= patience:
            print(f"🛑 Early stopping à l'epoch {epoch}")
            break

    save_checkpoint(model, optimizer, epoch, val_m, last_path)

    with open(LOGS_DIR/"training_history.json","w",encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("="*55)
    print("✅ ENTRAÎNEMENT v2 TERMINÉ")
    print("="*55)
    print(f"   Meilleur F1  : {best_f1:.4f}")
    print(f"   Modèle       : {best_path}")
    print()
    print("▶️  Relance l'API : python src/api/main.py")
    print("="*55 + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR)
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(epochs=args.epochs, batch_size=args.batch,
          lr=args.lr, patience=args.patience)
