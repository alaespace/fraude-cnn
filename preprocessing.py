"""
preprocessing.py
================
Prépare les images de data/raw/ pour l'entraînement CNN anti-fraude.

Ce que fait ce script :
  1. Lit metadata.csv (généré par download_dataset.py)
  2. Vérifie et nettoie les images corrompues
  3. Normalise + redimensionne toutes les images (224x224)
  4. Applique l'augmentation sur les images d'entraînement
  5. Sauvegarde dans data/processed/ avec nouveau metadata
  6. Affiche des statistiques complètes

Structure générée :
    data/processed/
    ├── train/
    │   ├── normal/      ← images normalisées + augmentées
    │   └── anomaly/
    ├── val/
    │   ├── normal/      ← images normalisées seulement
    │   └── anomaly/
    ├── test/
    │   ├── normal/      ← images normalisées seulement
    │   └── anomaly/
    ├── metadata_processed.csv
    └── stats.json       ← statistiques du dataset

Usage :
    python preprocessing.py
    python preprocessing.py --no-augment    ← sans augmentation
    python preprocessing.py --size 128      ← images 128x128
"""

import os
import csv
import json
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ─── Vérification dépendances ─────────────────────────────────────────────────
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("❌ Dépendances manquantes. Lance :")
    print("   python -m pip install Pillow numpy tqdm")
    exit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
METADATA_IN   = RAW_DIR / "metadata.csv"
METADATA_OUT  = PROCESSED_DIR / "metadata_processed.csv"
STATS_OUT     = PROCESSED_DIR / "stats.json"

IMAGE_SIZE    = (224, 224)
RANDOM_SEED   = 42

# Moyenne et écart-type ImageNet (standard pour transfer learning ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Nombre de copies augmentées par image anomalie (rééquilibrage)
AUGMENT_ANOMALY_TIMES = 3   # génère 3x plus d'anomalies
AUGMENT_NORMAL_TIMES  = 1   # normal reste 1x

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Création des dossiers ─────────────────────────────────────────────────────
def build_dirs() -> dict:
    dirs = {}
    for split in ["train", "val", "test"]:
        for cls in ["normal", "anomaly"]:
            p = PROCESSED_DIR / split / cls
            p.mkdir(parents=True, exist_ok=True)
            dirs[f"{split}/{cls}"] = p
    print("✅ Arborescence data/processed/ créée")
    return dirs


# ─── Chargement metadata ───────────────────────────────────────────────────────
def load_metadata() -> list:
    if not METADATA_IN.exists():
        print(f"❌ {METADATA_IN} introuvable.")
        print("   Lance d'abord : python data/download_dataset.py")
        exit(1)

    records = []
    with open(METADATA_IN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"✅ metadata.csv chargé : {len(records)} entrées")
    return records


# ─── Vérification image ────────────────────────────────────────────────────────
def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ─── Normalisation standard ────────────────────────────────────────────────────
def normalize(img: Image.Image) -> Image.Image:
    """
    Normalise une image PIL selon les stats ImageNet.
    Résultat : image PIL prête pour ResNet/EfficientNet.
    """
    arr  = np.array(img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS), dtype=np.float32)
    arr /= 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std

    # Reconvertit en uint8 pour sauvegarde JPEG
    arr  = np.clip(arr * std * 255 + mean * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─── Augmentations ────────────────────────────────────────────────────────────
def augment(img: Image.Image, version: int) -> Image.Image:
    """
    Applique une augmentation différente selon la version.
    Conçu pour les factures : transformations réalistes.
    """
    img = img.copy()

    if version == 0:
        # Rotation légère (document scanné de travers)
        angle = random.uniform(-5, 5)
        img   = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    elif version == 1:
        # Bruit gaussien (mauvaise qualité de scan)
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 8, arr.shape).astype(np.float32)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    elif version == 2:
        # Flou léger (document froissé)
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    elif version == 3:
        # Variation de luminosité (éclairage inégal)
        factor = random.uniform(0.75, 1.35)
        img    = ImageEnhance.Brightness(img).enhance(factor)

    elif version == 4:
        # Variation de contraste
        factor = random.uniform(0.8, 1.4)
        img    = ImageEnhance.Contrast(img).enhance(factor)

    elif version == 5:
        # Flip horizontal (document retourné)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    elif version == 6:
        # Recadrage aléatoire puis resize (zoom)
        w, h  = img.size
        left  = random.randint(0, w // 10)
        top   = random.randint(0, h // 10)
        right = w - random.randint(0, w // 10)
        bot   = h - random.randint(0, h // 10)
        img   = img.crop((left, top, right, bot)).resize(IMAGE_SIZE, Image.LANCZOS)

    elif version == 7:
        # Netteté augmentée (photocopie trop nette = suspect)
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(2.0, 4.0))

    return img


# ─── Traitement d'une image ────────────────────────────────────────────────────
def process_image(
    src_path: Path,
    dst_dir:  Path,
    filename: str,
    split:    str,
    cls:      str,
    augment_times: int,
    do_augment: bool,
) -> list:
    """
    Charge, normalise, (augmente), sauvegarde une image.
    Retourne la liste des métadonnées générées.
    """
    results = []

    if not src_path.exists():
        return results

    if not is_valid_image(src_path):
        print(f"   ⚠️  Image corrompue ignorée : {src_path.name}")
        return results

    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        print(f"   ⚠️  Erreur ouverture {src_path.name} : {e}")
        return results

    # ── Version de base (normalisée) ─────────────────────────────────────────
    base_img  = normalize(img)
    base_name = f"{filename}_base.jpg"
    base_path = dst_dir / base_name
    base_img.save(base_path, "JPEG", quality=92)

    results.append({
        "image_path" : str(base_path),
        "label"      : 1 if cls == "anomaly" else 0,
        "label_name" : cls,
        "split"      : split,
        "augmented"  : 0,
        "version"    : "base",
    })

    # ── Augmentations (train seulement) ──────────────────────────────────────
    if do_augment and split == "train" and augment_times > 1:
        versions = random.sample(range(8), min(augment_times - 1, 8))
        for v in versions:
            aug_img  = augment(img, v)
            aug_img  = normalize(aug_img)
            aug_name = f"{filename}_aug{v}.jpg"
            aug_path = dst_dir / aug_name
            aug_img.save(aug_path, "JPEG", quality=92)

            results.append({
                "image_path" : str(aug_path),
                "label"      : 1 if cls == "anomaly" else 0,
                "label_name" : cls,
                "split"      : split,
                "augmented"  : 1,
                "version"    : f"aug{v}",
            })

    return results


# ─── Pipeline principal ────────────────────────────────────────────────────────
def run(do_augment: bool, image_size: tuple):
    global IMAGE_SIZE
    IMAGE_SIZE = image_size

    print("\n" + "=" * 55)
    print("  PREPROCESSING — CNN Anti-Fraude Factures")
    print("=" * 55)
    print(f"  Source      : {RAW_DIR}")
    print(f"  Destination : {PROCESSED_DIR}")
    print(f"  Taille      : {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px")
    print(f"  Augmentation: {'✅ activée' if do_augment else '❌ désactivée'}")
    print("=" * 55 + "\n")

    dirs     = build_dirs()
    records  = load_metadata()
    metadata = []
    stats    = defaultdict(lambda: defaultdict(int))

    # Grouper par split
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    for split, items in by_split.items():
        print(f"\n📂 Traitement split [{split}] — {len(items)} images...")

        # Augmentation : anomalies x3, normales x1 (train seulement)
        pbar = tqdm(items, desc=f"  {split}", unit="img", ncols=70)

        for item in pbar:
            src_path = Path(item["image_path"])
            cls      = item["label_name"]
            dst_dir  = dirs[f"{split}/{cls}"]

            stem          = src_path.stem
            augment_times = (
                AUGMENT_ANOMALY_TIMES if cls == "anomaly"
                else AUGMENT_NORMAL_TIMES
            )

            results = process_image(
                src_path    = src_path,
                dst_dir     = dst_dir,
                filename    = stem,
                split       = split,
                cls         = cls,
                augment_times = augment_times,
                do_augment  = do_augment,
            )

            metadata.extend(results)

            for r in results:
                stats[split][cls] += 1

        print(f"   normal  : {stats[split]['normal']:5d} images")
        print(f"   anomaly : {stats[split]['anomaly']:5d} images")

    # ── Écriture metadata_processed.csv ──────────────────────────────────────
    with open(METADATA_OUT, "w", newline="", encoding="utf-8") as f:
        fields = ["image_path", "label", "label_name", "split", "augmented", "version"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)

    # ── Écriture stats.json ───────────────────────────────────────────────────
    total       = len(metadata)
    n_normal    = sum(1 for m in metadata if m["label"] == 0)
    n_anomaly   = sum(1 for m in metadata if m["label"] == 1)
    n_augmented = sum(1 for m in metadata if m["augmented"] == 1)

    stats_data = {
        "total_images"     : total,
        "normal"           : n_normal,
        "anomaly"          : n_anomaly,
        "augmented"        : n_augmented,
        "image_size"       : list(IMAGE_SIZE),
        "imagenet_mean"    : IMAGENET_MEAN,
        "imagenet_std"     : IMAGENET_STD,
        "splits"           : {s: dict(v) for s, v in stats.items()},
    }

    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    # ── Résumé final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅ PREPROCESSING TERMINÉ")
    print("=" * 55)
    print(f"   Total images     : {total}")
    print(f"   Normal           : {n_normal}  ({n_normal/total*100:.1f}%)")
    print(f"   Anomalie         : {n_anomaly}  ({n_anomaly/total*100:.1f}%)")
    print(f"   Augmentées       : {n_augmented}")
    print(f"   metadata_processed.csv → {METADATA_OUT}")
    print(f"   stats.json             → {STATS_OUT}")
    print()
    print("▶️  Prochaine étape :")
    print("   python mypro/src/detection/train.py")
    print("=" * 55 + "\n")


# ─── Argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Preprocessing CNN anti-fraude")
    p.add_argument(
        "--no-augment", action="store_true",
        help="Désactive l'augmentation des données"
    )
    p.add_argument(
        "--size", type=int, default=224,
        help="Taille des images carrées en pixels (défaut: 224)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        do_augment = not args.no_augment,
        image_size = (args.size, args.size),
    )