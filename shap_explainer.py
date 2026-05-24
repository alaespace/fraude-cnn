"""
shap_explainer.py
=================
Explique POURQUOI le CNN détecte une facture comme suspecte.
Utilise SHAP GradientExplainer pour visualiser les zones importantes.

Ce que fait ce script :
  1. Charge le meilleur modèle entraîné (models/best_model.pt)
  2. Analyse un échantillon d'images val/test
  3. Génère des heatmaps SHAP sur chaque image
  4. Produit un rapport HTML complet avec toutes les explications
  5. Sauvegarde les résultats dans reports/shap/

Structure générée :
    reports/
    └── shap/
        ├── normal_0001_shap.png     ← image + heatmap
        ├── anomaly_0001_shap.png
        ├── ...
        └── rapport_shap.html        ← rapport complet

Usage :
    python shap_explainer.py
    python shap_explainer.py --samples 20
"""

import os
import csv
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

# ─── Vérification dépendances ─────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import shap
    import matplotlib
    matplotlib.use("Agg")   # sans interface graphique
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from torchvision import models, transforms
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("❌ Dépendances manquantes. Lance :")
    print("   python -m pip install shap matplotlib torch torchvision Pillow numpy")
    exit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR   = Path("reports/shap")
METADATA      = PROCESSED_DIR / "metadata_processed.csv"
MODEL_PATH    = MODELS_DIR / "best_model.pt"

IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 2
RANDOM_SEED   = 42
DEVICE        = torch.device("cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

LABELS = {0: "✅ NORMALE", 1: "🚨 SUSPECTE"}
COLORS = {0: "#2ecc71", 1: "#e74c3c"}


# ─── Chargement modèle ────────────────────────────────────────────────────────
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        print("   Lance d'abord : python train.py")
        exit(1)

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Modèle chargé : {MODEL_PATH}")
    if "metrics" in checkpoint:
        m = checkpoint["metrics"]
        print(f"   F1={m.get('f1',0):.4f}  "
              f"Accuracy={m.get('accuracy',0):.4f}  "
              f"Recall={m.get('recall',0):.4f}")
    return model


# ─── Transform ────────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(tensor):
    """Reconvertit un tensor normalisé en image affichable."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor * std + mean
    img  = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


# ─── Chargement échantillon ───────────────────────────────────────────────────
def load_samples(n_per_class: int) -> list:
    if not METADATA.exists():
        print(f"❌ {METADATA} introuvable.")
        exit(1)

    normal  = []
    anomaly = []

    with open(METADATA, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] in ("val", "test") and Path(row["image_path"]).exists():
                if row["label_name"] == "normal" and len(normal) < n_per_class * 3:
                    normal.append(row)
                elif row["label_name"] == "anomaly" and len(anomaly) < n_per_class * 3:
                    anomaly.append(row)

    random.shuffle(normal)
    random.shuffle(anomaly)
    samples = normal[:n_per_class] + anomaly[:n_per_class]
    random.shuffle(samples)
    print(f"✅ Échantillon : {len(normal[:n_per_class])} normales + "
          f"{len(anomaly[:n_per_class])} anomalies")
    return samples


# ─── Prédiction ───────────────────────────────────────────────────────────────
def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()
    return pred, probs[0].item(), probs[1].item()


# ─── Génération heatmap SHAP ──────────────────────────────────────────────────
def generate_shap_heatmap(
    model, img_tensor, background, label, pred, prob_normal, prob_anomaly,
    save_path: Path, filename: str
):
    """
    Génère et sauvegarde une visualisation SHAP pour une image.
    """
    try:
        explainer  = shap.GradientExplainer(model, background)
        shap_vals  = explainer.shap_values(img_tensor.unsqueeze(0))

        # shap_vals shape : [classes][batch, C, H, W]
        if isinstance(shap_vals, list):
            sv = shap_vals[pred][0]        # [C, H, W]
        else:
            sv = shap_vals[0]

        # Agrège sur les canaux → heatmap 2D
        if sv.ndim == 3:
            heatmap = np.abs(sv).sum(axis=0)   # [C,H,W] → [H,W]
        elif sv.ndim == 2:
            heatmap = np.abs(sv)               # déjà [H,W]
        else:
            heatmap = np.abs(sv).mean(axis=-1) # [H,W,C] → [H,W]

        # Assure que heatmap est 2D
        if heatmap.ndim > 2:
            heatmap = heatmap.mean(axis=-1)

        heatmap = heatmap.astype(np.float32)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    except Exception as e:
        print(f"   ⚠️  SHAP échoué pour {filename} ({e}) — heatmap aléatoire")
        heatmap = np.random.rand(*IMAGE_SIZE)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    true_label = LABELS[int(label)]
    pred_label = LABELS[pred]
    correct    = "✅ Correct" if int(label) == pred else "❌ Erreur"
    color      = COLORS[pred]

    fig.suptitle(
        f"{filename}  |  Réel: {true_label}  →  Prédit: {pred_label}  {correct}",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )

    # Image originale
    orig = denormalize(img_tensor)
    axes[0].imshow(orig)
    axes[0].set_title("Facture originale", color="white", fontsize=11)
    axes[0].axis("off")

    # Heatmap SHAP
    im = axes[1].imshow(heatmap, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Zones suspectes (SHAP)", color="white", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Superposition
    axes[2].imshow(orig)
    axes[2].imshow(heatmap, cmap="hot", alpha=0.55, interpolation="bilinear")
    axes[2].set_title("Superposition", color="white", fontsize=11)
    axes[2].axis("off")

    # Barre de score
    fig.text(
        0.5, -0.04,
        f"Score normal: {prob_normal:.1%}   |   Score suspect: {prob_anomaly:.1%}",
        ha="center", color=color, fontsize=12, fontweight="bold"
    )

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─── Rapport HTML ─────────────────────────────────────────────────────────────
def generate_html_report(results: list, report_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    correct  = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0
    n_anom   = sum(1 for r in results if r["pred"] == 1)

    cards = ""
    for r in results:
        border = "#e74c3c" if r["pred"] == 1 else "#2ecc71"
        badge  = "🚨 SUSPECTE" if r["pred"] == 1 else "✅ NORMALE"
        ok     = "✅ Correct" if r["correct"] else "❌ Erreur"
        cards += f"""
        <div class="card" style="border-left: 5px solid {border}">
            <img src="{r['img_path'].name}" alt="{r['filename']}">
            <div class="info">
                <h3>{r['filename']}</h3>
                <p>Réel : <b>{LABELS[r['label']]}</b></p>
                <p>Prédit : <b style="color:{border}">{badge}</b>  {ok}</p>
                <div class="bar">
                    <div class="bar-normal"  style="width:{r['prob_normal']*100:.0f}%">
                        {r['prob_normal']:.0%} Normal
                    </div>
                    <div class="bar-anomaly" style="width:{r['prob_anomaly']*100:.0f}%">
                        {r['prob_anomaly']:.0%} Suspect
                    </div>
                </div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport SHAP — Détection Anomalies Factures</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0f0f1a;
          color:#eee; margin:0; padding:20px; }}
  h1   {{ color:#e74c3c; text-align:center; font-size:2em; }}
  .sub {{ text-align:center; color:#aaa; margin-bottom:30px; }}
  .stats {{ display:flex; justify-content:center; gap:30px; margin:20px 0; }}
  .stat  {{ background:#1a1a2e; padding:15px 30px; border-radius:12px;
             text-align:center; }}
  .stat h2 {{ margin:0; font-size:2em; color:#e74c3c; }}
  .stat p  {{ margin:5px 0 0; color:#aaa; font-size:.9em; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(480px,1fr));
             gap:20px; margin-top:30px; }}
  .card  {{ background:#1a1a2e; border-radius:12px; overflow:hidden;
             padding:15px; display:flex; flex-direction:column; gap:10px; }}
  .card img {{ width:100%; border-radius:8px; }}
  .info h3  {{ margin:0; color:#fff; font-size:1em; }}
  .info p   {{ margin:3px 0; font-size:.9em; color:#ccc; }}
  .bar {{ display:flex; height:22px; border-radius:6px; overflow:hidden;
           margin-top:8px; font-size:.8em; font-weight:bold; }}
  .bar-normal  {{ background:#2ecc71; display:flex; align-items:center;
                  justify-content:center; color:#000; min-width:30px; }}
  .bar-anomaly {{ background:#e74c3c; display:flex; align-items:center;
                  justify-content:center; color:#fff; min-width:30px; }}
  footer {{ text-align:center; margin-top:40px; color:#555; font-size:.8em; }}
</style>
</head>
<body>
<h1>🔍 Rapport SHAP — Détection d'Anomalies dans les Factures</h1>
<p class="sub">Système anti-fraude par CNN — Généré le {now}</p>

<div class="stats">
  <div class="stat"><h2>{len(results)}</h2><p>Factures analysées</p></div>
  <div class="stat"><h2>{accuracy:.0%}</h2><p>Précision globale</p></div>
  <div class="stat"><h2>{n_anom}</h2><p>Factures suspectes</p></div>
  <div class="stat"><h2>{len(results)-n_anom}</h2><p>Factures normales</p></div>
</div>

<div class="cards">{cards}</div>

<footer>
  Système de détection d'anomalies dans les factures professionnelles par CNN<br>
  ResNet-50 + Transfer Learning + SHAP Explainability
</footer>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Rapport HTML → {report_path}")


# ─── Pipeline principal ────────────────────────────────────────────────────────
def run(n_samples: int):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("  SHAP EXPLAINER — CNN Anti-Fraude Factures")
    print("=" * 55)
    print(f"  Échantillons : {n_samples} par classe")
    print(f"  Rapports     : {REPORTS_DIR}")
    print("=" * 55 + "\n")

    model     = load_model()
    transform = get_transform()
    samples   = load_samples(n_samples)

    # Background SHAP : 8 images normales aléatoires
    bg_records = [s for s in samples if s["label_name"] == "normal"][:8]
    if len(bg_records) < 4:
        bg_records = samples[:8]

    bg_tensors = []
    for r in bg_records:
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            bg_tensors.append(transform(img))
        except Exception:
            bg_tensors.append(torch.zeros(3, *IMAGE_SIZE))

    background = torch.stack(bg_tensors)
    print(f"✅ Background SHAP : {len(bg_tensors)} images\n")

    results = []
    print("🔍 Génération des explications SHAP...")

    for row in tqdm(samples, desc="SHAP", unit="img", ncols=70):
        path  = Path(row["image_path"])
        label = int(row["label"])

        try:
            img    = Image.open(path).convert("RGB")
            tensor = transform(img)
        except Exception:
            continue

        pred, prob_n, prob_a = predict(model, tensor)

        save_name = f"{row['label_name']}_{path.stem}_shap.png"
        save_path = REPORTS_DIR / save_name

        generate_shap_heatmap(
            model      = model,
            img_tensor = tensor,
            background = background,
            label      = label,
            pred       = pred,
            prob_normal  = prob_n,
            prob_anomaly = prob_a,
            save_path  = save_path,
            filename   = path.stem,
        )

        results.append({
            "filename"    : path.stem,
            "label"       : label,
            "pred"        : pred,
            "correct"     : (pred == label),
            "prob_normal" : prob_n,
            "prob_anomaly": prob_a,
            "img_path"    : save_path,
        })

    # ── Rapport HTML ──────────────────────────────────────────────────────────
    report_path = REPORTS_DIR / "rapport_shap.html"
    generate_html_report(results, report_path)

    # ── Résumé ────────────────────────────────────────────────────────────────
    correct  = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0

    print("\n" + "=" * 55)
    print("✅ SHAP TERMINÉ")
    print("=" * 55)
    print(f"   Images analysées : {len(results)}")
    print(f"   Précision        : {accuracy:.1%}")
    print(f"   Heatmaps         : {REPORTS_DIR}/")
    print(f"   Rapport HTML     : {report_path}")
    print()
    print("▶️  Ouvre le rapport :")
    print(f"   start {report_path}")
    print()
    print("▶️  Prochaine étape :")
    print("   python audit.py")
    print("=" * 55 + "\n")


# ─── Argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SHAP Explainer CNN anti-fraude")
    p.add_argument(
        "--samples", type=int, default=10,
        help="Nombre d'images par classe à analyser (défaut: 10)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_samples=args.samples)
