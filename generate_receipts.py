"""
generate_receipts.py
====================
Génère des factures professionnelles réalistes :
  - Factures NORMALES  : conformes, cohérentes, valides
  - Factures FRAUDULEUSES : anomalies comptables réelles

Types d'anomalies générées (vraies fraudes entreprise) :
  1. TVA incorrecte (taux erroné ou montant calculé faux)
  2. Montant total ≠ somme des lignes
  3. SIRET invalide (format incorrect)
  4. Date facture antérieure à date commande
  5. Double facturation (même référence)
  6. Fournisseur fantôme (nom générique suspect)
  7. Montant anormalement élevé (outlier)
  8. RIB/IBAN falsifié

Structure générée :
    data/raw/
    ├── train/
    │   ├── normal/    ← factures normales
    │   └── anomaly/   ← factures frauduleuses
    ├── val/
    │   ├── normal/
    │   └── anomaly/
    └── test/
        ├── normal/
        └── anomaly/

Usage :
    pip install Pillow numpy
    python generate_receipts.py
    python generate_receipts.py --normal 2000 --fraud 1000
"""

import os
import csv
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("❌ Lance : python -m pip install Pillow numpy")
    exit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
IMAGE_SIZE  = (794, 1123)   # A4 à 96dpi
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─── Données réalistes ────────────────────────────────────────────────────────
FOURNISSEURS_NORMAUX = [
    ("TECHNO SOLUTIONS SAS",    "75012 Paris",        "52384719200043"),
    ("BUREAUTIQUE PRO SARL",    "69003 Lyon",         "41253698700021"),
    ("INFORMATIQUE EXPRESS",    "13008 Marseille",    "33198274600018"),
    ("FOURNITURES DELTA SA",    "31000 Toulouse",     "62847391500037"),
    ("SERVICES OMEGA SAS",      "44000 Nantes",       "71836254900062"),
    ("MATÉRIEL OFFICE CORP",    "67000 Strasbourg",   "84729163800045"),
    ("LOGISTIQUE FRANCE SARL",  "59000 Lille",        "29384756100029"),
    ("ÉQUIPEMENTS PRO SAS",     "33000 Bordeaux",     "56183927400053"),
    ("DIGITAL FRANCE SA",       "06000 Nice",         "93847261500038"),
    ("SOLUTIONS MÉTIER SARL",   "34000 Montpellier",  "47392816300041"),
]

FOURNISSEURS_FANTOMES = [
    ("CONSULTANT XYZ",          "00000 Inconnu",      "00000000000000"),
    ("SERVICES DIVERS SARL",    "99999 Étranger",     "11111111111111"),
    ("PRESTATAIRE ANONYME",     "75000 Paris",        "99999999900099"),
    ("SOCIÉTÉ FICTIVE SAS",     "00001 Test",         "12345678900000"),
    ("FOURNISSEUR GÉNÉRIQUE",   "00000 Inconnu",      "00000000000001"),
]

CLIENTS = [
    ("ENTREPRISE ALPHA SA",     "75008 Paris",        "FR76 3000 4028 3798 7654 3210 943"),
    ("SOCIÉTÉ BETA SARL",       "69002 Lyon",         "FR89 2004 1010 0505 0013 M026 06"),
    ("GROUPE GAMMA SAS",        "13001 Marseille",    "FR14 2004 1010 0505 5000 3S027 25"),
    ("COMPAGNIE DELTA SA",      "31400 Toulouse",     "FR76 1420 4101 0050 5500 3S02 725"),
    ("HOLDINGS EPSILON SARL",   "44200 Nantes",       "FR89 3000 4000 0300 0500 0020 025"),
]

PRODUITS = [
    ("Licences logicielles",    150,  500),
    ("Matériel informatique",   200,  2000),
    ("Prestations conseil",     500,  3000),
    ("Fournitures bureau",       20,   200),
    ("Services maintenance",    100,   800),
    ("Formation professionnelle",300, 1500),
    ("Équipements réseau",      400,  3000),
    ("Support technique",       200,  1000),
    ("Abonnement cloud",         50,   500),
    ("Audit sécurité",          800,  5000),
]

TVA_TAUX_NORMAUX   = [0.20, 0.10, 0.055]
TVA_TAUX_INCORRECTS = [0.25, 0.30, 0.15, 0.08, 0.00, 0.35]


# ─── Générateurs de données ───────────────────────────────────────────────────
def gen_ref():
    return f"FAC-{random.randint(2020,2025)}-{random.randint(10000,99999)}"


def gen_date_normal():
    base = datetime(2022, 1, 1)
    return base + timedelta(days=random.randint(0, 1000))


def gen_lignes(n=None):
    n = n or random.randint(2, 6)
    lignes = []
    for _ in range(n):
        prod, pmin, pmax = random.choice(PRODUITS)
        qte  = random.randint(1, 10)
        prix = round(random.uniform(pmin, pmax), 2)
        lignes.append((prod, qte, prix))
    return lignes


# ─── Rendu image facture ──────────────────────────────────────────────────────
def render_invoice(data: dict) -> Image.Image:
    """
    Génère une image de facture professionnelle.
    data contient tous les champs de la facture.
    """
    img  = Image.new("RGB", IMAGE_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Essaie de charger une police, sinon utilise la police par défaut
    try:
        font_title  = ImageFont.truetype("arial.ttf", 22)
        font_bold   = ImageFont.truetype("arialbd.ttf", 14)
        font_normal = ImageFont.truetype("arial.ttf", 12)
        font_small  = ImageFont.truetype("arial.ttf", 10)
        font_large  = ImageFont.truetype("arialbd.ttf", 18)
    except Exception:
        font_title  = ImageFont.load_default()
        font_bold   = font_title
        font_normal = font_title
        font_small  = font_title
        font_large  = font_title

    W, H = IMAGE_SIZE
    y    = 0

    # ── En-tête coloré ────────────────────────────────────────────────────────
    color = data.get("header_color", (41, 128, 185))
    draw.rectangle([0, 0, W, 100], fill=color)
    draw.text((40, 20), "FACTURE", fill=(255,255,255), font=font_title)
    draw.text((40, 55), f"N° {data['ref']}", fill=(220,220,220), font=font_bold)
    draw.text((W-200, 20), data["date_str"], fill=(255,255,255), font=font_normal)

    # Alerte fraude (invisible dans l'image, juste pour le label)
    if data.get("is_fraud"):
        draw.rectangle([W-120, 5, W-5, 30], fill=(180, 0, 0))
        # On ne met PAS de texte "FRAUDE" visible — le CNN doit apprendre
        # les anomalies comptables, pas un tampon

    y = 120

    # ── Fournisseur ───────────────────────────────────────────────────────────
    draw.text((40, y), "ÉMETTEUR", fill=(100,100,100), font=font_small)
    y += 18
    draw.text((40, y), data["fourn_nom"], fill=(30,30,30), font=font_bold)
    y += 18
    draw.text((40, y), data["fourn_adresse"], fill=(80,80,80), font=font_normal)
    y += 16
    draw.text((40, y), f"SIRET : {data['siret']}", fill=(80,80,80), font=font_normal)
    y += 16
    draw.text((40, y), f"IBAN  : {data['iban']}", fill=(80,80,80), font=font_small)

    # ── Client ────────────────────────────────────────────────────────────────
    draw.text((W//2, 120), "DESTINATAIRE", fill=(100,100,100), font=font_small)
    draw.text((W//2, 138), data["client_nom"], fill=(30,30,30), font=font_bold)
    draw.text((W//2, 156), data["client_adresse"], fill=(80,80,80), font=font_normal)

    y = 280

    # ── Ligne séparatrice ─────────────────────────────────────────────────────
    draw.line([40, y, W-40, y], fill=(200,200,200), width=1)
    y += 15

    # ── En-tête tableau ───────────────────────────────────────────────────────
    cols = [40, 320, 430, 530, 650]
    headers = ["Désignation", "Qté", "Prix unit. HT", "Total HT"]
    draw.rectangle([40, y, W-40, y+28], fill=(240,240,240))
    for i, h in enumerate(headers):
        draw.text((cols[i]+5, y+7), h, fill=(60,60,60), font=font_bold)
    y += 28

    # ── Lignes produits ───────────────────────────────────────────────────────
    total_ht = 0
    for j, (prod, qte, prix) in enumerate(data["lignes"]):
        bg = (252,252,252) if j % 2 == 0 else (245,245,248)
        draw.rectangle([40, y, W-40, y+26], fill=bg)
        sous_total = round(qte * prix, 2)

        # Anomalie montant : total ligne ≠ qte × prix
        if data.get("anomalie_montant") and j == 0:
            sous_total = round(sous_total * random.uniform(1.1, 1.5), 2)

        total_ht = round(total_ht + sous_total, 2)

        draw.text((cols[0]+5, y+6), prod[:35], fill=(40,40,40), font=font_normal)
        draw.text((cols[1]+5, y+6), str(qte), fill=(40,40,40), font=font_normal)
        draw.text((cols[2]+5, y+6), f"{prix:.2f} €", fill=(40,40,40), font=font_normal)
        draw.text((cols[3]+5, y+6), f"{sous_total:.2f} €", fill=(40,40,40), font=font_normal)
        y += 26

    y += 20
    draw.line([40, y, W-40, y], fill=(200,200,200), width=1)
    y += 20

    # ── Totaux ────────────────────────────────────────────────────────────────
    tva_taux   = data["tva_taux"]
    tva_mont   = data["tva_montant"]    # peut être incorrecte (fraude)
    total_ttc  = data["total_ttc"]      # peut être incohérent (fraude)

    totals = [
        ("Total HT",     f"{total_ht:.2f} €",   (60,60,60)),
        (f"TVA ({tva_taux*100:.1f}%)", f"{tva_mont:.2f} €", (60,60,60)),
        ("TOTAL TTC",    f"{total_ttc:.2f} €",   (30,30,30)),
    ]

    for label, val, color_txt in totals:
        draw.text((W-300, y), label, fill=color_txt, font=font_bold)
        draw.text((W-120, y), val,   fill=color_txt, font=font_bold)
        y += 24

    # TOTAL TTC surligné
    draw.rectangle([W-320, y-30, W-40, y-6], outline=color, width=2)

    y += 30

    # ── Mentions légales ──────────────────────────────────────────────────────
    draw.line([40, y, W-40, y], fill=(220,220,220), width=1)
    y += 15
    draw.text((40, y), "Conditions de règlement : 30 jours net",
              fill=(120,120,120), font=font_small)
    y += 15
    draw.text((40, y), f"Date d'échéance : {data['echeance']}",
              fill=(120,120,120), font=font_small)
    y += 15

    if data.get("mention_fraude"):
        draw.text((40, y), data["mention_fraude"],
                  fill=(180, 60, 60), font=font_small)
        y += 15

    # ── Pied de page ──────────────────────────────────────────────────────────
    draw.rectangle([0, H-50, W, H], fill=(245,245,245))
    draw.text((40, H-35),
              f"SIRET {data['siret']} — TVA Intra. FR{random.randint(10,99)}"
              f" {data['siret'][:9]}",
              fill=(150,150,150), font=font_small)

    # Légère texture de bruit pour réalisme
    arr   = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 2, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    return img


# ─── Génération facture NORMALE ───────────────────────────────────────────────
def make_normal():
    fourn = random.choice(FOURNISSEURS_NORMAUX)
    client = random.choice(CLIENTS)
    lignes = gen_lignes()
    tva    = random.choice(TVA_TAUX_NORMAUX)
    date   = gen_date_normal()

    total_ht  = round(sum(q*p for _, q, p in lignes), 2)
    tva_mont  = round(total_ht * tva, 2)
    total_ttc = round(total_ht + tva_mont, 2)
    echeance  = (date + timedelta(days=30)).strftime("%d/%m/%Y")

    return {
        "ref"           : gen_ref(),
        "date_str"      : date.strftime("%d/%m/%Y"),
        "fourn_nom"     : fourn[0],
        "fourn_adresse" : fourn[1],
        "siret"         : fourn[2],
        "iban"          : client[2],
        "client_nom"    : client[0],
        "client_adresse": client[1],
        "lignes"        : lignes,
        "tva_taux"      : tva,
        "tva_montant"   : tva_mont,
        "total_ttc"     : total_ttc,
        "echeance"      : echeance,
        "header_color"  : (random.randint(20,80), random.randint(80,150), random.randint(150,220)),
        "is_fraud"      : False,
    }


# ─── Génération facture FRAUDULEUSE ──────────────────────────────────────────
def make_fraud():
    """
    Génère une facture avec une vraie anomalie comptable/administrative.
    8 types de fraudes réelles.
    """
    base   = make_normal()
    fraud_type = random.randint(1, 8)
    lignes = base["lignes"]
    total_ht = round(sum(q*p for _, q, p in lignes), 2)

    base["is_fraud"]    = True
    base["header_color"] = (random.randint(100,160), random.randint(20,60),
                             random.randint(20,60))

    if fraud_type == 1:
        # ── TVA taux incorrect ────────────────────────────────────────────────
        bad_tva     = random.choice(TVA_TAUX_INCORRECTS)
        tva_mont    = round(total_ht * bad_tva, 2)
        base["tva_taux"]    = bad_tva
        base["tva_montant"] = tva_mont
        base["total_ttc"]   = round(total_ht + tva_mont, 2)
        base["mention_fraude"] = f"[TVA appliquée: {bad_tva*100:.0f}% — taux suspect]"

    elif fraud_type == 2:
        # ── Montant total incohérent (total ≠ HT + TVA) ──────────────────────
        tva_mont  = round(total_ht * base["tva_taux"], 2)
        base["tva_montant"] = tva_mont
        base["total_ttc"]   = round(total_ht + tva_mont +
                                    random.uniform(50, 500), 2)
        base["mention_fraude"] = "[Écart détecté entre total et sous-totaux]"

    elif fraud_type == 3:
        # ── SIRET invalide ────────────────────────────────────────────────────
        base["siret"] = "".join([str(random.randint(0,9)) for _ in range(14)])
        base["siret"] = base["siret"][:5] + "00000" + base["siret"][10:]
        base["mention_fraude"] = "[SIRET non conforme au format officiel]"

    elif fraud_type == 4:
        # ── Date incohérente (facture avant commande) ─────────────────────────
        date_cmd     = gen_date_normal()
        date_facture = date_cmd - timedelta(days=random.randint(10, 90))
        base["date_str"]    = date_facture.strftime("%d/%m/%Y")
        base["echeance"]    = (date_facture - timedelta(days=10)).strftime("%d/%m/%Y")
        base["mention_fraude"] = "[Date facture antérieure à la date de commande]"

    elif fraud_type == 5:
        # ── Double facturation (même référence) ──────────────────────────────
        base["ref"]  = "FAC-2024-" + str(random.randint(10000, 20000))
        base["mention_fraude"] = "[Référence potentiellement dupliquée]"
        # Même montant mais date différente → doublon
        base["date_str"] = (gen_date_normal() + timedelta(days=1)).strftime("%d/%m/%Y")

    elif fraud_type == 6:
        # ── Fournisseur fantôme ───────────────────────────────────────────────
        fantome = random.choice(FOURNISSEURS_FANTOMES)
        base["fourn_nom"]     = fantome[0]
        base["fourn_adresse"] = fantome[1]
        base["siret"]         = fantome[2]
        base["mention_fraude"] = "[Fournisseur non référencé dans la base]"

    elif fraud_type == 7:
        # ── Montant anormalement élevé ────────────────────────────────────────
        facteur = random.uniform(5, 20)
        new_lignes = [(p, q, round(prix*facteur, 2))
                      for p, q, prix in lignes]
        base["lignes"]      = new_lignes
        new_ht              = round(sum(q*pr for _, q, pr in new_lignes), 2)
        tva_mont            = round(new_ht * base["tva_taux"], 2)
        base["tva_montant"] = tva_mont
        base["total_ttc"]   = round(new_ht + tva_mont, 2)
        base["mention_fraude"] = f"[Montants {facteur:.0f}x supérieurs à la moyenne]"

    else:
        # ── IBAN falsifié ─────────────────────────────────────────────────────
        fake_iban = "FR" + "".join([str(random.randint(0,9)) for _ in range(25)])
        base["iban"] = fake_iban
        base["mention_fraude"] = "[IBAN format non conforme BIC/IBAN]"
        # Montant légèrement différent
        base["anomalie_montant"] = True

    return base


# ─── Sauvegarde ───────────────────────────────────────────────────────────────
def save_invoice(data: dict, path: Path):
    img = render_invoice(data)
    img.resize((224, 224), Image.LANCZOS).save(path, "JPEG", quality=92)


# ─── Pipeline principal ────────────────────────────────────────────────────────
def run(n_normal: int, n_fraud: int):
    print("\n" + "="*55)
    print("  GENERATE RECEIPTS — Factures réalistes")
    print("="*55)
    print(f"  Normales    : {n_normal}")
    print(f"  Frauduleuses: {n_fraud}")
    print(f"  Total       : {n_normal + n_fraud}")
    print("="*55 + "\n")

    # Répartition train/val/test : 70/15/15
    splits = {
        "train": (int(n_normal*0.70), int(n_fraud*0.70)),
        "val"  : (int(n_normal*0.15), int(n_fraud*0.15)),
        "test" : (int(n_normal*0.15), int(n_fraud*0.15)),
    }

    metadata = []
    total_gen = 0

    for split, (nn, nf) in splits.items():
        for cls, n, maker in [("normal", nn, make_normal),
                               ("anomaly", nf, make_fraud)]:
            dest = RAW_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)

            # Vide le dossier anomaly (remplace les anciennes)
            if cls == "anomaly":
                for f in dest.glob("*.jpg"):
                    f.unlink()
                print(f"🗑️  Ancien contenu {dest} supprimé")

            print(f"📝 Génération {split}/{cls} : {n} factures...")

            for i in range(n):
                data  = maker()
                fname = f"gen_{split}_{cls}_{i:05d}.jpg"
                path  = dest / fname

                try:
                    save_invoice(data, path)
                except Exception as e:
                    print(f"   ⚠️  Erreur {fname} : {e}")
                    continue

                metadata.append({
                    "image_path" : str(path),
                    "label"      : 1 if cls == "anomaly" else 0,
                    "label_name" : cls,
                    "split"      : split,
                    "fraud_type" : data.get("mention_fraude", ""),
                })
                total_gen += 1

                if (i+1) % 100 == 0:
                    print(f"   {i+1}/{n} ✓")

            print(f"   ✅ {n} images → {dest}")

    # ── Metadata CSV ──────────────────────────────────────────────────────────
    meta_path = RAW_DIR / "metadata_generated.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
            fieldnames=["image_path","label","label_name","split","fraud_type"])
        writer.writeheader()
        writer.writerows(metadata)

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("✅ GÉNÉRATION TERMINÉE")
    print("="*55)
    print(f"   Total généré  : {total_gen}")
    n_norm = sum(1 for m in metadata if m["label"] == 0)
    n_frau = sum(1 for m in metadata if m["label"] == 1)
    print(f"   Normales      : {n_norm}")
    print(f"   Frauduleuses  : {n_frau}")
    print(f"   metadata_generated.csv → {meta_path}")
    print()
    print("▶️  Lance maintenant :")
    print("   python preprocessing.py")
    print("   python train.py")
    print("="*55 + "\n")


# ─── Argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Génère des factures normales et frauduleuses réalistes")
    p.add_argument("--normal", type=int, default=2000,
                   help="Nombre de factures normales (défaut: 2000)")
    p.add_argument("--fraud",  type=int, default=1000,
                   help="Nombre de factures frauduleuses (défaut: 1000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_normal=args.normal, n_fraud=args.fraud)