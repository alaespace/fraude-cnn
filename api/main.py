"""
src/api/main.py
===============
API FastAPI pour le système de détection d'anomalies
dans les factures professionnelles par CNN.

Endpoints :
  POST /analyze          ← analyse une image de facture
  GET  /stats            ← statistiques globales
  GET  /history          ← historique des décisions
  GET  /health           ← état du serveur

Usage :
    python src/api/main.py
    ou
    uvicorn src.api.main:app --reload --port 8000
"""

import io
import sys
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# ─── Vérification dépendances ─────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("❌ Lance : python -m pip install fastapi uvicorn python-multipart torch torchvision Pillow")
    sys.exit(1)

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pt"
DB_PATH    = ROOT / "logs" / "audit.db"
LOGS_DIR   = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE  = (224, 224)
NUM_CLASSES = 2
DEVICE      = torch.device("cpu")
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

LABELS = {0: "NORMALE", 1: "SUSPECTE"}
SEUIL  = 0.60   # score anomalie > 60% → SUSPECTE


# ─── Chargement modèle ────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    model = models.resnet50(weights=None)
    in_f  = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("metrics", {})


# ─── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─── Base de données ──────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_decisions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT,
            sha256       TEXT,
            decision     TEXT,
            score_normal REAL,
            score_anomaly REAL,
            analysed_at  TEXT
        )
    """)
    conn.commit()
    return conn


def log_decision(filename, sha256, decision, score_n, score_a):
    conn = get_db()
    conn.execute("""
        INSERT INTO api_decisions
        (filename, sha256, decision, score_normal, score_anomaly, analysed_at)
        VALUES (?,?,?,?,?,?)
    """, (filename, sha256, decision, score_n, score_a, datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Chargement du modèle CNN...")
    try:
        model, metrics = load_model()
        ml_model["model"]   = model
        ml_model["metrics"] = metrics
        ml_model["loaded"]  = True
        print(f"✅ Modèle chargé — F1={metrics.get('f1',0):.4f}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        ml_model["loaded"] = False
    yield
    ml_model.clear()
    print("🛑 Serveur arrêté")


# ─── App FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "API Anti-Fraude Factures",
    description = "Détection d'anomalies dans les factures par CNN ResNet-50",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # React sur localhost:3000
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "status"      : "ok" if ml_model.get("loaded") else "model_not_loaded",
        "model_loaded": ml_model.get("loaded", False),
        "timestamp"   : datetime.now().isoformat(),
        "version"     : "1.0.0",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyse une image de facture et retourne le score d'anomalie.

    Returns:
        decision      : "NORMALE" ou "SUSPECTE"
        score_normal  : probabilité [0-1]
        score_anomaly : probabilité [0-1]
        alerte        : true si score_anomaly > seuil
        sha256        : hash intégrité du fichier
    """
    if not ml_model.get("loaded"):
        raise HTTPException(503, "Modèle non chargé")

    # Validation type fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Fichier doit être une image (JPEG, PNG, etc.)")

    # Lecture image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Fichier vide")

    # Hash SHA256
    sha256 = hashlib.sha256(contents).hexdigest()

    # Preprocessing
    try:
        img    = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(422, f"Impossible de lire l'image : {e}")

    # Inférence CNN
    model = ml_model["model"]
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        score_normal  = round(probs[0].item(), 4)
        score_anomaly = round(probs[1].item(), 4)
        pred          = probs.argmax().item()

    decision = "SUSPECTE" if score_anomaly >= SEUIL else "NORMALE"
    alerte   = score_anomaly >= SEUIL

    # Log en base
    log_decision(file.filename, sha256, decision, score_normal, score_anomaly)

    return {
        "filename"     : file.filename,
        "decision"     : decision,
        "score_normal" : score_normal,
        "score_anomaly": score_anomaly,
        "alerte"       : alerte,
        "sha256"       : sha256[:16] + "...",
        "seuil"        : SEUIL,
        "analysed_at"  : datetime.now().isoformat(),
        "message"      : (
            "🚨 Facture SUSPECTE — vérification humaine requise"
            if alerte else
            "✅ Facture NORMALE — aucune anomalie détectée"
        ),
    }


@app.get("/stats")
async def stats():
    """Statistiques globales du système."""
    conn = get_db()
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) as total FROM api_decisions")
    total = cur.fetchone()["total"]

    cur.execute("SELECT COUNT(*) FROM api_decisions WHERE decision='SUSPECTE'")
    n_susp = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM api_decisions WHERE decision='NORMALE'")
    n_norm = cur.fetchone()[0]

    cur.execute("""
        SELECT AVG(score_anomaly) as avg_score FROM api_decisions
    """)
    avg = cur.fetchone()["avg_score"] or 0
    conn.close()

    metrics = ml_model.get("metrics", {})

    return {
        "total_analyses"  : total,
        "normales"        : n_norm,
        "suspectes"       : n_susp,
        "taux_fraude"     : round(n_susp / total, 4) if total > 0 else 0,
        "score_moyen"     : round(avg, 4),
        "model_f1"        : round(metrics.get("f1", 0), 4),
        "model_accuracy"  : round(metrics.get("accuracy", 0), 4),
        "model_recall"    : round(metrics.get("recall", 0), 4),
        "seuil_alerte"    : SEUIL,
    }


@app.get("/history")
async def history(limit: int = 20):
    """Historique des dernières décisions."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT filename, decision, score_normal, score_anomaly, analysed_at
        FROM api_decisions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"decisions": rows, "count": len(rows)}


# ─── Lancement ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  API Anti-Fraude Factures — FastAPI")
    print("=" * 55)
    print(f"  Modèle : {MODEL_PATH}")
    print(f"  DB     : {DB_PATH}")
    print(f"  URL    : http://localhost:8000")
    print(f"  Docs   : http://localhost:8000/docs")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)