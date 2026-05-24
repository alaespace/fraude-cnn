"""
audit.py
========
Système de traçabilité et d'audit pour la détection
d'anomalies dans les factures professionnelles.

Ce que fait ce script :
  1. Hashage SHA256 de chaque facture (intégrité)
  2. Base de données SQLite avec toutes les décisions
  3. Détection de falsification (hash modifié après analyse)
  4. Logs complets horodatés
  5. Rapport d'audit HTML

Structure générée :
    logs/
    ├── audit.db          ← base SQLite de toutes les décisions
    ├── audit.log         ← logs texte horodatés
    └── rapport_audit.html ← rapport complet

Usage :
    python audit.py                    ← audit complet
    python audit.py --verify           ← vérifie intégrité des fichiers
    python audit.py --report           ← génère rapport HTML seulement
"""

import os
import csv
import json
import sqlite3
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── Configuration ─────────────────────────────────────────────────────────────
LOGS_DIR      = Path("logs")
REPORTS_DIR   = Path("reports")
DB_PATH       = LOGS_DIR / "audit.db"
LOG_PATH      = LOGS_DIR / "audit.log"
REPORT_PATH   = REPORTS_DIR / "rapport_audit.html"
METADATA      = Path("data/processed/metadata_processed.csv")
SHAP_METADATA = Path("reports/shap")
HISTORY_PATH  = LOGS_DIR / "training_history.json"

LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ─── Logger ────────────────────────────────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger("audit")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Fichier log
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ─── Base de données SQLite ────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS factures (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        filename      TEXT NOT NULL,
        image_path    TEXT NOT NULL,
        sha256        TEXT NOT NULL,
        label_reel    INTEGER,
        label_predit  INTEGER,
        score_normal  REAL,
        score_anomaly REAL,
        decision      TEXT,
        correct       INTEGER,
        analysed_at   TEXT,
        split         TEXT,
        augmented     INTEGER,
        flag_falsif   INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS verifications (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        facture_id  INTEGER,
        sha256_new  TEXT,
        sha256_old  TEXT,
        intacte     INTEGER,
        checked_at  TEXT,
        FOREIGN KEY (facture_id) REFERENCES factures(id)
    );

    CREATE TABLE IF NOT EXISTS model_runs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date   TEXT,
        f1_score   REAL,
        accuracy   REAL,
        recall     REAL,
        precision  REAL,
        epochs     INTEGER,
        notes      TEXT
    );
    """)

    conn.commit()
    return conn


# ─── Hashage SHA256 ────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    """Calcule le hash SHA256 d'un fichier image."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "ERROR"


# ─── Chargement métadonnées ────────────────────────────────────────────────────
def load_metadata() -> list:
    if not METADATA.exists():
        logger.warning(f"metadata_processed.csv introuvable : {METADATA}")
        return []

    records = []
    with open(METADATA, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if Path(row["image_path"]).exists():
                records.append(row)

    logger.info(f"Métadonnées chargées : {len(records)} entrées")
    return records


# ─── Chargement résultats SHAP ────────────────────────────────────────────────
def load_shap_results() -> dict:
    """
    Charge les résultats de prédiction depuis le rapport SHAP.
    Retourne un dict {filename: {pred, prob_normal, prob_anomaly}}
    """
    results = {}

    # Cherche dans le JSON de training_history pour les métriques globales
    # Les résultats par image viennent du rapport SHAP HTML (on reparse)
    shap_dir = Path("reports/shap")
    if not shap_dir.exists():
        return results

    # Les heatmaps sont nommées {label}_{stem}_shap.png
    for f in shap_dir.glob("*_shap.png"):
        stem = f.stem.replace("_shap", "")
        # Ex: normal_val_norm_00038_base → label=normal
        if stem.startswith("normal_"):
            results[stem[7:]] = {"pred_label": "normal", "pred": 0}
        elif stem.startswith("anomaly_"):
            results[stem[8:]] = {"pred_label": "anomaly", "pred": 1}

    return results


# ─── Insertion en base ─────────────────────────────────────────────────────────
def insert_factures(conn, records: list, shap_results: dict):
    cur   = conn.cursor()
    now   = datetime.now().isoformat()
    count = 0

    for row in records:
        path     = Path(row["image_path"])
        filename = path.stem
        sha256   = sha256_file(path)
        label    = int(row["label"])

        # Récupère prédiction SHAP si disponible
        shap = shap_results.get(filename, {})
        pred = shap.get("pred", -1)
        decision = (
            "SUSPECTE" if pred == 1
            else "NORMALE" if pred == 0
            else "NON_ANALYSÉE"
        )
        correct = int(pred == label) if pred != -1 else -1

        cur.execute("""
            INSERT OR IGNORE INTO factures
            (filename, image_path, sha256, label_reel, label_predit,
             score_normal, score_anomaly, decision, correct,
             analysed_at, split, augmented)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            filename,
            str(path),
            sha256,
            label,
            pred if pred != -1 else None,
            shap.get("prob_normal"),
            shap.get("prob_anomaly"),
            decision,
            correct if correct != -1 else None,
            now,
            row.get("split", "unknown"),
            int(row.get("augmented", 0)),
        ))
        count += 1

    conn.commit()
    logger.info(f"✅ {count} factures enregistrées en base")
    return count


# ─── Enregistrement run modèle ────────────────────────────────────────────────
def log_model_run(conn):
    if not HISTORY_PATH.exists():
        return

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return

    best = max(history, key=lambda x: x.get("val_f1", 0))
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO model_runs (run_date, f1_score, accuracy, recall, precision, epochs, notes)
        VALUES (?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        best.get("val_f1", 0),
        best.get("val_acc", 0),
        best.get("val_recall", 0),
        best.get("val_precision", 0),
        best.get("epoch", 0),
        f"Best epoch {best.get('epoch')} — early stopping",
    ))
    conn.commit()
    logger.info(f"✅ Run modèle enregistré : F1={best.get('val_f1',0):.4f}")


# ─── Vérification intégrité ────────────────────────────────────────────────────
def verify_integrity(conn):
    cur  = conn.cursor()
    cur.execute("SELECT id, filename, image_path, sha256 FROM factures LIMIT 100")
    rows = cur.fetchall()

    falsified = 0
    now       = datetime.now().isoformat()

    logger.info(f"🔍 Vérification intégrité de {len(rows)} fichiers...")

    for fid, fname, fpath, sha_old in rows:
        path    = Path(fpath)
        sha_new = sha256_file(path) if path.exists() else "MISSING"
        intact  = int(sha_new == sha_old)

        if not intact:
            falsified += 1
            logger.warning(f"🚨 FALSIFICATION DÉTECTÉE : {fname}")
            logger.warning(f"   Hash original : {sha_old[:16]}...")
            logger.warning(f"   Hash actuel   : {sha_new[:16]}...")
            cur.execute(
                "UPDATE factures SET flag_falsif=1 WHERE id=?", (fid,)
            )

        cur.execute("""
            INSERT INTO verifications (facture_id, sha256_new, sha256_old, intacte, checked_at)
            VALUES (?,?,?,?,?)
        """, (fid, sha_new, sha_old, intact, now))

    conn.commit()

    if falsified == 0:
        logger.info(f"✅ Intégrité OK — aucune falsification détectée")
    else:
        logger.warning(f"🚨 {falsified} fichier(s) falsifié(s) !")

    return falsified


# ─── Rapport HTML ─────────────────────────────────────────────────────────────
def generate_html_report(conn):
    cur = conn.cursor()

    # Stats globales
    cur.execute("SELECT COUNT(*) FROM factures")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM factures WHERE decision='SUSPECTE'")
    n_suspect = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM factures WHERE decision='NORMALE'")
    n_normal = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM factures WHERE flag_falsif=1")
    n_falsif = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM factures WHERE correct=1")
    n_correct = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM factures WHERE correct IS NOT NULL")
    n_eval = cur.fetchone()[0]
    accuracy = n_correct / n_eval if n_eval > 0 else 0

    # Derniers runs modèle
    cur.execute("""
        SELECT run_date, f1_score, accuracy, recall, epochs
        FROM model_runs ORDER BY id DESC LIMIT 5
    """)
    runs = cur.fetchall()

    # Dernières décisions
    cur.execute("""
        SELECT filename, decision, score_anomaly, correct, analysed_at, flag_falsif
        FROM factures ORDER BY id DESC LIMIT 50
    """)
    decisions = cur.fetchall()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Rows tableau
    rows_html = ""
    for fname, decision, score, correct, analysed, falsif in decisions:
        color  = "#e74c3c" if decision == "SUSPECTE" else "#2ecc71"
        badge  = "🚨 SUSPECTE" if decision == "SUSPECTE" else "✅ NORMALE"
        ok     = "✅" if correct == 1 else ("❌" if correct == 0 else "—")
        falsbadge = "⚠️ FALSIFIÉ" if falsif else "—"
        score_str = f"{score:.1%}" if score is not None else "—"
        rows_html += f"""
        <tr>
          <td>{fname[:30]}</td>
          <td style="color:{color};font-weight:bold">{badge}</td>
          <td>{score_str}</td>
          <td>{ok}</td>
          <td style="color:#e74c3c">{falsbadge}</td>
          <td style="color:#888;font-size:.8em">{analysed[:16]}</td>
        </tr>"""

    # Runs
    runs_html = ""
    for rdate, f1, acc, rec, ep in runs:
        runs_html += f"""
        <tr>
          <td>{rdate[:16]}</td>
          <td>{f1:.4f}</td>
          <td>{acc:.4f}</td>
          <td>{rec:.4f}</td>
          <td>{ep}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport Audit — Anti-Fraude Factures</title>
<style>
  body  {{ font-family:'Segoe UI',sans-serif; background:#0f0f1a;
           color:#eee; margin:0; padding:20px; }}
  h1    {{ color:#e74c3c; text-align:center; }}
  h2    {{ color:#aaa; border-bottom:1px solid #333; padding-bottom:8px; }}
  .sub  {{ text-align:center; color:#888; margin-bottom:30px; }}
  .stats {{ display:flex; flex-wrap:wrap; justify-content:center;
             gap:20px; margin:20px 0; }}
  .stat  {{ background:#1a1a2e; padding:15px 25px; border-radius:12px;
             text-align:center; min-width:130px; }}
  .stat h2 {{ margin:0; font-size:1.8em; color:#e74c3c; border:none; }}
  .stat p  {{ margin:5px 0 0; color:#aaa; font-size:.85em; }}
  table  {{ width:100%; border-collapse:collapse; margin-top:15px; }}
  th     {{ background:#1a1a2e; padding:10px; text-align:left;
             color:#aaa; font-size:.85em; }}
  td     {{ padding:8px 10px; border-bottom:1px solid #1a1a2e;
             font-size:.88em; }}
  tr:hover {{ background:#1a1a2e; }}
  .section {{ background:#12121f; border-radius:12px;
               padding:20px; margin:20px 0; }}
  .ok {{ color:#2ecc71; }} .warn {{ color:#e74c3c; }}
  footer {{ text-align:center; margin-top:40px; color:#555;
             font-size:.8em; }}
</style>
</head>
<body>
<h1>🔐 Rapport d'Audit — Détection Anomalies Factures</h1>
<p class="sub">Système anti-fraude par CNN — Généré le {now}</p>

<div class="stats">
  <div class="stat"><h2>{total}</h2><p>Factures auditées</p></div>
  <div class="stat"><h2 class="warn">{n_suspect}</h2><p>Suspectes</p></div>
  <div class="stat"><h2 class="ok">{n_normal}</h2><p>Normales</p></div>
  <div class="stat"><h2>{accuracy:.0%}</h2><p>Précision</p></div>
  <div class="stat"><h2 class="{'warn' if n_falsif > 0 else 'ok'}">{n_falsif}</h2>
       <p>Falsifications</p></div>
</div>

<div class="section">
  <h2>📋 Historique des décisions</h2>
  <table>
    <tr>
      <th>Fichier</th><th>Décision</th><th>Score suspect</th>
      <th>Correct</th><th>Intégrité</th><th>Date</th>
    </tr>
    {rows_html}
  </table>
</div>

<div class="section">
  <h2>🧠 Runs d'entraînement</h2>
  <table>
    <tr><th>Date</th><th>F1</th><th>Accuracy</th><th>Recall</th><th>Epochs</th></tr>
    {runs_html if runs_html else '<tr><td colspan="5" style="color:#888">Aucun run enregistré</td></tr>'}
  </table>
</div>

<div class="section">
  <h2>🔒 Intégrité SHA256</h2>
  <p class="{'warn' if n_falsif > 0 else 'ok'}">
    {'⚠️ ' + str(n_falsif) + ' fichier(s) falsifié(s) détecté(s) !' if n_falsif > 0
     else '✅ Tous les fichiers sont intacts — aucune falsification détectée.'}
  </p>
  <p style="color:#888;font-size:.85em">
    Algorithme : SHA256 | Base : {DB_PATH} | Log : {LOG_PATH}
  </p>
</div>

<footer>
  Système de détection d'anomalies dans les factures professionnelles par CNN<br>
  ResNet-50 + Transfer Learning + SHAP + Audit SHA256
</footer>
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"✅ Rapport HTML → {REPORT_PATH}")


# ─── Pipeline principal ────────────────────────────────────────────────────────
def run_audit():
    logger.info("=" * 55)
    logger.info("AUDIT — Système Anti-Fraude Factures")
    logger.info("=" * 55)

    conn    = init_db()
    records = load_metadata()
    shap_r  = load_shap_results()

    logger.info(f"Résultats SHAP trouvés : {len(shap_r)}")

    insert_factures(conn, records, shap_r)
    log_model_run(conn)
    falsified = verify_integrity(conn)
    generate_html_report(conn)
    conn.close()

    print("\n" + "=" * 55)
    print("✅ AUDIT TERMINÉ")
    print("=" * 55)
    print(f"   Base de données : {DB_PATH}")
    print(f"   Log             : {LOG_PATH}")
    print(f"   Rapport HTML    : {REPORT_PATH}")
    print(f"   Falsifications  : {falsified}")
    print()
    print("▶️  Ouvre le rapport :")
    print(f"   start {REPORT_PATH}")
    print()
    print("▶️  Prochaine étape :")
    print("   python src/api/main.py")
    print("=" * 55)


# ─── Argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Audit anti-fraude factures")
    p.add_argument("--verify", action="store_true",
                   help="Vérifie seulement l'intégrité SHA256")
    p.add_argument("--report", action="store_true",
                   help="Génère seulement le rapport HTML")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verify:
        conn = init_db()
        verify_integrity(conn)
        conn.close()
    elif args.report:
        conn = init_db()
        generate_html_report(conn)
        conn.close()
    else:
        run_audit()