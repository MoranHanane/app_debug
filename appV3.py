import os
os.environ["KERAS_BACKEND"] = "torch"

import io
import base64
import time
import csv
from datetime import datetime

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

import numpy as np
import keras
import flask_monitoringdashboard as dashboard
from flask_monitoringdashboard import config

import logging
from logging.handlers import SMTPHandler, RotatingFileHandler

from PIL import Image
from utils import preprocess_from_pil
from metrics import log_request_time, log_prediction, compute_metrics

import sys
print("PYTHONPATH =", sys.path)

# ---------------- Config ----------------
from dotenv import load_dotenv
load_dotenv()  # charge .env dans les variables d'environnement

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
CLASSES = ['desert', 'forest', 'meadow', 'mountain']

app = Flask(__name__)

# ---------------- Monitoring ----------------
app.secret_key = os.getenv("FLASK_SECRET_KEY")

dashboard.config.init_from(os.path.join(os.path.dirname(__file__), "Dashboard.cfg"))
#dashboard.config.init_from(file=r'DATABASE=sqlite:///C:\\Users\\Utilisateur\\Documents\\Simplon_avant_crash\\Python_notebooks\\Bertrand\\application_a_debugger\\app\\app\\flask_monitoringdashboard.db')
dashboard.bind(app)


#  Journalisation (RotatingFileHandler) + alerting (SMTPHandler)
#----------------------------------------------------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # appV3.py

log_dir = os.path.join(BASE_DIR, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

handler = RotatingFileHandler(
    os.path.join(log_dir, "app.log"),
    maxBytes=1_000_000,
    backupCount=3
)
handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


smtp_handler = SMTPHandler(
    mailhost=(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT"))),
    fromaddr=os.getenv("SMTP_FROM"),
    toaddrs=[os.getenv("SMTP_TO")],
    subject="⚠ Alerte Monitorage IA",
    credentials=(
        os.getenv("SMTP_USERNAME"),
        os.getenv("SMTP_PASSWORD")
    ),
    secure=()
)

smtp_handler.setLevel(logging.ERROR)
app.logger.addHandler(smtp_handler)


# ---------------- Feedbacks utilisateurs ----------

def save_feedback(image: str, cls: str, conf: float):
    """Enregistre un feedback utilisateur dans un fichier CSV.

    Chaque ligne contient :
      - la date et l'heure du feedback
      - le nom de l'image
      - la classe prédite
      - le score de confiance

    Args:
        image: Nom ou identifiant de l'image soumise.
        cls: Classe prédite retournée à l'utilisateur.
        conf: Score de confiance associé à la prédiction.
    """
    with open("feedback.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), image, cls, conf])


# ---------------- Model ----------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_cnn.keras"

try:
    model = keras.saving.load_model(str(MODEL_PATH), compile=False)
    app.logger.info(f"Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    app.logger.error("Échec du chargement du modèle : %s", e)
    model = None



# ---------------- Utils ----------------

def allowed_file(filename: str) -> bool:
    """Vérifie si le nom de fichier possède une extension autorisée.

    La vérification est insensible à la casse et regarde uniquement
    la sous-chaîne après le dernier point.

    Args:
        filename: Nom du fichier soumis (par exemple "photo.PNG").

    Returns:
        True si l’extension (par exemple "png", "jpg") est dans ALLOWED_EXT,
        sinon False.

    Examples:
        >>> allowed_file("img.JPG")
        True
        >>> allowed_file("archive.tar.gz")
        False
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_data_url(pil_img: Image.Image, fmt="JPEG") -> str:
    """Convertit une image PIL en Data URL base64 affichable dans un <img src="...">.

    L’image est encodée en mémoire (sans I/O disque), sérialisée en base64,
    puis encapsulée comme `data:<mime>;base64,<payload>`. Le type MIME est
    déduit de `fmt`.

    Args:
        pil_img: Image PIL à encoder.
        fmt: Format d’encodage PIL (ex. "JPEG", "PNG"). Par défaut "JPEG".

    Returns:
        Chaîne Data URL prête à être insérée dans une balise <img>.

    Raises:
        ValueError: si la sauvegarde PIL échoue pour le format demandé.

    Examples:
        >>> url = to_data_url(Image.new("RGB", (10, 10), "red"), fmt="PNG")
        >>> url.startswith("data:image/png;base64,")
        True
    """
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


# ---------------- Routes ----------------


@app.route("/", methods=["GET"])
def index():
    """Affiche la page d’upload.

    Returns:
        Réponse HTML rendant le template "upload.html".
    """
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Traite l’upload d’une image, exécute la prédiction, met à jour
    les métriques définies dans metrics.py, et affiche le résultat.

    Le workflow est le suivant :
      1) Validation de présence et d’extension du fichier.
      2) Lecture du contenu en mémoire et ouverture en PIL.
      3) Prétraitement avec preprocess_from_pil.
      4) Prédiction avec le modèle ou fallback.
      5) Mise à jour des métriques de latence et confiance.
      6) Vérification et génération d’une alerte si nécessaire.
      7) Encodage en Data URL et rendu du template "result.html".

    Redirects:
        - Redirige vers "/" si le fichier est manquant ou invalide.

    Returns:
        Réponse HTML rendant "result.html" avec :
        - `image_data_url`: image encodée en base64
        - `predicted_label`: classe prédite sous forme de chaîne de caractères
        - `confidence`: score de confiance (float)
        - `classes`: la liste complète des classes possibles
    """
    if "file" not in request.files:
        return redirect("/")

    file = request.files["file"]
    if file.filename == "" or not allowed_file(secure_filename(file.filename)):
        return redirect("/")

    raw = file.read()
    pil_img = Image.open(io.BytesIO(raw))

    # Début chronométrage pour métriques de latence
    start_time = time.time()

    img_array = preprocess_from_pil(pil_img)

    if model is None:
        # Fallback lorsque le modèle n’est pas chargé : plan de secours
        cls_idx = 0
        label = CLASSES[cls_idx]
        conf = 0.34
    else:
        # Prédiction normale avec le modèle
        probs = model.predict(img_array, verbose=0)[0]
        cls_idx = int(np.argmax(probs))
        label = CLASSES[cls_idx]
        conf = float(probs[cls_idx])

    # Fin de mesure de latence
    duration = time.time() - start_time
    log_request_time(duration)
    log_prediction(label, conf)

    metrics = compute_metrics()
    # ---- Alerte score moyen ----

    if metrics["avg_conf"] < 0.80:
        app.logger.error(
            "Score moyen en dessous du seuil critique: %.3f",
            metrics["avg_conf"]
        )

    # ---- Alerte error rate (latence > 1s) ----
    if metrics["error_rate"] > 0.30:
        app.logger.error(
            "Taux d'erreur (latence >1s) critique: %.2f%%",
            metrics["error_rate"] * 100
        )

    # ---- Alerte déséquilibre de classes ----
    if metrics["total_predictions"] >= 10 and metrics["class_distribution"]:
        max_class_count = max(metrics["class_distribution"].values())
        dominance_ratio = max_class_count / metrics["total_predictions"]

        if dominance_ratio >= 0.50:         #alerte si>= 50% d'une classe domine
            app.logger.error(
                "Distribution anormale détectée: %.2f%% d'une même classe sur %d prédictions",
                dominance_ratio * 100,
                metrics["total_predictions"]
            )

    image_data_url = to_data_url(pil_img, fmt="JPEG")

    return render_template(
        "result.html",
        image_data_url=image_data_url,
        predicted_label=label,
        confidence=conf,
        classes=CLASSES,
    )


@app.route("/feedback", methods=["POST"])
def post_feedback():
    """Traite l’envoi d’un feedback depuis le formulaire."""
    image = request.form.get("image_name")
    label = request.form.get("predicted_label")
    conf = request.form.get("confidence")
    fb = request.form.get("feedback")  # correct ou incorrect

    # Enregistrement dans le CSV (date, image, label, confidence, feedback)
    with open("feedback.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), image, label, conf, fb])

    return render_template("feedback_ok.html", feedback=fb)


# ---------------- Exécution ----------------

if __name__ == "__main__":
    app.run(debug=False)





