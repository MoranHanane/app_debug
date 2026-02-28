# tests/test_app.py
import os, io, sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import random

os.environ['KERAS_BACKEND'] = "torch"

import keras

# # --- permet de rendre importable app/app/utils.py ---
# REPO_ROOT = Path(__file__).resolve().parents[1]     # .../app
# MODULE_DIR = REPO_ROOT / "app"                      # .../app/app
# sys.path.insert(0, str(MODULE_DIR))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import preprocess_from_pil
from appV2 import app as flask_app


# # Flask client pour tester les routes
# @pytest.fixture()    #--> l’environnement créé par la fixture sera constant et identique pour chaque test qui l'appellera
# def client():        # --> la fixture client() évite de répéter la création/suppression du client Flask dans les tests suivants et met l’app en mode TESTING=True
#     app.config.update(TESTING=True)
#     with app.test_client() as c:
#         yield c


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# ---------- TEST DE PRETRAITEMENT ----------

def test_preprocess_shape():
    """Doit redimensionner en 224x224, dtype float32, valeurs dans [0,1]."""
    deformate = Image.new("RGB", (600, 600), "red")       #création d'une nouvelle image aux paramètre spécifiés
    x = preprocess_from_pil(deformate, )
    assert x.shape == (1, 224, 224, 3)              # forme lot(1) x H x W x C
    assert x.dtype == np.float32                    # normalisation float32


def test_range_data():
    deformate = Image.new("RGB", (600, 600), "red")       #création d'une nouvelle image aux paramètre spécifiés
    x = preprocess_from_pil(deformate)
    m, M = float(x.min()), float(x.max())
    assert 0.0 <= m <= 1.0 and 0.0 <= M <= 1.0


# ---------- TESTS DE ROUTES  ----------

def test_index_ok(client):
    """GET / est sensé répondre 200 (page d'accueil)."""
    r = client.get("/")
    assert r.status_code == 200


# def test_preprocess_resizes_and_normalizes():
#     # 2 tailles dont une aléatoire pour couvrir des cas variés
#     for w, h in [(600, 600), (random.randint(50, 900), random.randint(50, 900))]:
#         arr = preprocess_from_pil(Image.new("RGB", (w, h), "red"))
#         assert arr.shape == (1, 224, 224, 3)
#         assert arr.dtype == np.float32
#         mn, mx = float(arr.min()), float(arr.max())
#         assert 0.0 <= mn <= 1.0 and 0.0 <= mx <= 1.0


