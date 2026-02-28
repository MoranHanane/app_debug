# app/app/utils.py
# isole le prétraitement d'image du reste du code + le rend testable indépendamment


import numpy as np
from PIL import Image

def preprocess_from_pil(pil_img: Image.Image, size=(224, 224)) -> np.ndarray:
    img = pil_img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

