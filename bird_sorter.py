import os
import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

# =========================
# CONFIG
# =========================

PHOTO_DIR = Path("./data_to_class")

# seuil minimal de confiance
CONFIDENCE_THRESHOLD = 0.35

# extensions JPEG
JPEG_EXTENSIONS = [".jpg", ".jpeg", ".JPG", ".JPEG"]

# extensions RAW
RAW_EXTENSIONS = [
    ".cr2", ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".rw2",
    ".orf",
    ".dng"
]

# =========================
# MODEL
# =========================

print("Chargement du modèle...")

classifier = pipeline(
    "image-classification",
    model="chriamue/bird-species-classifier",
    device=0 if torch.cuda.is_available() else -1
)

print("Modèle chargé.")

# =========================
# FUNCTIONS
# =========================

def clean_folder_name(name):
    """
    Nettoie le nom pour créer un dossier propre.
    """
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    return name

def find_raw_files(base_path):
    """
    Cherche les RAW correspondant au JPEG.
    """
    raw_files = []

    for ext in RAW_EXTENSIONS:
        raw_candidate = base_path.with_suffix(ext)
        if raw_candidate.exists():
            raw_files.append(raw_candidate)

        raw_candidate_upper = base_path.with_suffix(ext.upper())
        if raw_candidate_upper.exists():
            raw_files.append(raw_candidate_upper)

    return raw_files

# =========================
# MAIN
# =========================

photo_dir = Path(PHOTO_DIR)

jpeg_files = []

for ext in JPEG_EXTENSIONS:
    jpeg_files.extend(photo_dir.glob(f"*{ext}"))

print(f"{len(jpeg_files)} JPEG trouvés.")

for jpeg_path in tqdm(jpeg_files):

    try:
        # =========================
        # FIND RAW FILES FIRST
        # =========================

        base_path = jpeg_path.with_suffix("")

        raw_files = find_raw_files(base_path)

        # =========================
        # CLASSIFICATION
        # =========================

        image = Image.open(jpeg_path).convert("RGB")

        results = classifier(image)

        best = results[0]

        label = best["label"]
        score = best["score"]

        if score < CONFIDENCE_THRESHOLD:
            species_name = "Uncertain"
        else:
            species_name = clean_folder_name(label)

        target_dir = photo_dir / species_name

        target_dir.mkdir(exist_ok=True)

        # =========================
        # MOVE JPEG
        # =========================

        target_jpeg = target_dir / jpeg_path.name

        shutil.move(str(jpeg_path), str(target_jpeg))

        # =========================
        # MOVE RAW FILES
        # =========================

        for raw_file in raw_files:

            if raw_file.exists():

                target_raw = target_dir / raw_file.name

                shutil.move(str(raw_file), str(target_raw))

    except Exception as e:
        print(f"Erreur avec {jpeg_path.name}: {e}")

print("Tri terminé.")