import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

# =========================
# CONFIG
# =========================

ORIGINAL_DIR = Path("data_to_class")
CROPPED_DIR = Path("cropped_birds")

CONFIDENCE_THRESHOLD = 0.35

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
    #model="chriamue/bird-species-classifier",
    #model="birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio",
    model="facebook/dinov2-small-imagenet1k-1-layer",
    device=0 if torch.cuda.is_available() else -1,
    batch_size=8
)

print("Modèle chargé.")

# =========================
# FUNCTIONS
# =========================

def clean_folder_name(name):
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    return name

def find_raw_files(base_path):

    raw_files = []

    for ext in RAW_EXTENSIONS:

        candidate1 = base_path.with_suffix(ext)
        candidate2 = base_path.with_suffix(ext.upper())

        if candidate1.exists():
            raw_files.append(candidate1)

        if candidate2.exists():
            raw_files.append(candidate2)

    return raw_files

# =========================
# FILES
# =========================

jpeg_files = []
jpeg_files += list(CROPPED_DIR.glob("*.jpg"))
jpeg_files += list(CROPPED_DIR.glob("*.jpeg"))
jpeg_files += list(CROPPED_DIR.glob("*.JPG"))
jpeg_files += list(CROPPED_DIR.glob("*.JPEG"))

jpeg_files = list(set(jpeg_files))

print(f"{len(jpeg_files)} crops trouvés.")

# =========================
# MAIN
# =========================

for crop_path in tqdm(jpeg_files):

    try:

        # =========================
        # CLASSIFICATION
        # =========================

        image = Image.open(crop_path).convert("RGB")

        results = classifier(image)

        best = results[0]

        label = best["label"]
        score = best["score"]

        if score < CONFIDENCE_THRESHOLD:
            species_name = "Uncertain"
        else:
            species_name = clean_folder_name(label)

        # =========================
        # TARGET DIR
        # =========================

        target_dir = ORIGINAL_DIR / species_name

        target_dir.mkdir(exist_ok=True)

        # =========================
        # ORIGINAL JPG
        # =========================

        original_jpg = ORIGINAL_DIR / crop_path.name

        if original_jpg.exists():

            target_jpg = target_dir / original_jpg.name

            shutil.move(str(original_jpg), str(target_jpg))

        # =========================
        # RAW FILES
        # =========================

        base_path = ORIGINAL_DIR / crop_path.stem

        raw_files = find_raw_files(base_path)

        for raw_file in raw_files:

            target_raw = target_dir / raw_file.name

            shutil.move(str(raw_file), str(target_raw))

    except Exception as e:
        print(f"Erreur avec {crop_path.name}: {e}")

print("Tri terminé.")