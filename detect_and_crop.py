from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = Path("data_to_class")
OUTPUT_DIR = Path("cropped_birds")

CONFIDENCE = 0.25
PADDING = 0.15

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD YOLO
# =========================

print("Chargement YOLO...")

model = YOLO("yolov8x.pt")

print("YOLO chargé.")

# =========================
# FILES
# =========================

jpeg_files = []
jpeg_files += list(INPUT_DIR.glob("*.jpg"))
jpeg_files += list(INPUT_DIR.glob("*.jpeg"))
jpeg_files += list(INPUT_DIR.glob("*.JPG"))
jpeg_files += list(INPUT_DIR.glob("*.JPEG"))

jpeg_files = list(set(jpeg_files))

print(f"{len(jpeg_files)} images trouvées.")

# =========================
# PROCESS
# =========================

for image_path in tqdm(jpeg_files):

    try:

        results = model(
            str(image_path),
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        best_area = 0
        best_box = None

        # Classe COCO bird = 14
        for box in boxes:

            cls = int(box.cls[0])

            if cls != 14:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            continue

        x1, y1, x2, y2 = best_box

        # =========================
        # PADDING
        # =========================

        pad_x = (x2 - x1) * PADDING
        pad_y = (y2 - y1) * PADDING

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))

        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        # =========================
        # CROP
        # =========================

        cropped = image.crop((x1, y1, x2, y2))

        output_path = OUTPUT_DIR / image_path.name

        cropped.save(output_path, quality=95)

    except Exception as e:
        print(f"Erreur {image_path.name}: {e}")

print("Cropping terminé.")