from pathlib import Path

import pandas as pd
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
INPUT_DIR = Path("data/images")
OUTPUT_DIR = Path("outputs")
RESULTS_CSV = OUTPUT_DIR / "dog_detections.csv"

DOG_CLASS_NAME = "dog"
CONFIDENCE_THRESHOLD = 0.25


def find_images(folder: Path) -> list[Path]:
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    return [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in image_extensions
    ]


def detect_dogs(model: YOLO, image_path: Path) -> list[dict]:
    results = model(
        str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        project=str(OUTPUT_DIR),
        name="detections",
        exist_ok=True,
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name != DOG_CLASS_NAME:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "image": image_path.name,
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            )

    return detections


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    image_paths = find_images(INPUT_DIR)

    if not image_paths:
        print("Nema slika u data/images folderu.")
        return

    model = YOLO(MODEL_PATH)

    all_detections = []

    for image_path in image_paths:
        print(f"Obrađujem sliku: {image_path.name}")

        dog_detections = detect_dogs(model, image_path)
        all_detections.extend(dog_detections)

        print(f"Broj pronađenih pasa: {len(dog_detections)}")

    df = pd.DataFrame(all_detections)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    print("Detekcija završena.")
    print(f"CSV rezultati su sačuvani u: {RESULTS_CSV}")
    print("Slike sa bounding box-ovima su u: outputs/detections/")


if __name__ == "__main__":
    main()