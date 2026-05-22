from ultralytics import YOLO
from pathlib import Path


def main():

    model = YOLO("yolov8n.pt")

    image_folder = Path("data/images")

    image_paths = list(image_folder.glob("*"))

    if len(image_paths) == 0:
        print("Nema slika u data/images folderu.")
        return

    for image_path in image_paths:

        print(f"Obrađujem: {image_path.name}")

        results = model(
            str(image_path),
            save=True,
            project="outputs",
            name="detections"
        )

    print("Detekcija završena.")


if __name__ == "__main__":
    main()