from ultralytics import YOLO

#MODEL_PATH = "outputs/dog_training/weights/best.pt"
MODEL_PATH = "yolov8n.pt"


def main() -> None:

    model = YOLO(MODEL_PATH)

    results = model(
        source="data/images/test",
        save=True,
        conf=0.25,
    )

    print("Inferencija završena.")


if __name__ == "__main__":
    main()