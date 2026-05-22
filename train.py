from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.yaml")

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        project="outputs",
        name="dog_training",
    )


if __name__ == "__main__":
    main()