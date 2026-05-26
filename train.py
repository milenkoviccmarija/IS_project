from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="roboflow_dataset/data.yaml",
    epochs=50,
    imgsz=512,
    project="runs",
    name="marija_masa_model",
    exist_ok=True
)