from ultralytics import YOLO

MODEL_PATH = "runs/marija_masa_model/weights/best.pt"
DATA_PATH = "roboflow_dataset/data.yaml"

model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA_PATH,
    split="test",
    imgsz=512
)

print("Evaluacija završena.")
print(metrics)