from ultralytics import YOLO
import time

DATA_PATH = "roboflow_dataset/data.yaml"

model = YOLO("yolov8n.pt")

start = time.time()

results = model.train(
    data=DATA_PATH,
    epochs=50,
    imgsz=512,
    batch=8,
    project="runs",
    name="marija_masa_model",
    exist_ok=True
)

end = time.time()

print(f"Trening završen za {(end - start) / 60:.2f} minuta.")
print("Najbolji model je u:")
print("runs/marija_masa_model/weights/best.pt")