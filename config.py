from dataclasses import dataclass


@dataclass
class Config:
    # Podaci
    data_yaml: str = "roboflow_dataset/data.yaml"

    # YOLO arhitektura
    model_yaml: str = "yolov8n.yaml"
    # Koristi se samo ako je train_from_scratch=False.
    pretrained_weights: str = "yolov8n.pt"
    train_from_scratch: bool = True

    # Trening
    epochs: int = 50
    imgsz: int = 512
    batch: int = 8
    patience: int = 10
    seed: int = 42

    # Validacija i evaluacija
    val_split: str = "val"
    test_split: str = "test"

    # Cuvanje rezultata
    project: str = "runs"
    name: str = "marija_masa_model"
    exist_ok: bool = True
