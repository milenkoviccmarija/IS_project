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
    save_period: int = 1

    # Validacija i evaluacija
    val_split: str = "val"
    test_split: str = "test"
    error_conf_threshold: float = 0.25
    error_iou_threshold: float = 0.5
    track_epoch_errors: bool = True

    # Cuvanje rezultata
    project: str = "runs"
    name: str = "marija_masa_model"
    exist_ok: bool = True
