from dataclasses import dataclass

@dataclass
class Config:
    # Podaci
    data_yaml: str = "roboflow_dataset/data.yaml"

    # YOLO arhitektura
    model_yaml: str = "yolov8n.yaml"
    pretrained_weights: str = "yolov8n.pt"
    train_from_scratch: bool = True

    # Trening
    epochs: int = 200
    imgsz: int = 640
    batch: int = 8
    patience: int = 40
    seed: int = 42
    save_period: int = -1
    resume_training: bool = False
    resume_weights: str = "runs/marija_masa_model/weights/last.pt"

    # Augmentacije tokom treninga
    hsv_h: float = 0.015
    hsv_s: float = 0.6
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.05
    scale: float = 0.20
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.5
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: int = 15

    # Validacija i evaluacija
    val_split: str = "val"
    test_split: str = "test"
    error_conf_threshold: float = 0.25
    error_iou_threshold: float = 0.5
    track_epoch_errors: bool = False
    save_validation_comparisons: bool = True
    save_epoch_validation_snapshots: bool = True
    epoch_validation_snapshot_images: int | None = None

    # Čuvanje rezultata
    project: str = "runs"
    name: str = "marija_masa_model"
    exist_ok: bool = True