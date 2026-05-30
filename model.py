from pathlib import Path

from ultralytics import YOLO


def create_model(model_yaml: str = "yolov8n.yaml") -> YOLO:
    return YOLO(model_yaml)


def load_model(weights_path: str | Path) -> YOLO:
    """
    Ucitava istrenirani YOLO model iz .pt fajla.
    """
    return YOLO(str(weights_path))
