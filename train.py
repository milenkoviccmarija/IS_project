from pathlib import Path

import torch

from config import Config
from model import create_model, load_model


PROJECT_DIR = Path(__file__).resolve().parent


def resolve_project_path(path: str) -> str:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return str(path_obj)

    return str(PROJECT_DIR / path_obj)


def resolve_model_path(path: str) -> str:
    local_path = PROJECT_DIR / path

    if local_path.exists():
        return str(local_path)

    return path


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def train_yolo(config: Config, device: str):
    model_path = config.model_yaml if config.train_from_scratch else config.pretrained_weights
    model = create_model(resolve_model_path(model_path))

    return model.train(
        data=resolve_project_path(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=device,
        pretrained=not config.train_from_scratch,
        patience=config.patience,
        seed=config.seed,
        project=resolve_project_path(config.project),
        name=config.name,
        exist_ok=config.exist_ok,
    )


def validate_yolo(config: Config, weights_path: Path, device: str):
    model = load_model(weights_path)

    return model.val(
        data=resolve_project_path(config.data_yaml),
        split=config.val_split,
        imgsz=config.imgsz,
        batch=config.batch,
        device=device,
        project=resolve_project_path(config.project),
        name=f"{config.name}_validation",
        exist_ok=config.exist_ok,
    )


def print_validation_metrics(metrics) -> None:
    print("Validacija zavrsena.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


def main() -> None:
    config = Config()
    device = resolve_device()
    model_description = "prazna YOLO arhitektura" if config.train_from_scratch else "pretrained YOLO model"

    print(f"Uredjaj: {device}")
    print(f"Model: {config.model_yaml} ({model_description})")
    print(f"Dataset: {config.data_yaml}")
    print("Pokrecem treniranje od nule...")

    results = train_yolo(config, device)

    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    print(f"Trening zavrsen. Rezultati su sacuvani u: {save_dir}")
    print(f"Najbolji model: {best_model}")
    print(f"Poslednji model: {last_model}")
    print("Pokrecem validaciju najboljeg modela...")

    metrics = validate_yolo(config, best_model, device)
    print_validation_metrics(metrics)


if __name__ == "__main__":
    main()
