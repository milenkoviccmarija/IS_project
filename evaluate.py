from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from dataset import load_dataset_config, split_image_dir
from model import load_model


PROJECT_DIR = Path(__file__).resolve().parent


def resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return PROJECT_DIR / path_obj


def best_model_path(config: Config) -> Path:
    return resolve_project_path(config.project) / config.name / "weights" / "best.pt"


def run_evaluation(config: Config, split: str | None = None):

    model = load_model(best_model_path(config))

    return model.val(
        data=str(resolve_project_path(config.data_yaml)),
        split=split or config.test_split,
        imgsz=config.imgsz,
        batch=config.batch,
        project=str(resolve_project_path(config.project)),
        name=f"{config.name}_evaluation",
        exist_ok=config.exist_ok,
    )


def print_detection_metrics(metrics) -> None:
    print("Evaluacija zavrsena.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


def load_training_history(config: Config) -> pd.DataFrame:
    results_csv = resolve_project_path(config.project) / config.name / "results.csv"
    return pd.read_csv(results_csv)


def plot_training_curves(history: pd.DataFrame, title: str = "") -> plt.Figure:
    """
    Crta YOLO loss i mAP krive kroz epohe.
    """
    history = history.rename(columns=lambda column: column.strip())
    epochs = history["epoch"] + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train/box_loss"], label="train box")
    ax1.plot(epochs, history["val/box_loss"], label="val box")
    ax1.plot(epochs, history["train/cls_loss"], label="train cls")
    ax1.plot(epochs, history["val/cls_loss"], label="val cls")
    ax1.set_xlabel("Epoha")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss{' - ' + title if title else ''}")
    ax1.legend(fontsize=8)

    ax2.plot(epochs, history["metrics/mAP50(B)"], label="mAP50")
    ax2.plot(epochs, history["metrics/mAP50-95(B)"], label="mAP50-95")
    ax2.plot(epochs, history["metrics/precision(B)"], label="precision")
    ax2.plot(epochs, history["metrics/recall(B)"], label="recall")
    ax2.set_xlabel("Epoha")
    ax2.set_ylabel("Vrednost")
    ax2.set_title(f"Metrike{' - ' + title if title else ''}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(config: Config) -> plt.Figure:

    image_path = resolve_project_path(config.project) / config.name / "confusion_matrix.png"
    image = plt.imread(image_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Matrica konfuzije")
    plt.tight_layout()
    return fig


def plot_predictions(
    config: Config,
    split: str | None = None,
    n: int = 6,
    seed: int = 0,
) -> plt.Figure:
    """
    Prikazuje nekoliko slika sa YOLO predikcijama.
    """
    dataset_config = load_dataset_config(config.data_yaml)
    image_dir = split_image_dir(dataset_config, split or config.test_split, config.data_yaml)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    rng = np.random.default_rng(seed)
    selected = rng.choice(image_paths, size=min(n, len(image_paths)), replace=False)

    model = load_model(best_model_path(config))
    results = model.predict(
        source=[str(path) for path in selected],
        imgsz=config.imgsz,
        conf=0.25,
        verbose=False,
    )

    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    for ax, result in zip(axes, results):
        ax.imshow(result.plot()[..., ::-1])
        ax.axis("off")
        ax.set_title(Path(result.path).name, fontsize=9)

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def main() -> None:
    config = Config()
    metrics = run_evaluation(config)
    print_detection_metrics(metrics)


if __name__ == "__main__":
    main()
