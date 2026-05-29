import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitStats:
    name: str
    images: int
    labels: int
    objects: int
    class_counts: Counter[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return PROJECT_DIR / path_obj


def load_dataset_config(data_yaml: str | Path = "roboflow_dataset/data.yaml") -> dict:
    with resolve_path(data_yaml).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_class_names(config: dict) -> list[str]:
    names = config["names"]

    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]

    return list(names)


def get_dataset_root(config: dict, data_yaml: str | Path = "roboflow_dataset/data.yaml") -> Path:
    dataset_path = Path(config["path"])

    if dataset_path.is_absolute():
        return dataset_path

    yaml_dir = resolve_path(data_yaml).parent

    if yaml_dir.name == dataset_path.name:
        return yaml_dir

    return PROJECT_DIR / dataset_path


def split_image_dir(
    config: dict,
    split: str,
    data_yaml: str | Path = "roboflow_dataset/data.yaml",
) -> Path:
    return get_dataset_root(config, data_yaml) / config[split]


def split_label_dir(
    config: dict,
    split: str,
    data_yaml: str | Path = "roboflow_dataset/data.yaml",
) -> Path:
    image_dir = split_image_dir(config, split, data_yaml)
    return image_dir.parent / "labels"


def list_images(image_dir: str | Path) -> list[Path]:
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_labels(label_dir: str | Path) -> list[Path]:
    return sorted(path for path in Path(label_dir).glob("*.txt") if path.is_file())


def class_distribution(label_dir: str | Path) -> Counter[int]:
    counts: Counter[int] = Counter()

    for label_file in list_labels(label_dir):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            counts[class_id] += 1

    return counts


def split_stats(
    config: dict,
    split: str,
    data_yaml: str | Path = "roboflow_dataset/data.yaml",
) -> SplitStats:
    image_dir = split_image_dir(config, split, data_yaml)
    label_dir = split_label_dir(config, split, data_yaml)
    counts = class_distribution(label_dir)

    return SplitStats(
        name=split,
        images=len(list_images(image_dir)),
        labels=len(list_labels(label_dir)),
        objects=sum(counts.values()),
        class_counts=counts,
    )


def dataset_summary(data_yaml: str | Path = "roboflow_dataset/data.yaml") -> dict[str, SplitStats]:
    config = load_dataset_config(data_yaml)
    return {
        split: split_stats(config, split, data_yaml)
        for split in ("train", "val", "test")
        if split in config
    }


def majority_class_accuracy(label_dir: str | Path) -> float:
    counts = class_distribution(label_dir)

    if not counts:
        return 0.0

    return max(counts.values()) / sum(counts.values())


def print_dataset_summary(data_yaml: str | Path = "roboflow_dataset/data.yaml") -> None:
    config = load_dataset_config(data_yaml)
    class_names = get_class_names(config)

    print(f"Dataset: {resolve_path(data_yaml)}")
    print(f"Klase: {', '.join(class_names)}")

    for stats in dataset_summary(data_yaml).values():
        print(
            f"{stats.name}: "
            f"slike={stats.images}, "
            f"label fajlovi={stats.labels}, "
            f"objekti={stats.objects}, "
            f"raspodela={dict(stats.class_counts)}"
        )


def plot_class_distribution(
    label_dir: str | Path,
    class_names: list[str],
    title: str = "Raspodela klasa",
) -> plt.Figure:
    counts = class_distribution(label_dir)
    classes = range(len(class_names))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(
        [class_names[class_id] for class_id in classes],
        [counts[class_id] for class_id in classes],
        color="steelblue",
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Klasa")
    ax.set_ylabel("Broj objekata")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_sample_images(
    image_dir: str | Path,
    n: int = 6,
    title: str = "Uzorci iz dataseta",
) -> plt.Figure:
    images = list_images(image_dir)[:n]
    columns = max(1, len(images))

    fig, axes = plt.subplots(1, columns, figsize=(columns * 2.2, 2.6))
    axes = np.atleast_1d(axes)

    for ax, image_path in zip(axes, images):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(image_path.stem[:18], fontsize=8)
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print_dataset_summary()
