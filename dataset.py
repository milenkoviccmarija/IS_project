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


@dataclass
class BBoxRecord:
    split: str
    image_name: str
    class_id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


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
    yaml_dir = resolve_path(data_yaml).parent
    dataset_path = Path(config.get("path", yaml_dir.name))

    if dataset_path.is_absolute():
        return dataset_path

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


def parse_label_file(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []

    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id, x, y, width, height = line.split()
        boxes.append((int(float(class_id)), float(x), float(y), float(width), float(height)))

    return boxes


def collect_bbox_records(
    config: dict,
    data_yaml: str | Path = "roboflow_dataset/data.yaml",
) -> list[BBoxRecord]:
    records = []

    for split in ("train", "val", "test"):
        if split not in config:
            continue

        image_dir = split_image_dir(config, split, data_yaml)
        label_dir = split_label_dir(config, split, data_yaml)

        for image_path in list_images(image_dir):
            label_file = label_dir / f"{image_path.stem}.txt"
            if not label_file.exists():
                continue

            for class_id, x, y, width, height in parse_label_file(label_file):
                records.append(
                    BBoxRecord(
                        split=split,
                        image_name=image_path.name,
                        class_id=class_id,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    )
                )

    return records


def bbox_size_category(area: float) -> str:
    if area < 0.01:
        return "tiny <1%"
    if area < 0.05:
        return "small 1-5%"
    if area < 0.20:
        return "medium 5-20%"
    return "large >20%"


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


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_objects_by_split(summary: dict[str, SplitStats]) -> plt.Figure:
    splits = list(summary)
    images = [summary[split].images for split in splits]
    objects = [summary[split].objects for split in splits]

    fig, ax = plt.subplots(figsize=(7, 4))
    x_positions = np.arange(len(splits))
    width = 0.36
    image_bars = ax.bar(x_positions - width / 2, images, width, label="Slike", color="#4C78A8")
    object_bars = ax.bar(x_positions + width / 2, objects, width, label="Objekti", color="#F58518")

    ax.bar_label(image_bars, padding=3, fontsize=8)
    ax.bar_label(object_bars, padding=3, fontsize=8)
    ax.set_xticks(x_positions, splits)
    ax.set_ylabel("Broj")
    ax.set_title("Broj slika i objekata po splitu")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_class_distribution_by_split(
    summary: dict[str, SplitStats],
    class_names: list[str],
) -> plt.Figure:
    splits = list(summary)
    x_positions = np.arange(len(splits))
    width = 0.8 / max(1, len(class_names))

    fig, ax = plt.subplots(figsize=(7, 4))
    for class_id, class_name in enumerate(class_names):
        values = [summary[split].class_counts[class_id] for split in splits]
        offset = (class_id - (len(class_names) - 1) / 2) * width
        bars = ax.bar(x_positions + offset, values, width, label=class_name)
        ax.bar_label(bars, padding=3, fontsize=8)

    ax.set_xticks(x_positions, splits)
    ax.set_ylabel("Broj objekata")
    ax.set_title("Raspodela klasa po splitu")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_bbox_area_histogram(records: list[BBoxRecord]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for split in ("train", "val", "test"):
        values = [record.area * 100 for record in records if record.split == split]
        if values:
            ax.hist(values, bins=20, alpha=0.55, label=split)

    ax.set_xlabel("Povrsina bbox-a (% slike)")
    ax.set_ylabel("Broj bbox-eva")
    ax.set_title("Distribucija povrsine bounding box-eva")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_bbox_width_height(records: list[BBoxRecord]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for split in ("train", "val", "test"):
        widths = [record.width * 100 for record in records if record.split == split]
        heights = [record.height * 100 for record in records if record.split == split]
        if widths:
            axes[0].hist(widths, bins=20, alpha=0.55, label=split)
        if heights:
            axes[1].hist(heights, bins=20, alpha=0.55, label=split)

    axes[0].set_title("Sirina bbox-a")
    axes[0].set_xlabel("Sirina (% slike)")
    axes[0].set_ylabel("Broj bbox-eva")
    axes[1].set_title("Visina bbox-a")
    axes[1].set_xlabel("Visina (% slike)")

    for ax in axes:
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    return fig


def plot_bbox_size_categories(records: list[BBoxRecord]) -> plt.Figure:
    categories = ["tiny <1%", "small 1-5%", "medium 5-20%", "large >20%"]
    splits = ["train", "val", "test"]
    counts = {
        split: Counter(bbox_size_category(record.area) for record in records if record.split == split)
        for split in splits
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(splits))

    colors = ["#E45756", "#F58518", "#54A24B", "#4C78A8"]
    for category, color in zip(categories, colors):
        values = np.array([counts[split][category] for split in splits])
        bars = ax.bar(splits, values, bottom=bottom, label=category, color=color)
        ax.bar_label(bars, labels=[str(value) if value else "" for value in values], label_type="center", fontsize=8)
        bottom += values

    ax.set_ylabel("Broj bbox-eva")
    ax.set_title("Koliko bbox-eva ima po velicini")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_bbox_scatter(records: list[BBoxRecord], class_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]

    for class_id, class_name in enumerate(class_names):
        class_records = [record for record in records if record.class_id == class_id]
        ax.scatter(
            [record.width * 100 for record in class_records],
            [record.height * 100 for record in class_records],
            s=26,
            alpha=0.55,
            label=class_name,
            color=colors[class_id % len(colors)],
        )

    ax.set_xlabel("Sirina bbox-a (% slike)")
    ax.set_ylabel("Visina bbox-a (% slike)")
    ax.set_title("Odnos sirine i visine bbox-eva")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def plot_bbox_min_median_max(records: list[BBoxRecord]) -> plt.Figure:
    splits = ["train", "val", "test"]
    metrics = {
        "min": [],
        "median": [],
        "max": [],
    }

    for split in splits:
        areas = [record.area * 100 for record in records if record.split == split]
        metrics["min"].append(min(areas) if areas else 0)
        metrics["median"].append(float(np.median(areas)) if areas else 0)
        metrics["max"].append(max(areas) if areas else 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_positions = np.arange(len(splits))
    width = 0.24

    for index, (name, values) in enumerate(metrics.items()):
        offset = (index - 1) * width
        bars = ax.bar(x_positions + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_xticks(x_positions, splits)
    ax.set_ylabel("Povrsina bbox-a (% slike)")
    ax.set_title("Min / median / max povrsina bbox-a")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_objects_per_image(
    config: dict,
    data_yaml: str | Path = "roboflow_dataset/data.yaml",
) -> plt.Figure:
    splits = ["train", "val", "test"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for split in splits:
        if split not in config:
            continue

        image_dir = split_image_dir(config, split, data_yaml)
        label_dir = split_label_dir(config, split, data_yaml)
        object_counts = []

        for image_path in list_images(image_dir):
            label_file = label_dir / f"{image_path.stem}.txt"
            object_counts.append(len(parse_label_file(label_file)) if label_file.exists() else 0)

        if object_counts:
            bins = np.arange(max(object_counts) + 2) - 0.5
            ax.hist(object_counts, bins=bins, alpha=0.55, label=split)

    ax.set_xlabel("Broj objekata na slici")
    ax.set_ylabel("Broj slika")
    ax.set_title("Distribucija broja objekata po slici")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def bbox_text_summary(records: list[BBoxRecord], class_names: list[str]) -> str:
    lines = ["BBox summary", ""]

    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        if not split_records:
            continue

        areas = np.array([record.area for record in split_records])
        widths = np.array([record.width for record in split_records])
        heights = np.array([record.height for record in split_records])
        category_counts = Counter(bbox_size_category(record.area) for record in split_records)
        class_counts = Counter(record.class_id for record in split_records)

        lines.extend(
            [
                f"{split}:",
                f"  bbox-evi: {len(split_records)}",
                f"  klase: {', '.join(f'{class_names[class_id]}={class_counts[class_id]}' for class_id in range(len(class_names)))}",
                f"  area min/median/max: {areas.min() * 100:.2f}% / {np.median(areas) * 100:.2f}% / {areas.max() * 100:.2f}%",
                f"  width min/median/max: {widths.min() * 100:.2f}% / {np.median(widths) * 100:.2f}% / {widths.max() * 100:.2f}%",
                f"  height min/median/max: {heights.min() * 100:.2f}% / {np.median(heights) * 100:.2f}% / {heights.max() * 100:.2f}%",
                "  velicine: "
                + ", ".join(
                    f"{category}={category_counts[category]}"
                    for category in ("tiny <1%", "small 1-5%", "medium 5-20%", "large >20%")
                ),
                "",
            ]
        )

    return "\n".join(lines)


def generate_dataset_quality_report(
    data_yaml: str | Path = "roboflow_dataset/data.yaml",
    output_dir: str | Path = "reports/dataset_quality",
) -> Path:
    config = load_dataset_config(data_yaml)
    class_names = get_class_names(config)
    summary = dataset_summary(data_yaml)
    records = collect_bbox_records(config, data_yaml)
    output_path = resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {
        "01_images_objects_by_split.png": plot_objects_by_split(summary),
        "02_class_distribution_by_split.png": plot_class_distribution_by_split(summary, class_names),
        "03_bbox_area_histogram.png": plot_bbox_area_histogram(records),
        "04_bbox_width_height_histograms.png": plot_bbox_width_height(records),
        "05_bbox_size_categories.png": plot_bbox_size_categories(records),
        "06_bbox_width_height_scatter.png": plot_bbox_scatter(records, class_names),
        "07_bbox_min_median_max_area.png": plot_bbox_min_median_max(records),
        "08_objects_per_image.png": plot_objects_per_image(config, data_yaml),
    }

    for filename, fig in plots.items():
        save_figure(fig, output_path / filename)

    (output_path / "bbox_summary.txt").write_text(bbox_text_summary(records, class_names), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print_dataset_summary()
    report_dir = generate_dataset_quality_report()
    print(f"Grafici sacuvani u: {report_dir}")
