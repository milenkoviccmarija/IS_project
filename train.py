from pathlib import Path
import re

import torch
from PIL import Image

from config import Config
from dataset import list_images, load_dataset_config, split_image_dir, split_label_dir
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


def check_dataset(config: Config) -> None:
    dataset_config = load_dataset_config(config.data_yaml)
    errors = []
    required_splits = ("train", config.val_split, config.test_split)
    class_count = int(dataset_config.get("nc", len(dataset_config.get("names", []))))

    for split in required_splits:
        if split not in dataset_config:
            errors.append(f"Split '{split}' ne postoji u data.yaml.")
            continue

        images_dir = split_image_dir(dataset_config, split, config.data_yaml)
        labels_dir = split_label_dir(dataset_config, split, config.data_yaml)

        if not images_dir.exists():
            errors.append(f"Folder sa slikama ne postoji za '{split}': {images_dir}")
            continue

        if not labels_dir.exists():
            errors.append(f"Folder sa labelama ne postoji za '{split}': {labels_dir}")
            continue

        image_paths = list_images(images_dir)
        if not image_paths:
            errors.append(f"Split '{split}' nema nijednu sliku.")

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Nedostaje label fajl za sliku: {image_path}")
                continue

            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Los format labele u {label_path}:{line_number}")
                    continue

                try:
                    class_id = int(float(parts[0]))
                    coordinates = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"Labela nije broj u {label_path}:{line_number}")
                    continue

                if not 0 <= class_id < class_count:
                    errors.append(f"Nepostojeca klasa {class_id} u {label_path}:{line_number}")

                if any(value < 0 or value > 1 for value in coordinates):
                    errors.append(f"Koordinate nisu normalizovane u {label_path}:{line_number}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        extra = f"\n... i jos {len(errors) - 20} problema." if len(errors) > 20 else ""
        raise ValueError(f"Dataset provera nije prosla:\n{preview}{extra}")

    print("Dataset provera je prosla.")


def train_yolo(config: Config, device: str):
    if config.resume_training:
        resume_path = Path(resolve_project_path(config.resume_weights))
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint za nastavak treninga ne postoji: {resume_path}")

        model = load_model(resume_path)
        return model.train(resume=True, device=device)

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
        save_period=config.save_period,
        project=resolve_project_path(config.project),
        name=config.name,
        exist_ok=config.exist_ok,
    )


def validate_yolo(config: Config, weights_path: Path, device: str, split: str | None = None, name_suffix: str = "validation"):
    model = load_model(weights_path)

    return model.val(
        data=resolve_project_path(config.data_yaml),
        split=split or config.val_split,
        imgsz=config.imgsz,
        batch=config.batch,
        device=device,
        project=resolve_project_path(config.project),
        name=f"{config.name}_{name_suffix}",
        exist_ok=config.exist_ok,
        plots=True,
    )


def detection_f1(metrics) -> float:
    precision = metrics.box.mp
    recall = metrics.box.mr
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def print_detection_metrics(metrics, title: str) -> None:
    print(f"{title} zavrsena.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"F1: {detection_f1(metrics):.4f}")


def print_validation_metrics(metrics) -> None:
    print_detection_metrics(metrics, "Validacija")


def yolo_label_to_xyxy(values: list[float], image_width: int, image_height: int) -> list[float]:
    x_center, y_center, width, height = values
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return [x1, y1, x2, y2]


def read_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[dict]:
    if not label_path.exists():
        return []

    labels = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        box = yolo_label_to_xyxy([float(value) for value in parts[1:5]], image_width, image_height)
        labels.append({"class_id": class_id, "box": box})

    return labels


def box_iou(first_box: list[float], second_box: list[float]) -> float:
    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first_box[2] - first_box[0]) * max(0, first_box[3] - first_box[1])
    second_area = max(0, second_box[2] - second_box[0]) * max(0, second_box[3] - second_box[1])
    union = first_area + second_area - intersection

    return intersection / union if union > 0 else 0


def class_name(class_id: int, names) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))

    if 0 <= class_id < len(names):
        return names[class_id]

    return str(class_id)


def collect_detection_errors(model, config: Config, image_paths: list[Path], labels_dir: Path, device: str) -> tuple[int, int, list[str]]:
    report_lines = []
    error_count = 0
    images_with_errors = 0

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=config.imgsz,
        conf=config.error_conf_threshold,
        device=device,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        label_path = labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        labels = read_yolo_labels(label_path, image_width, image_height)
        predictions = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            predictions = [
                {"class_id": int(class_id), "box": box, "conf": conf}
                for box, class_id, conf in zip(boxes, classes, confidences)
            ]

        matched_predictions = set()
        image_errors = []

        for label in labels:
            best_prediction_index = None
            best_iou = 0

            for prediction_index, prediction in enumerate(predictions):
                if prediction_index in matched_predictions:
                    continue

                iou = box_iou(label["box"], prediction["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_prediction_index = prediction_index

            expected = class_name(label["class_id"], model.names)

            if best_prediction_index is None or best_iou < config.error_iou_threshold:
                image_errors.append(f"  - promasen objekat: ocekivano '{expected}'")
                continue

            matched_predictions.add(best_prediction_index)
            prediction = predictions[best_prediction_index]
            predicted = class_name(prediction["class_id"], model.names)

            if prediction["class_id"] != label["class_id"]:
                image_errors.append(
                    f"  - pogresna klasa: ocekivano '{expected}', dobijeno '{predicted}' "
                    f"(conf={prediction['conf']:.2f}, IoU={best_iou:.2f})"
                )

        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions:
                continue

            predicted = class_name(prediction["class_id"], model.names)
            image_errors.append(f"  - visak detekcija: '{predicted}' (conf={prediction['conf']:.2f})")

        if image_errors:
            error_count += len(image_errors)
            images_with_errors += 1
            report_lines.append(str(image_path))
            report_lines.extend(image_errors)
            report_lines.append("")

    return error_count, images_with_errors, report_lines


def validation_dataset_paths(config: Config) -> tuple[list[Path], Path]:
    dataset_config = load_dataset_config(config.data_yaml)
    images_dir = split_image_dir(dataset_config, config.val_split, config.data_yaml)
    labels_dir = split_label_dir(dataset_config, config.val_split, config.data_yaml)
    return list_images(images_dir), labels_dir


def find_validation_errors(config: Config, weights_path: Path, device: str, save_dir: Path) -> None:
    model = load_model(weights_path)
    image_paths, labels_dir = validation_dataset_paths(config)

    print("Proveravam na kojim validacionim slikama najbolji model gresi...")
    error_count, _, report_lines = collect_detection_errors(model, config, image_paths, labels_dir, device)

    report_path = save_dir / "validation_errors.txt"
    if report_lines:
        report_path.write_text("\n".join(report_lines))
        print(f"Pronadjeno gresaka: {error_count}")
        print("Slike na kojima je model pogresio:")
        print("\n".join(report_lines))
        print(f"Spisak slika sa greskama sacuvan je u: {report_path}")
    else:
        report_path.write_text("Nema pronadjenih gresaka na validacionom skupu.\n")
        print("Nema pronadjenih gresaka na validacionom skupu.")


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def find_validation_errors_by_epoch(config: Config, save_dir: Path, device: str) -> None:
    if not config.track_epoch_errors:
        return

    weights_dir = save_dir / "weights"
    epoch_weights = sorted(weights_dir.glob("epoch*.pt"), key=checkpoint_epoch)

    if not epoch_weights:
        print("Nema epoch*.pt checkpointa. Za pracenje kroz epohe postavi save_period=1 i ponovo pokreni trening.")
        return

    image_paths, labels_dir = validation_dataset_paths(config)
    report_lines = []
    summary_lines = ["epoch,checkpoint,errors,images_with_errors"]

    print("Proveravam greske kroz epohe...")

    for weights_path in epoch_weights:
        epoch = checkpoint_epoch(weights_path)
        model = load_model(weights_path)
        error_count, images_with_errors, errors = collect_detection_errors(
            model,
            config,
            image_paths,
            labels_dir,
            device,
        )

        summary_lines.append(f"{epoch},{weights_path.name},{error_count},{images_with_errors}")
        report_lines.append(f"=== Epoha {epoch} ({weights_path.name}) ===")

        if errors:
            report_lines.extend(errors)
        else:
            report_lines.append("Nema pronadjenih gresaka na validacionom skupu.")
            report_lines.append("")

        print(f"Epoha {epoch}: greske={error_count}, slike sa greskama={images_with_errors}")

    report_path = save_dir / "validation_errors_by_epoch.txt"
    summary_path = save_dir / "validation_errors_by_epoch.csv"
    report_path.write_text("\n".join(report_lines))
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Greske kroz epohe sacuvane su u: {report_path}")
    print(f"Kratak pregled po epohama sacuvan je u: {summary_path}")


def metric_lines(title: str, metrics) -> list[str]:
    return [
        f"{title}:",
        f"  mAP50: {metrics.box.map50:.4f}",
        f"  mAP50-95: {metrics.box.map:.4f}",
        f"  Precision: {metrics.box.mp:.4f}",
        f"  Recall: {metrics.box.mr:.4f}",
        f"  F1: {detection_f1(metrics):.4f}",
    ]


def write_training_summary(
    config: Config,
    save_dir: Path,
    best_model: Path,
    last_model: Path,
    device: str,
    validation_metrics,
    test_metrics,
) -> None:
    model_path = config.resume_weights if config.resume_training else (
        config.model_yaml if config.train_from_scratch else config.pretrained_weights
    )
    lines = [
        "YOLO trening summary",
        "",
        f"Uredjaj: {device}",
        f"Dataset: {config.data_yaml}",
        f"Model: {model_path}",
        f"Treniranje od nule: {config.train_from_scratch}",
        f"Resume trening: {config.resume_training}",
        f"Epohe: {config.epochs}",
        f"Velicina slike: {config.imgsz}",
        f"Batch: {config.batch}",
        f"Patience: {config.patience}",
        f"Seed: {config.seed}",
        "",
        f"Rezultati folder: {save_dir}",
        f"Najbolji model: {best_model}",
        f"Poslednji model: {last_model}",
        "",
        *metric_lines("Validacija", validation_metrics),
        "",
        *metric_lines("Test", test_metrics),
    ]

    summary_path = save_dir / "training_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Summary treninga sacuvan je u: {summary_path}")


def main() -> None:
    config = Config()
    device = resolve_device()
    if config.resume_training:
        model_description = f"nastavak treninga iz {config.resume_weights}"
    else:
        model_description = "prazna YOLO arhitektura" if config.train_from_scratch else "pretrained YOLO model"

    print(f"Uredjaj: {device}")
    print(f"Model: {config.model_yaml} ({model_description})")
    print(f"Dataset: {config.data_yaml}")
    print("Proveravam dataset pre treninga...")
    check_dataset(config)

    if config.resume_training:
        print("Nastavljam prekinuti trening...")
    else:
        print("Pokrecem treniranje od nule...")

    results = train_yolo(config, device)

    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    print(f"Trening zavrsen. Rezultati su sacuvani u: {save_dir}")
    print(f"Najbolji model: {best_model}")
    print(f"Poslednji model: {last_model}")
    print("Pokrecem validaciju najboljeg modela...")

    validation_metrics = validate_yolo(config, best_model, device)
    print_validation_metrics(validation_metrics)
    print("Pokrecem test evaluaciju najboljeg modela...")
    test_metrics = validate_yolo(config, best_model, device, split=config.test_split, name_suffix="test")
    print_detection_metrics(test_metrics, "Test evaluacija")
    write_training_summary(config, save_dir, best_model, last_model, device, validation_metrics, test_metrics)
    find_validation_errors(config, best_model, device, save_dir)
    find_validation_errors_by_epoch(config, save_dir, device)


if __name__ == "__main__":
    main()
