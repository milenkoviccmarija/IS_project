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
        save_period=config.save_period,
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
        plots=True,
    )


def print_validation_metrics(metrics) -> None:
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print("Validacija zavrsena.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


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

    for result in model.predict(
        source=[str(path) for path in image_paths],
        imgsz=config.imgsz,
        conf=config.error_conf_threshold,
        device=device,
        verbose=False,
    ):
        image_path = Path(result.path)
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
    find_validation_errors(config, best_model, device, save_dir)
    find_validation_errors_by_epoch(config, save_dir, device)


if __name__ == "__main__":
    main()
