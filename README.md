# YOLOv8 Detection Pipeline

YOLOv8 pipeline za detekciju klasa `marija` i `masa`.
Projekat obuhvata dataset, konfiguraciju, trening, validaciju i evaluaciju.

Python   YOLOv8   Ultralytics   PyTorch

## Brzi Start

```bash
git clone https://github.com/milenkoviccmarija/IS_project.git
cd IS_project
uv sync
uv run python train.py
```

Evaluacija:

```bash
uv run python evaluate.py
```

Statistika dataseta:

```bash
uv run python dataset.py
```

## Struktura Projekta

```text
IS_project/
├── config.py              ← hiperparametri
├── dataset.py             ← statistika i vizualizacija dataseta
├── model.py               ← kreiranje i ucitavanje YOLO modela
├── train.py               ← trening + validacija
├── evaluate.py            ← evaluacija i grafici
├── roboflow_dataset/      ← YOLO dataset
├── runs/                  ← rezultati treninga
├── pyproject.toml
└── README.md
```

## Dataset

Dataset je u YOLO formatu:

```yaml
path: roboflow_dataset
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['marija', 'masa']
```

## Trening

Glavna podesavanja su u `config.py`.

Trenutno se koristi:

```python
model_yaml = "yolov8n.yaml"
train_from_scratch = True
epochs = 50
imgsz = 512
batch = 8
```

## Evaluacija

Koriste se metrike za detekciju objekata:

- `mAP50`
- `mAP50-95`
- `Precision`
- `Recall`

Rezultati treninga i najbolji model cuvaju se u:

```text
runs/marija_masa_model/
```
