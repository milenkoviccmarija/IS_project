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

Dataset je sopstveni dataset napravljen i anotiran u Roboflow-u. Slike su
rasporedjene u train/validation/test skup i eksportovane u YOLO formatu.

```yaml
path: roboflow_dataset
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['marija', 'masa']
```

Trenutna podela dataseta:

| Skup | Slike | Label fajlovi | Objekti |
| --- | ---: | ---: | ---: |
| Train | 242 | 242 | 266 |
| Validation | 49 | 49 | 50 |
| Test | 38 | 38 | 43 |

Raspodela klasa je priblizno balansirana. U trening skupu ima 134 objekta
klase `marija` i 132 objekta klase `masa`.

Pravila anotacije:

- svaki objekat je oznacen bounding box-om
- box obuhvata vidljivi deo objekta koji pripada jednoj od dve klase
- klase su `marija` i `masa`
- slike bez objekta se koriste kao background primeri

Kvalitet dataseta se proverava u `train.py` pre treninga: proverava se da li
postoje slike i labele, da li svaka slika ima odgovarajuci `.txt` fajl i da li
su YOLO koordinate normalizovane u opsegu od 0 do 1.

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

Model je treniran od nule, bez pretrained tezina.

## Evaluacija

Koriste se metrike za detekciju objekata:

- `mAP50`
- `mAP50-95`
- `Precision`
- `Recall`
- `F1`

Dobijeni rezultati:

| Skup | mAP50 | mAP50-95 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.3175 | 0.1925 | 0.3345 | 0.5800 | 0.4243 |
| Test | 0.3497 | 0.2228 | 0.3165 | 0.5871 | 0.4113 |

Model uspeva da pronadje deo objekata, ali jos pravi greske u klasifikaciji i
ima visak detekcija. To je ocekivano za YOLO model treniran od nule nad
relativno malim datasetom.

Rezultati treninga i najbolji model cuvaju se u:

```text
runs/marija_masa_model/
```

Najvazniji fajlovi za pregled:

- `runs/marija_masa_model/results.png`
- `runs/marija_masa_model/confusion_matrix.png`
- `runs/marija_masa_model/training_summary.txt`
- `runs/marija_masa_model/validation_errors.txt`
- `runs/marija_masa_model/weights/best.pt`

## Izvori

- Ultralytics YOLOv8
- PyTorch
- Roboflow
