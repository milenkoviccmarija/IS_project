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

Statistika dataseta i dataset quality report:

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
├── reports/               ← statistika i grafici dataseta
├── runs/                  ← rezultati treninga i evaluacije
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

roboflow:
  version: 17
```

Trenutna podela dataseta:

| Skup | Slike | Label fajlovi | Objekti |
| --- | ---: | ---: | ---: |
| Train | 378 | 378 | 415 |
| Validation | 69 | 69 | 70 |
| Test | 75 | 75 | 80 |

Raspodela klasa je balansirana. U trening skupu ima 208 objekata klase
`marija` i 207 objekata klase `masa`. U validation skupu ima 35/35 objekata,
a u test skupu 40/40 objekata po klasama.

Dataset quality grafici cuvaju se u:

```text
reports/dataset_quality/
```

Najvazniji uvid iz dataset reporta je da ima dosta malih bounding box-eva.
Medijana povrsine bbox-a je 3.98% slike za train, 6.93% za validation i
5.98% za test.

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
epochs = 200
imgsz = 640
batch = 8
patience = 40
```

Model je treniran od nule, bez pretrained tezina. Velicina slike je povecana
na 640 zbog malih objekata u datasetu.

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
| Validation | 0.8385 | 0.5843 | 0.7849 | 0.8143 | 0.7993 |
| Test | 0.8860 | 0.6200 | 0.8701 | 0.8603 | 0.8652 |

Rezultati na test skupu po klasama:

| Klasa | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| marija | 0.888 | 0.796 | 0.842 | 0.608 |
| masa | 0.852 | 0.925 | 0.930 | 0.632 |

Model dobro detektuje obe klase. Klasa `masa` ima veci recall, dok klasa
`marija` ima veci precision. Najcesce greske su visak detekcija, pogresna
klasa na manjem broju slika i poneki promasen objekat.

Na validation skupu je u `validation_errors.txt` pronadjeno 49 prijavljenih
gresaka na 34 slike:

- 35 visak detekcija
- 5 pogresnih klasa
- 9 promasenih objekata

F1-confidence kriva pokazuje da je najbolji prag pouzdanosti oko 0.59. Zato
je za predikcije prakticnije koristiti `conf` oko 0.55-0.60, umesto vrlo
niskog praga 0.25 koji se koristi za detaljniju analizu gresaka.

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
- `runs/marija_masa_model_validation/`
- `runs/marija_masa_model_test/`
- `runs/marija_masa_model_evaluation/`

## Izvori

- Ultralytics YOLOv8
- PyTorch
- Roboflow
