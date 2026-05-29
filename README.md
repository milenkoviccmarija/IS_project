# YOLOv8 Face/Object Detection Pipeline

Kompletan YOLOv8 trening pipeline za detekciju klasa `marija` i `masa`.
Demonstracija VNM projekta - dataset, konfiguracija, trening, validacija,
evaluacija i vizualizacija rezultata.

Python   YOLOv8   Ultralytics   PyTorch

## Brzi Start

Lokalno:

```bash
git clone https://github.com/milenkoviccmarija/IS_project.git
cd IS_project
uv sync
uv run python train.py
```

Evaluacija istreniranog modela:

```bash
uv run python evaluate.py
```

Pregled dataseta:

```bash
uv run python dataset.py
```

## Napomena Za Trening

Projekat trenutno trenira YOLO arhitekturu od nule:

```python
model_yaml = "yolov8n.yaml"
train_from_scratch = True
```

Ako zelis pretrained trening, u `config.py` promeni:

```python
train_from_scratch = False
```

Tada se koristi `yolov8n.pt`, koji Ultralytics moze da preuzme automatski ako
ne postoji lokalno.

## Struktura Projekta

```text
IS_project/
│
├── config.py                  ← svi hiperparametri na jednom mestu
├── dataset.py                 ← ucitavanje data.yaml, statistika i vizualizacija
├── model.py                   ← kreiranje i ucitavanje YOLO modela
├── train.py                   ← trening + validacija najboljeg modela
├── evaluate.py                ← test evaluacija, metrike i grafici
│
├── roboflow_dataset/          ← YOLO dataset u Roboflow formatu
│   ├── data.yaml              ← putanje, klase i splitovi
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/                      ← rezultati treninga i sacuvani modeli
│   └── marija_masa_model/
│       ├── weights/best.pt
│       ├── weights/last.pt
│       ├── results.csv
│       ├── results.png
│       └── confusion_matrix.png
│
├── pyproject.toml             ← zavisnosti projekta
└── README.md
```

## Sta Pipeline Pokriva

| Sekcija | Sadrzaj |
| --- | --- |
| 1. Konfiguracija | Dataset, YOLO arhitektura, batch size, broj epoha, seed |
| 2. Dataset | Roboflow YOLO format, train/valid/test splitovi, raspodela klasa |
| 3. Model | Ultralytics YOLOv8 model kroz `model.py` |
| 4. Trening | Treniranje YOLOv8n arhitekture od nule |
| 5. Validacija | Validacija najboljeg modela na `valid` skupu |
| 6. Evaluacija | Test evaluacija na `test` skupu |
| 7. Metrike | mAP50, mAP50-95, precision, recall |
| 8. Vizualizacije | Loss/mAP krive, matrica konfuzije, predikcije na slikama |

## Zasto Nema Rucni CNN Kao Kod MNIST-a

Profesorov MNIST primer koristi rucno definisane `MLP` i `CNN` klase jer je to
zadatak klasifikacije slika:

```text
slika -> jedna klasa
```

Ovaj projekat je zadatak detekcije objekata:

```text
slika -> bounding box + klasa + confidence
```

Zato se ne pise rucno `nn.Module` CNN arhitektura, vec se koristi Ultralytics
YOLO implementacija. `model.py` i dalje ima istu ulogu kao u PyTorch pipeline-u:
na jednom mestu se kreira ili ucitava model.

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

YOLO automatski koristi odgovarajuce `labels/` foldere uz svaki split.
Zbog toga projekat nema rucno definisan PyTorch `DataLoader`.

## Korisne Komande

Trening:

```bash
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

Ako ne koristis `uv`, mozes pokrenuti preko aktivnog virtuelnog okruzenja:

```bash
python train.py
python evaluate.py
python dataset.py
```

## Kako Adaptirati Za Drugi Projekat

Struktura pipeline-a ostaje ista. Menja se:

- `roboflow_dataset/data.yaml` -> putanje i klase novog dataseta
- `config.py` -> hiperparametri treninga
- `model.py` -> YOLO varijanta ili putanja do modela
- `evaluate.py` -> dodatne metrike i vizualizacije po potrebi
