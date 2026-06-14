# Traffic Sign Recognition — ML Research

Deep learning models for classifying 43 classes of German Traffic Signs (GTSRB dataset).

---

## Notebooks

| File | Framework | Model | Val Accuracy |
| ---- | --------- | ----- | ------------ |
| `MNV2_TS.ipynb` | TensorFlow/Keras | MobileNetV2 | ~99% |
| `MNV3_TRAFFIC SIGNS_FINAL.ipynb` | TensorFlow/Keras | MobileNetV3 Large | 99.14% |
| `EFFICIENTNET-B7.ipynb` | PyTorch | EfficientNet-B7 | **99.66%** |
| `XAI_TS.ipynb` | PyTorch | LIME Explainability | — |

Dataset: 43 traffic sign classes, 224×224px images, ~87K train images (2023 per class).

---

## Setup — Any Device

### 1. Install Python 3.11

Download: <https://www.python.org/downloads/release/python-3110/>

### 2. Install dependencies

For TensorFlow notebooks (MNV2, MNV3):

```bash
pip install -r requirements_tensorflow.txt
```

For PyTorch notebooks (EfficientNet-B7, XAI):

```bash
pip install -r requirements_pytorch.txt
```

### 3. Install Jupyter

```bash
pip install jupyter notebook
```

### 4. Set your dataset paths

Open `paths_config.py` and update the 3 paths to match your machine:

```python
DATASET_ROOT = r"I:\DATASET\TRAFFIC SIGNS PICKLE DATA 0 224x224"
LABEL_CSV    = r"G:\Datasets\Traffic Signs\label_names.csv"
XAI_SAMPLES  = r"I:\DATASET\XAI SAMPLES"
```

Then at the top of each notebook, replace hardcoded paths with:

```python
from paths_config import TRAIN_DIR, VALID_DIR, TEST_DIR, LABEL_CSV, XAI_SAMPLES
```

### 5. Dataset folder structure

```
TRAFFIC SIGNS PICKLE DATA 0 224x224/
├── train/
│   ├── 0/   (Speed limit 20km/h)
│   ├── 1/   (Speed limit 30km/h)
│   └── ...  (up to 42/)
├── valid/
└── test/
```

Label CSV (`label_names.csv`) columns: `ClassId`, `SignName`

### 6. Run

```bash
jupyter notebook
```

Open any `.ipynb` and run all cells.

---

## Hardware Used

- GPU: NVIDIA GeForce GTX 1070
- Python: 3.11

GPU strongly recommended. EfficientNet-B7 takes ~1h per epoch on GTX 1070.

---

## Model Results

### EfficientNet-B7 (PyTorch) — Best Model

| Epoch | Train Acc | Val Acc |
| ----- | --------- | ------- |
| 1 | 95.18% | 99.59% |
| 5 | 99.93% | 99.52% |
| 10 | 100.00% | **99.66%** |

### MobileNetV3 Large (TensorFlow) — Phase 2 Fine-tuning

| Epoch | Train Acc | Val Acc |
| ----- | --------- | ------- |
| 1 | 99.28% | 98.96% |
| 5 | 99.68% | **99.14%** |

---

## GitHub

Main repo: <https://github.com/TabassumAuthoy123/CODES>
