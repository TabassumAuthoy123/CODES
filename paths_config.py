import os

# ── Edit these 3 lines to match your machine ──────────────────────────────────
DATASET_ROOT   = r"I:\DATASET\TRAFFIC SIGNS PICKLE DATA 0 224x224"
LABEL_CSV      = r"G:\Datasets\Traffic Signs\label_names.csv"
XAI_SAMPLES    = r"I:\DATASET\XAI SAMPLES"
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")
