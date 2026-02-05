from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

for p in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 5 seasons (adjust as desired)
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

# Filter to reduce small-sample noise
MIN_GP = 15