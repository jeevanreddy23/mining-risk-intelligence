from pathlib import Path


# src/app/config.py -> repo root is two levels above src/
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "wa_mining_synthetic.csv"
MODEL_PATH = DATA_DIR / "model.joblib"
METRICS_PATH = DATA_DIR / "metrics.json"
