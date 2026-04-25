from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.config import DATA_DIR
from app.training import train_and_save_model


if __name__ == '__main__':
    train_and_save_model(DATA_DIR / 'final_training_table.csv')
