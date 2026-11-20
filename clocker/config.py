"""Paramètres globaux du projet (chemins, tailles, hyperparamètres)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "gtsrb"          # contenu du zip Kaggle
CACHE_DIR = DATA_DIR / "cache"        # tableaux .npz prétraités
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "clocker_cnn.keras"

NUM_CLASSES = 43
IMG_SIZE = 32                # 32 ou 64
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (4, 4)     # grille adaptée aux petites images 32×32

VAL_SPLIT = 0.15
SEED = 42

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
