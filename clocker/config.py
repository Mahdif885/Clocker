"""Paramètres globaux du projet (chemins, tailles, hyperparamètres)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "gtsrb"          # contenu du zip Kaggle
CACHE_DIR = DATA_DIR / "cache"        # tableaux .npz prétraités
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "clahe_debruit.keras"

NUM_CLASSES = 43
IMG_SIZE = 32                # 32 ou 64
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (4, 4)     # grille adaptée aux petites images 32×32
# Débruitage bilatéral avant CLAHE (lisse le bruit, préserve les contours)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 5

VAL_SPLIT = 0.15
SEED = 42

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3


def prep_path(name: str):
    """Fichier models/<nom>.json : prétraitement utilisé pour entraîner le modèle."""
    return MODELS_DIR / f"{name}.json"


def save_prep(name: str, **settings) -> None:
    import json

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prep_path(name).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def load_prep(name: str, use_clahe: bool = True, denoise: bool = False) -> dict:
    """Relit les réglages du modèle ; à défaut, utilise les valeurs passées en argument."""
    import json

    settings = {"clahe": use_clahe, "denoise": denoise}
    if prep_path(name).exists():
        settings.update(json.loads(prep_path(name).read_text(encoding="utf-8")))
    return settings
