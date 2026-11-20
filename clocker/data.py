"""Chargement du dataset GTSRB (version Kaggle) et mise en cache.

Structure attendue dans data/gtsrb/ (contenu du zip Kaggle) :
    Train.csv, Test.csv, Meta.csv
    Train/<ClassId>/*.png
    Test/*.png
Colonnes des CSV : Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from clocker import config
from clocker.preprocessing import crop_roi, load_rgb, preprocess


def read_split(split: str, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Lit Train.csv ou Test.csv et ajoute le chemin absolu de chaque image."""
    csv_path = raw_dir / f"{split.capitalize()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Lancez d'abord : python scripts/download_data.py"
        )
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={"Roi.X1": "x1", "Roi.Y1": "y1", "Roi.X2": "x2", "Roi.Y2": "y2"}
    )
    df["abs_path"] = df["Path"].apply(lambda p: str(raw_dir / p))
    return df


def load_images(
    df: pd.DataFrame,
    size: int = config.IMG_SIZE,
    use_clahe: bool = True,
    use_roi: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge, recadre et prétraite toutes les images d'un DataFrame."""
    images, labels = [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Images"):
        img = load_rgb(row.abs_path)
        if use_roi:
            img = crop_roi(img, row.x1, row.y1, row.x2, row.y2)
        images.append(preprocess(img, size=size, use_clahe=use_clahe))
        labels.append(row.ClassId)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def cache_path(size: int, use_clahe: bool) -> Path:
    """Nom du fichier cache selon la configuration de prétraitement."""
    tag = "clahe" if use_clahe else "raw"
    return config.CACHE_DIR / f"gtsrb_{size}_{tag}.npz"


def build_cache(size: int = config.IMG_SIZE, use_clahe: bool = True) -> Path:
    """Prétraite train (+ découpage validation stratifié) et test, puis sauvegarde en .npz."""
    train_df = read_split("train")
    test_df = read_split("test")

    x_all, y_all = load_images(train_df, size=size, use_clahe=use_clahe)
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all,
        test_size=config.VAL_SPLIT,
        stratify=y_all,
        random_state=config.SEED,
    )
    x_test, y_test = load_images(test_df, size=size, use_clahe=use_clahe)

    out = cache_path(size, use_clahe)
    out.parent.mkdir(parents=True, exist_ok=True)
    # float16 divise la taille du cache par deux sans perte utile
    np.savez_compressed(
        out,
        x_train=x_train.astype(np.float16), y_train=y_train,
        x_val=x_val.astype(np.float16), y_val=y_val,
        x_test=x_test.astype(np.float16), y_test=y_test,
    )
    print(f"Cache écrit : {out}  (train={len(y_train)}, val={len(y_val)}, test={len(y_test)})")
    return out


def load_cache(size: int = config.IMG_SIZE, use_clahe: bool = True) -> dict[str, np.ndarray]:
    """Charge le cache .npz (le construit s'il n'existe pas)."""
    path = cache_path(size, use_clahe)
    if not path.exists():
        build_cache(size, use_clahe)
    with np.load(path) as data:
        return {k: data[k].astype(np.float32) if k.startswith("x_") else data[k] for k in data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prétraite GTSRB et crée le cache .npz")
    parser.add_argument("--size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--no-clahe", action="store_true")
    args = parser.parse_args()
    build_cache(size=args.size, use_clahe=not args.no_clahe)


if __name__ == "__main__":
    main()
