"""Prétraitement des images : CLAHE sur le canal L (Lab), redimensionnement, normalisation.

Pourquoi Lab ? CLAHE appliqué directement sur R, G et B modifie chaque canal
indépendamment et fausse les couleurs (or la couleur rouge/bleu est un indice
fort pour les panneaux). En Lab, L porte la luminance seule : on rehausse le
contraste sans toucher à la chrominance (a, b).
"""

from __future__ import annotations

import cv2
import numpy as np

from clocker import config


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = config.CLAHE_TILE_GRID,
) -> np.ndarray:
    """Applique CLAHE sur le canal de luminance L d'une image RGB uint8."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2RGB)


def denoise_bilateral(image_rgb: np.ndarray) -> np.ndarray:
    """Débruitage bilatéral : moyenne les pixels voisins de couleur proche.

    CLAHE amplifie le contraste local, donc aussi le bruit du capteur
    (mesuré : écart-type du bruit ×2 à ×3 après CLAHE). Lisser avant CLAHE
    limite cet effet sans effacer les contours du pictogramme.
    """
    return cv2.bilateralFilter(
        image_rgb, config.BILATERAL_D, config.BILATERAL_SIGMA_COLOR, config.BILATERAL_SIGMA_SPACE
    )


def crop_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Recadre l'image sur la boîte du panneau (coordonnées fournies par GTSRB)."""
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def preprocess(
    image_rgb: np.ndarray,
    size: int = config.IMG_SIZE,
    use_clahe: bool = True,
    denoise: bool = False,
) -> np.ndarray:
    """Pipeline complet : resize -> [débruitage] -> [CLAHE] -> float32 dans [0, 1].

    Le redimensionnement est fait avant CLAHE pour que la grille de tuiles
    soit identique quelle que soit la taille d'origine.
    """
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    interp = cv2.INTER_AREA if min(image_rgb.shape[:2]) > size else cv2.INTER_CUBIC
    img = cv2.resize(image_rgb, (size, size), interpolation=interp)
    return finish(img, use_clahe, denoise)


def finish(img_u8: np.ndarray, use_clahe: bool = True, denoise: bool = False) -> np.ndarray:
    """Fin du pipeline sur une image déjà redimensionnée (uint8 RGB)."""
    if denoise:
        img_u8 = denoise_bilateral(img_u8)
    if use_clahe:
        img_u8 = apply_clahe(img_u8)
    return img_u8.astype(np.float32) / 255.0


def preprocess_batch(images: list[np.ndarray], **kwargs) -> np.ndarray:
    """Applique `preprocess` à une liste d'images et empile le résultat (N, H, W, 3)."""
    return np.stack([preprocess(img, **kwargs) for img in images])


def load_rgb(path) -> np.ndarray:
    """Lit une image depuis le disque en RGB (OpenCV lit en BGR)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image illisible : {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
