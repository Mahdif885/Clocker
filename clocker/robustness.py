"""Perturbations artificielles pour tester la robustesse (flou, bruit, éclairage).

Toutes les fonctions prennent et renvoient des images float32 dans [0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np

_RNG = np.random.default_rng(0)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def motion_blur(img: np.ndarray, length: int) -> np.ndarray:
    """Flou de bougé horizontal (véhicule en mouvement)."""
    length = int(length)
    if length <= 1:
        return img
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return img
    return np.clip(img + _RNG.normal(0, std, img.shape).astype(np.float32), 0, 1)


def salt_pepper(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    out = img.copy()
    mask = _RNG.random(img.shape[:2])
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return out


def darken(img: np.ndarray, factor: float) -> np.ndarray:
    """Assombrit l'image (factor = 1 : inchangée, 0,3 : très sombre)."""
    return np.clip(img * factor, 0, 1)


# Nom -> (fonction, niveaux testés). Le premier niveau = image non perturbée.
PERTURBATIONS = {
    "Flou gaussien (sigma)": (gaussian_blur, [0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    "Flou de bougé (px)": (motion_blur, [0, 3, 5, 7, 9]),
    "Bruit gaussien (écart-type)": (gaussian_noise, [0, 0.02, 0.05, 0.1, 0.15, 0.2]),
    "Sel et poivre (proportion)": (salt_pepper, [0, 0.01, 0.03, 0.05, 0.1]),
    "Assombrissement (facteur)": (darken, [1.0, 0.8, 0.6, 0.4, 0.25]),
}


def apply_to_batch(fn, images: np.ndarray, level: float) -> np.ndarray:
    return np.stack([fn(img, level) for img in images]).astype(np.float32)
