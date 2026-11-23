"""Reconnaissance sur une scène complète : détection des candidats puis classification.

Module partagé par l'évaluation sur GTSDB (scripts/eval_detection.py) et la
démo vidéo en direct (scripts/live.py). Il réutilise les fonctions existantes
de `clocker.detection` et `clocker.preprocessing` sans les modifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clocker.detection import Candidate, crop, detect_signs
from clocker.preprocessing import preprocess


@dataclass
class Detection:
    cand: Candidate
    class_id: int
    conf: float
    probs: np.ndarray

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.cand.box


def recognise(model, image_rgb: np.ndarray, use_clahe: bool = True, denoise: bool = False,
              threshold: float = 0.0, use_hough: bool = False) -> list[Detection]:
    """Détecte les panneaux candidats, classe chaque recadrage, garde ceux ≥ threshold."""
    cands = detect_signs(image_rgb, use_hough=use_hough)
    if not cands:
        return []
    size = model.input_shape[1]
    x = np.stack([preprocess(crop(image_rgb, c), size=size, use_clahe=use_clahe,
                             denoise=denoise) for c in cands])
    probs = model(x, training=False).numpy()
    out = []
    for c, p in zip(cands, probs):
        k = int(p.argmax())
        if p[k] >= threshold:
            out.append(Detection(c, k, float(p[k]), p))
    return out


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection sur union de deux boîtes (x1, y1, x2, y2)."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
