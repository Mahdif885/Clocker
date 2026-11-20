"""Détection / localisation des panneaux candidats (partie bonus du sujet).

Deux indices combinés :
1. Couleur : masques HSV rouge et bleu. Le rouge est à cheval sur 0° de teinte,
   d'où deux plages (0-10 et 170-180 dans l'échelle OpenCV 0-180).
2. Forme : cercles (cv2.HoughCircles) et polygones (triangles, octogone du Stop)
   via cv2.approxPolyDP sur les contours du masque.

À améliorer par le Membre B (réglage des seuils sur de vraies photos de rue).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Seuils HSV (H: 0-180, S: 0-255, V: 0-255) — à ajuster
RED_RANGES = [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))]
BLUE_RANGES = [((100, 120, 50), (130, 255, 255))]

MIN_AREA_RATIO = 0.0004   # aire minimale d'un candidat / aire de l'image
MAX_ASPECT = 1.5          # panneaux ≈ carrés


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    color: str   # "rouge" ou "bleu"
    shape: str   # "cercle", "triangle", "octogone", "autre"

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def color_mask(image_rgb: np.ndarray, ranges) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _shape_name(contour) -> str:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n = len(approx)
    if n == 3:
        return "triangle"
    if n == 8:
        return "octogone"
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
    return "cercle" if circularity > 0.7 else "autre"


def _from_mask(mask: np.ndarray, color: str, min_area: float) -> list[Candidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area or max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
            continue
        out.append(Candidate(x, y, w, h, color, _shape_name(c)))
    return out


def hough_circles(image_rgb: np.ndarray) -> list[Candidate]:
    """Cercles détectés par la transformée de Hough sur l'image en niveaux de gris."""
    gray = cv2.medianBlur(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) / 8,
        param1=120, param2=40, minRadius=max(6, min(h, w) // 60), maxRadius=min(h, w) // 3,
    )
    if circles is None:
        return []
    out = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        out.append(Candidate(max(0, cx - r), max(0, cy - r), 2 * r, 2 * r, "?", "cercle"))
    return out


def _iou(a: Candidate, b: Candidate) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def detect_signs(image_rgb: np.ndarray, use_hough: bool = True) -> list[Candidate]:
    """Renvoie les boîtes candidates (couleur + forme), sans doublons."""
    min_area = MIN_AREA_RATIO * image_rgb.shape[0] * image_rgb.shape[1]
    cands = _from_mask(color_mask(image_rgb, RED_RANGES), "rouge", min_area)
    cands += _from_mask(color_mask(image_rgb, BLUE_RANGES), "bleu", min_area)
    if use_hough:
        for c in hough_circles(image_rgb):
            if all(_iou(c, k) < 0.3 for k in cands):
                cands.append(c)
    return cands


def crop(image_rgb: np.ndarray, cand: Candidate, margin: float = 0.1) -> np.ndarray:
    """Recadre un candidat avec une petite marge (comme les ROI GTSRB)."""
    h, w = image_rgb.shape[:2]
    mx, my = int(cand.w * margin), int(cand.h * margin)
    x1, y1 = max(0, cand.x - mx), max(0, cand.y - my)
    x2, y2 = min(w, cand.x + cand.w + mx), min(h, cand.y + cand.h + my)
    return image_rgb[y1:y2, x1:x2]
