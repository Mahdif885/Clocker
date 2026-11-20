"""Tests rapides (sans dataset) : prétraitement, labels, détection, modèle."""

import cv2
import numpy as np

from clocker import config
from clocker.detection import detect_signs
from clocker.labels import CLASS_CATEGORY, CLASS_NAMES
from clocker.preprocessing import apply_clahe, crop_roi, preprocess
from clocker.robustness import PERTURBATIONS, apply_to_batch


def _random_rgb(h=50, w=60):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_labels_cover_all_classes():
    assert set(CLASS_NAMES) == set(range(config.NUM_CLASSES))
    assert set(CLASS_CATEGORY) == set(range(config.NUM_CLASSES))


def test_preprocess_shape_and_range():
    out = preprocess(_random_rgb(), size=32)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_clahe_increases_contrast_of_dark_image():
    dark = (_random_rgb(64, 64) * 0.2).astype(np.uint8)
    lum = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0]  # noqa: E731
    assert lum(apply_clahe(dark)).std() > lum(dark).std()


def test_crop_roi_clips_to_image():
    img = _random_rgb(40, 40)
    assert crop_roi(img, 5, 5, 30, 25).shape[:2] == (20, 25)
    assert crop_roi(img, -10, -10, 100, 100).shape[:2] == (40, 40)


def test_perturbations_keep_shape():
    batch = np.random.default_rng(0).random((4, 32, 32, 3)).astype(np.float32)
    for fn, levels in PERTURBATIONS.values():
        out = apply_to_batch(fn, batch, levels[-1])
        assert out.shape == batch.shape
        assert 0.0 <= out.min() and out.max() <= 1.0


def test_detects_red_disc():
    scene = np.full((240, 320, 3), 200, np.uint8)
    cv2.circle(scene, (160, 120), 40, (220, 20, 20), -1)
    cands = detect_signs(scene, use_hough=False)
    assert any(c.color == "rouge" and abs(c.x + c.w / 2 - 160) < 10 for c in cands)


def test_model_output_shape():
    from clocker.model import build_model

    model = build_model(32)
    out = model(np.zeros((2, 32, 32, 3), np.float32), training=False)
    assert out.shape == (2, config.NUM_CLASSES)
    assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-5)
