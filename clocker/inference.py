"""Prédiction sur une image quelconque (utilisé par la démo et en ligne de commande).

Exemple :
    python -m clocker.inference chemin/vers/image.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from clocker import config
from clocker.labels import class_category, class_name
from clocker.preprocessing import load_rgb, preprocess


@dataclass
class Prediction:
    class_id: int
    name: str
    category: str
    confidence: float
    top_k: list[tuple[int, str, float]]


def load_model(path=config.MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def predict_image(model: tf.keras.Model, image_rgb: np.ndarray, use_clahe: bool = True,
                  denoise: bool = False, k: int = 3) -> Prediction:
    size = model.input_shape[1]
    x = preprocess(image_rgb, size=size, use_clahe=use_clahe, denoise=denoise)[None]
    probs = model(x, training=False).numpy()[0]
    order = np.argsort(probs)[::-1][:k]
    top = [(int(i), class_name(i), float(probs[i])) for i in order]
    best = int(order[0])
    return Prediction(best, class_name(best), class_category(best), float(probs[best]), top)


def main() -> None:
    p = argparse.ArgumentParser(description="Prédit le panneau présent sur une image")
    p.add_argument("image")
    p.add_argument("--model", default=str(config.MODEL_PATH))
    a = p.parse_args()
    prep = config.load_prep(Path(a.model).stem)
    pred = predict_image(load_model(a.model), load_rgb(a.image), prep["clahe"], prep["denoise"])
    print(f"{pred.name}  [{pred.category}]  confiance = {pred.confidence:.1%}")
    for cid, name, pr in pred.top_k:
        print(f"  {cid:2d}  {name:45s} {pr:.1%}")


if __name__ == "__main__":
    main()
