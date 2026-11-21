"""Entraînement du CNN.

Exemple :
    python -m clocker.train                      # configuration par défaut
    python -m clocker.train --epochs 5 --no-clahe  # expérience d'ablation
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from clocker import config  # noqa: E402
from clocker.data import load_cache  # noqa: E402
from clocker.model import build_model, compile_model  # noqa: E402


def plot_history(history: dict, out_path) -> None:
    """Sauvegarde les courbes loss / accuracy (train vs validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Perte", "Accuracy"]):
        ax.plot(history[key], label="entraînement")
        ax.plot(history[f"val_{key}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Époque")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    img_size: int = config.IMG_SIZE,
    use_clahe: bool = True,
    denoise: bool = False,
    augment: bool = True,
    use_class_weight: bool = True,
    filters: tuple[int, ...] = (32, 64, 128),
    run_name: str = "clocker_cnn",
    data: dict | None = None,
) -> tf.keras.Model:
    """Entraîne le modèle et sauvegarde modèle, courbes et historique."""
    tf.keras.utils.set_random_seed(config.SEED)
    data = data if data is not None else load_cache(img_size, use_clahe, denoise)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    class_weight = None
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes.tolist(), weights.tolist()))

    model = compile_model(build_model(img_size, filters=filters, augment=augment))
    model.summary()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"{run_name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(model_path)
    config.save_prep(run_name, size=img_size, clahe=use_clahe, denoise=denoise,
                     augment=augment, blocks=len(filters))

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.REPORTS_DIR / f"{run_name}_history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, config.FIGURES_DIR / f"{run_name}_curves.png")

    best = max(hist["val_accuracy"])
    print(f"\nMeilleure accuracy validation : {best:.4f} — modèle : {model_path}")
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Entraîne le CNN Clocker sur GTSRB")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--size", type=int, default=config.IMG_SIZE)
    p.add_argument("--blocks", type=int, default=3, choices=[2, 3])
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--denoise", action="store_true", help="débruitage bilatéral avant CLAHE")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--name", default="clocker_cnn", help="nom de l'expérience")
    a = p.parse_args()
    train(
        epochs=a.epochs,
        batch_size=a.batch_size,
        img_size=a.size,
        use_clahe=not a.no_clahe,
        denoise=a.denoise,
        augment=not a.no_augment,
        use_class_weight=not a.no_class_weight,
        filters=(32, 64, 128)[: a.blocks],
        run_name=a.name,
    )


if __name__ == "__main__":
    main()
