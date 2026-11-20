"""Architecture du CNN de classification (TensorFlow / Keras).

Choix volontairement simple, comme demandé par le sujet :
3 blocs [Conv -> BatchNorm -> ReLU -> Conv -> ReLU -> MaxPool -> Dropout],
puis Dense + Dropout + softmax. ~0,6 M paramètres pour une entrée 32×32,
ce qui reste compatible avec un système embarqué.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from clocker import config


def _conv_block(x, filters: int, dropout: float, name: str):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return layers.Dropout(dropout, name=f"{name}_drop")(x)


def build_augmentation() -> tf.keras.Sequential:
    """Augmentation appliquée seulement pendant l'entraînement.

    Pas de flip horizontal : il transformerait « virage à gauche » en
    « virage à droite » (classes 19/20, 33/34, 36/37, 38/39).
    """
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.04),             # ±~15°
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(
    img_size: int = config.IMG_SIZE,
    num_classes: int = config.NUM_CLASSES,
    filters: tuple[int, ...] = (32, 64, 128),
    augment: bool = True,
) -> tf.keras.Model:
    """Construit le CNN. `filters` fixe le nombre de blocs (2 ou 3 pour le sujet)."""
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = build_augmentation()(inputs) if augment else inputs
    for i, f in enumerate(filters, start=1):
        x = _conv_block(x, f, dropout=0.25, name=f"block{i}")
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dense_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return models.Model(inputs, outputs, name="clocker_cnn")


def compile_model(model: tf.keras.Model, lr: float = config.LEARNING_RATE) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
