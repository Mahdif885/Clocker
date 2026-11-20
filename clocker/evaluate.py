"""Évaluation du modèle sur le jeu de test officiel GTSRB.

Produit dans reports/ :
    - metrics.json                  accuracy, macro-F1, latence, FPS
    - classification_report.csv     précision / rappel / F1 par classe
    - figures/confusion_matrix.png  matrice de confusion normalisée
    - figures/top_confusions.png    paires de classes les plus confondues
    - figures/robustness.png        accuracy vs intensité de perturbation

Exemple :
    python -m clocker.evaluate
    python -m clocker.evaluate --skip-robustness
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from clocker import config  # noqa: E402
from clocker.data import load_cache  # noqa: E402
from clocker.labels import class_name  # noqa: E402
from clocker.robustness import PERTURBATIONS, apply_to_batch  # noqa: E402


def predict(model: tf.keras.Model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return model.predict(x, batch_size=batch_size, verbose=0)


def measure_latency(model: tf.keras.Model, x: np.ndarray, n: int = 200) -> dict:
    """Latence image par image (cas embarqué : une image à la fois) et débit en lot."""
    single = tf.function(lambda t: model(t, training=False))
    sample = tf.convert_to_tensor(x[:1])
    for _ in range(10):  # échauffement
        single(sample)
    times = []
    for i in range(min(n, len(x))):
        t0 = time.perf_counter()
        single(tf.convert_to_tensor(x[i : i + 1]))
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    predict(model, x[:2000])
    batch_time = time.perf_counter() - t0
    ms = 1000 * float(np.median(times))
    return {
        "latence_ms_image": round(ms, 3),
        "fps_image_par_image": round(1000 / ms, 1),
        "fps_par_lot": round(min(2000, len(x)) / batch_time, 1),
    }


def plot_confusion(cm: np.ndarray, out_path) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title("Matrice de confusion normalisée (jeu de test GTSRB)")
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, k: int = 10) -> pd.DataFrame:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:k]
    rows = [
        {
            "reelle": int(t),
            "predite": int(p),
            "nom_reelle": class_name(t),
            "nom_predite": class_name(p),
            "nb_erreurs": int(off[t, p]),
        }
        for t, p in idx
        if off[t, p] > 0
    ]
    return pd.DataFrame(rows)


def plot_failures(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, out_path,
                  n: int = 16) -> None:
    """Affiche les erreurs les plus « confiantes » (les plus instructives)."""
    wrong = np.where(y != y_pred)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-conf[wrong])][:n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(np.clip(x[i], 0, 1))
        ax.set_title(f"vrai : {y[i]}\nprédit : {y_pred[i]} ({conf[i]:.0%})", fontsize=8)
    fig.suptitle("Erreurs les plus confiantes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def robustness(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, n: int = 3000) -> dict:
    """Accuracy pour chaque perturbation et chaque niveau, sur un sous-échantillon.

    Remarque : la perturbation est appliquée à l'image déjà prétraitée (cache).
    Pour l'étude CLAHE vs sans CLAHE sur images sombres, comparer deux modèles
    (python -m clocker.train --no-clahe) sur le même jeu perturbé.
    """
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    xs, ys = x[idx], y[idx]
    results = {}
    for name, (fn, levels) in PERTURBATIONS.items():
        accs = []
        for level in levels:
            xp = apply_to_batch(fn, xs, level)
            accs.append(float(accuracy_score(ys, predict(model, xp).argmax(1))))
        results[name] = {"niveaux": levels, "accuracy": accs}
        print(f"  {name:32s} " + "  ".join(f"{a:.3f}" for a in accs))
    return results


def plot_robustness(results: dict, out_path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        ax.plot(r["niveaux"], r["accuracy"], marker="o")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(model_path=config.MODEL_PATH, img_size: int = config.IMG_SIZE,
             use_clahe: bool = True, run_robustness: bool = True,
             data: dict | None = None) -> dict:
    model = tf.keras.models.load_model(model_path)
    data = data if data is not None else load_cache(img_size, use_clahe)
    x_test, y_test = data["x_test"], data["y_test"]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    probs = predict(model, x_test)
    y_pred, conf = probs.argmax(1), probs.max(1)

    labels = list(range(config.NUM_CLASSES))
    metrics = {
        "modele": str(model_path),
        "nb_images_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=labels,
                                   zero_division=0)),
        "nb_parametres": int(model.count_params()),
    }
    metrics.update(measure_latency(model, x_test))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0,
        target_names=[f"{i} - {class_name(i)}" for i in labels],
    )
    pd.DataFrame(report).T.to_csv(config.REPORTS_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion(cm, config.FIGURES_DIR / "confusion_matrix.png")
    top = top_confusions(cm)
    top.to_csv(config.REPORTS_DIR / "top_confusions.csv", index=False)
    print("\nConfusions les plus fréquentes :\n", top.to_string(index=False))
    plot_failures(x_test, y_test, y_pred, conf, config.FIGURES_DIR / "failures.png")

    if run_robustness:
        print("\nRobustesse (accuracy par niveau) :")
        rob = robustness(model, x_test, y_test)
        plot_robustness(rob, config.FIGURES_DIR / "robustness.png")
        metrics["robustesse"] = rob

    with open(config.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Évalue le CNN Clocker sur le test GTSRB")
    p.add_argument("--model", default=str(config.MODEL_PATH))
    p.add_argument("--size", type=int, default=config.IMG_SIZE)
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    a = p.parse_args()
    evaluate(a.model, a.size, not a.no_clahe, not a.skip_robustness)


if __name__ == "__main__":
    main()
