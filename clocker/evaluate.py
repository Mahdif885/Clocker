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
    python -m clocker.evaluate --name sans_clahe --no-clahe   # résultats dans reports/sans_clahe/
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
from clocker.preprocessing import finish  # noqa: E402
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


# Perturbations dont l'intensité est une distance en pixels : on la met à l'échelle
# de la résolution d'entrée pour que 32×32 et 64×64 subissent le même flou relatif.
SCALED_WITH_SIZE = {"Flou gaussien (sigma)", "Flou de bougé (px)"}


def _to_model_input(x_raw: np.ndarray, use_clahe: bool, denoise: bool) -> np.ndarray:
    """Images brutes [0, 1] -> entrée du modèle (prétraitement après la perturbation)."""
    if not use_clahe and not denoise:
        return x_raw
    u8 = (np.clip(x_raw, 0, 1) * 255).astype(np.uint8)
    return np.stack([finish(im, use_clahe, denoise) for im in u8])


def robustness(model: tf.keras.Model, x_raw: np.ndarray, y: np.ndarray, use_clahe: bool,
               denoise: bool = False, n: int = 3000) -> dict:
    """Accuracy pour chaque perturbation et chaque niveau, sur un sous-échantillon.

    Comme sur un vrai véhicule, la perturbation (flou, bruit, obscurité) touche
    l'image brute de la caméra, AVANT le prétraitement : on perturbe donc les
    images non prétraitées, puis on applique CLAHE si le modèle l'utilise.
    Les niveaux de flou sont définis pour une image 32×32 et mis à l'échelle
    pour les autres résolutions.
    """
    scale = x_raw.shape[1] / 32
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(x_raw), size=min(n, len(x_raw)), replace=False)
    xs, ys = x_raw[idx], y[idx]
    results = {}
    for name, (fn, levels) in PERTURBATIONS.items():
        accs = []
        for level in levels:
            lv = level * scale if name in SCALED_WITH_SIZE else level
            xp = _to_model_input(apply_to_batch(fn, xs, lv), use_clahe, denoise)
            accs.append(float(accuracy_score(ys, predict(model, xp).argmax(1))))
        results[name] = {"niveaux": levels, "accuracy": accs}
        print(f"  {name:32s} " + "  ".join(f"{a:.3f}" for a in accs))
    return results


def plot_robustness(results: dict, out_path) -> None:
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (name, r) in zip(axes, results.items()):
        ax.plot(r["niveaux"], r["accuracy"], marker="o")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if name.endswith("(facteur)"):
            ax.invert_xaxis()  # de gauche à droite : de plus en plus dégradé
    for ax in axes[::cols]:
        ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def output_dirs(name: str):
    """Dossiers de sortie : reports/ pour le modèle principal, reports/<nom>/ sinon."""
    if name == config.MODEL_PATH.stem:
        return config.REPORTS_DIR, config.FIGURES_DIR
    out = config.REPORTS_DIR / name
    return out, out / "figures"


def evaluate(name: str = config.MODEL_PATH.stem, use_clahe: bool = True,
             run_robustness: bool = True, data: dict | None = None) -> dict:
    """Évalue models/<name>.keras ; la taille d'entrée est lue dans le modèle."""
    model_path = config.MODELS_DIR / f"{name}.keras"
    prep = config.load_prep(name, use_clahe)  # réglages enregistrés à l'entraînement
    use_clahe, denoise = prep["clahe"], prep["denoise"]
    reports_dir, figures_dir = output_dirs(name)
    model = tf.keras.models.load_model(model_path)
    img_size = model.input_shape[1]
    data = data if data is not None else load_cache(img_size, use_clahe, denoise)
    x_test, y_test = data["x_test"], data["y_test"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    probs = predict(model, x_test)
    y_pred, conf = probs.argmax(1), probs.max(1)

    labels = list(range(config.NUM_CLASSES))
    metrics = {
        "modele": name,
        "taille_entree": int(img_size),
        "clahe": use_clahe,
        "debruitage": denoise,
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
    pd.DataFrame(report).T.to_csv(reports_dir / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion(cm, figures_dir / "confusion_matrix.png")
    top = top_confusions(cm)
    top.to_csv(reports_dir / "top_confusions.csv", index=False)
    print("\nConfusions les plus fréquentes :\n", top.to_string(index=False))
    plot_failures(x_test, y_test, y_pred, conf, figures_dir / "failures.png")

    if run_robustness:
        print("\nRobustesse (accuracy par niveau) :")
        raw = load_cache(img_size, use_clahe=False)  # images non prétraitées
        rob = robustness(model, raw["x_test"], raw["y_test"], use_clahe, denoise)
        plot_robustness(rob, figures_dir / "robustness.png")
        metrics["robustesse"] = rob

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Évalue un modèle Clocker sur le test GTSRB")
    p.add_argument("--name", default=config.MODEL_PATH.stem,
                   help="nom de l'expérience (models/<nom>.keras), comme pour train")
    p.add_argument("--no-clahe", action="store_true",
                   help="modèle entraîné avec --no-clahe (inutile si models/<nom>.json existe)")
    p.add_argument("--skip-robustness", action="store_true")
    a = p.parse_args()
    evaluate(a.name, not a.no_clahe, not a.skip_robustness)


if __name__ == "__main__":
    main()
