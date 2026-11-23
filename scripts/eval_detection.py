"""Évaluation de la détection + classification sur des scènes complètes (GTSDB).

Mesure, sur des photos de route annotées :
  - le rappel du détecteur seul (couleur HSV, + Hough en option) : part des
    vrais panneaux pour lesquels au moins une zone candidate est trouvée ;
  - la précision et le rappel du système complet selon le seuil de confiance ;
  - le rappel par catégorie de panneau et par taille (petits panneaux lointains).

Une détection est correcte si son IoU avec un panneau annoté est ≥ 0,5.

Exemples :
    python scripts/eval_detection.py                    # couleur seule, 900 images
    python scripts/eval_detection.py --hough            # couleur + cercles de Hough
    python scripts/eval_detection.py --limit 100        # essai rapide
    python scripts/eval_detection.py --min-area 0.0001  # accepter des zones plus petites

Résultats dans reports/detection/<nom>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clocker import config, detection  # noqa: E402
from clocker.labels import class_category  # noqa: E402
from clocker.preprocessing import load_rgb  # noqa: E402
from clocker.scene import iou, recognise  # noqa: E402

GTSDB_DIR = config.DATA_DIR / "gtsdb"
IMG_EXT = {".ppm", ".png", ".jpg", ".jpeg"}
THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
SIZE_BINS = [("petits (< 32 px)", 0, 32), ("moyens (32-64 px)", 32, 64),
             ("grands (> 64 px)", 64, 10_000)]
CAT_COLORS = {"Limitation de vitesse": (29, 95, 191), "Danger": (215, 38, 61),
              "Interdiction": (215, 38, 61), "Obligation": (29, 95, 191),
              "Priorité": (245, 197, 24), "Fin d'interdiction": (154, 161, 169)}


def load_gtsdb(root: Path, images_dir: Path | None = None
                ) -> tuple[list[Path], dict[str, list[tuple]], Path]:
    """Images de scène + annotations {nom sans extension: [(x1, y1, x2, y2, classe)]}.

    L'archive Kaggle contient plusieurs dossiers dont les images portent les mêmes
    noms (00000.ppm...) : le dossier d'entraînement, décrit par gt.txt, et le dossier
    de test, sans annotations. On choisit donc explicitement le dossier : --images
    s'il est donné, sinon un dossier dont le nom contient « train », sinon celui
    qui couvre le plus d'images annotées. Les sous-dossiers 00 à 42 (panneaux déjà
    recadrés) sont ignorés.
    """
    gts = sorted(root.rglob("gt.txt"))
    if not gts:
        sys.exit(f"gt.txt introuvable sous {root}. Lancez : python scripts/download_gtsdb.py")
    ann: dict[str, list[tuple]] = {}
    for line in gts[0].read_text().splitlines():
        parts = line.strip().split(";")
        if len(parts) != 6:
            continue
        name, x1, y1, x2, y2, cls = parts
        ann.setdefault(Path(name).stem, []).append(
            (int(x1), int(y1), int(x2), int(y2), int(cls)))

    def is_scene(p: Path) -> bool:
        return (p.suffix.lower() in IMG_EXT and p.stem.isdigit() and len(p.stem) == 5
                and not (p.parent.name.isdigit() and len(p.parent.name) == 2))

    folders: dict[Path, dict[str, Path]] = {}
    search = images_dir if images_dir is not None else root
    for p in sorted(search.rglob("*")):
        if p.is_file() and is_scene(p):
            folders.setdefault(p.parent, {}).setdefault(p.stem, p)
    if not folders:
        listing = sorted({str(p.relative_to(root).parent) for p in root.rglob("*") if p.is_file()})
        sys.exit("Aucune image de scène (00000.ppm, 00001.ppm...) trouvée.\n"
                 f"Annotations lues dans : {gts[0]}\n"
                 "Dossiers contenant des fichiers : " + ", ".join(listing[:15]))

    print(f"Annotations : {gts[0]} ({sum(len(v) for v in ann.values())} panneaux, "
          f"{len(ann)} images annotées)")
    print("Dossiers d'images de scène trouvés :")
    for f, imgs in folders.items():
        print(f"  {f}  ({len(imgs)} images)")

    def score(item: tuple[Path, dict[str, Path]]) -> tuple[int, int]:
        f, imgs = item
        is_train = int("train" in str(f).lower())
        return is_train, sum(k in imgs for k in ann)

    chosen, imgs = max(folders.items(), key=score)
    print(f"Dossier utilisé : {chosen}"
          + ("" if images_dir else "  (autre choix : option --images <dossier>)"))
    images = [imgs[k] for k in sorted(imgs)]  # y compris les scènes sans panneau
    return images, ann, gts[0]


def match(dets: list, gts: list[tuple], thr: float) -> tuple[int, int, int, list[bool]]:
    """Appariement glouton par confiance décroissante (IoU ≥ 0,5).

    Renvoie (vrais positifs, bien classés, faux positifs, panneau annoté trouvé ?).
    """
    kept = sorted((d for d in dets if d.conf >= thr), key=lambda d: -d.conf)
    found = [False] * len(gts)
    tp = correct = fp = 0
    for d in kept:
        best, best_iou = -1, 0.5
        for j, g in enumerate(gts):
            if not found[j]:
                v = iou(d.box, g[:4])
                if v >= best_iou:
                    best, best_iou = j, v
        if best < 0:
            fp += 1
        else:
            found[best] = True
            tp += 1
            correct += int(d.class_id == gts[best][4])
    return tp, correct, fp, found


def size_bin(g: tuple) -> str:
    s = max(g[2] - g[0], g[3] - g[1])
    return next(name for name, lo, hi in SIZE_BINS if lo <= s < hi)


def draw(image: np.ndarray, gts: list[tuple], dets: list, thr: float) -> np.ndarray:
    out = image.copy()
    for g in gts:  # vérité terrain : blanc épais
        cv2.rectangle(out, g[:2], g[2:4], (255, 255, 255), 4)
    for d in dets:
        if d.conf < thr:
            continue
        x1, y1, x2, y2 = d.box
        color = CAT_COLORS.get(class_category(d.class_id), (60, 200, 60))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d.class_id} {d.conf:.0%}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Évalue la détection sur GTSDB")
    p.add_argument("--model", default=config.MODEL_PATH.stem)
    p.add_argument("--hough", action="store_true", help="ajoute les cercles de Hough")
    p.add_argument("--threshold", type=float, default=0.6, help="seuil d'alerte rapporté")
    p.add_argument("--min-area", type=float, default=None,
                   help=f"aire minimale relative d'un candidat (défaut "
                        f"{detection.MIN_AREA_RATIO})")
    p.add_argument("--limit", type=int, default=None, help="n'utiliser que N images")
    p.add_argument("--name", default=None, help="nom du dossier de résultats")
    p.add_argument("--images", default=None,
                   help="dossier des images de scène (par défaut : choisi automatiquement)")
    a = p.parse_args()

    if a.min_area is not None:  # réglage pour cette exécution seulement
        detection.MIN_AREA_RATIO = a.min_area

    import tensorflow as tf

    model = tf.keras.models.load_model(config.MODELS_DIR / f"{a.model}.keras")
    prep = config.load_prep(a.model)
    images, ann, gt_file = load_gtsdb(GTSDB_DIR, Path(a.images) if a.images else None)
    if a.limit:
        images = images[: a.limit]
    name = a.name or ("couleur_hough" if a.hough else "couleur")
    out_dir = config.REPORTS_DIR / "detection" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vérification visuelle : les boîtes annotées doivent entourer de vrais panneaux
    check = [p for p in images if p.stem in ann][:4]
    if check:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, path in zip(axes.ravel(), check):
            ax.imshow(draw(load_rgb(path), ann[path.stem], [], 1.0))
            ax.set_title(f"{path.name} : boîtes de {gt_file.name}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "verification_annotations.png", dpi=90)
        plt.close(fig)
        print(f"Vérification visuelle : {out_dir / 'verification_annotations.png'}")
    print(f"{len(images)} images, modèle {a.model}, détecteur : {name}")

    records, times = [], []
    for i, path in enumerate(images, 1):
        img = load_rgb(path)
        t0 = time.perf_counter()
        dets = recognise(model, img, prep["clahe"], prep["denoise"], 0.0, a.hough)
        times.append(time.perf_counter() - t0)
        records.append((path, ann.get(path.stem, []), dets))
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images, {np.mean(times) * 1000:.0f} ms/image")

    n_gt = sum(len(g) for _, g, _ in records)
    if n_gt == 0:
        sys.exit("Aucun panneau annoté dans ces images : vérifiez gt.txt.")

    # 1) Courbe selon le seuil
    rows = []
    for thr in THRESHOLDS:
        tp = correct = fp = 0
        for _, gts, dets in records:
            a_tp, a_c, a_fp, _ = match(dets, gts, thr)
            tp, correct, fp = tp + a_tp, correct + a_c, fp + a_fp
        n_det = tp + fp
        rows.append({
            "seuil": thr,
            "detections": n_det,
            "precision_localisation": tp / n_det if n_det else 0.0,
            "rappel_localisation": tp / n_gt,
            "precision_complete": correct / n_det if n_det else 0.0,
            "rappel_complet": correct / n_gt,
            "faux_positifs_par_image": fp / len(records),
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(out_dir / "seuils.csv", index=False)

    # 2) Rappel par catégorie et par taille, au seuil 0 (détecteur seul) et au seuil choisi
    per = []
    for _, gts, dets in records:
        _, _, _, found0 = match(dets, gts, 0.0)
        kept = [d for d in dets if d.conf >= a.threshold]
        for j, g in enumerate(gts):
            hit = [d for d in kept if iou(d.box, g[:4]) >= 0.5]
            best = max(hit, key=lambda d: d.conf) if hit else None
            per.append({"categorie": class_category(g[4]), "taille": size_bin(g),
                        "candidat_trouve": found0[j],
                        "detecte_au_seuil": best is not None,
                        "bien_classe": best is not None and best.class_id == g[4]})
    per = pd.DataFrame(per)

    def summary(col: str) -> pd.DataFrame:
        s = per.groupby(col).agg(
            panneaux=("candidat_trouve", "size"),
            detecteur_seul=("candidat_trouve", "mean"),
            detecte_au_seuil=("detecte_au_seuil", "mean"),
            detecte_et_bien_classe=("bien_classe", "mean"),
        ).reset_index()
        for c in ("detecteur_seul", "detecte_au_seuil", "detecte_et_bien_classe"):
            s[c] = (100 * s[c]).round(1)
        return s

    by_cat, by_size = summary("categorie"), summary("taille")
    by_cat.to_csv(out_dir / "par_categorie.csv", index=False)
    by_size.to_csv(out_dir / "par_taille.csv", index=False)

    at = curve.iloc[(curve["seuil"] - a.threshold).abs().argmin()]
    metrics = {
        "modele": a.model, "detecteur": name, "images": len(records), "panneaux_annotes": n_gt,
        "min_area_ratio": detection.MIN_AREA_RATIO,
        "rappel_detecteur_seul": float(per["candidat_trouve"].mean()),
        "seuil": float(at["seuil"]),
        "precision_complete": float(at["precision_complete"]),
        "rappel_complet": float(at["rappel_complet"]),
        "precision_localisation": float(at["precision_localisation"]),
        "rappel_localisation": float(at["rappel_localisation"]),
        "faux_positifs_par_image": float(at["faux_positifs_par_image"]),
        "temps_ms_par_image": round(1000 * float(np.median(times)), 1),
        "fps_scene": round(1 / float(np.median(times)), 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    # 3) Figures : courbe précision / rappel et exemples
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(curve["rappel_localisation"], curve["precision_localisation"], "o-",
            label="Localisation (panneau trouvé)")
    ax.plot(curve["rappel_complet"], curve["precision_complete"], "s-",
            label="Système complet (trouvé et bien classé)")
    for _, r in curve.iterrows():
        ax.annotate(f"{r['seuil']:g}", (r["rappel_complet"], r["precision_complete"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(loc="lower left")
    ax.set_title(f"GTSDB, détecteur {name} (étiquettes = seuil de confiance)")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_rappel.png", dpi=150)
    plt.close(fig)

    # Exemples : les images avec le plus de panneaux manqués, puis des réussites
    scored = []
    for path, gts, dets in records:
        if gts:
            _, c, fp, found = match(dets, gts, a.threshold)
            scored.append((len(gts) - c + fp, path, gts, dets))
    scored.sort(key=lambda s: -s[0])
    picks = scored[:4] + [s for s in scored if s[0] == 0][:2]
    if picks:
        n_rows = (len(picks) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.2 * n_rows))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, (err, path, gts, dets) in zip(np.atleast_1d(axes).ravel(), picks):
            ax.imshow(draw(load_rgb(path), gts, dets, a.threshold))
            ax.set_title(f"{path.name} : {err} erreur(s). Blanc = annoté, couleur = détecté",
                         fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "exemples.png", dpi=110)
        plt.close(fig)

    print("\n" + json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nPar catégorie (%) :\n" + by_cat.to_string(index=False))
    print("\nPar taille (%) :\n" + by_size.to_string(index=False))
    print(f"\nRésultats dans {out_dir}")


if __name__ == "__main__":
    main()
