"""Télécharge GTSDB (German Traffic Sign Detection Benchmark) dans data/gtsdb/.

Même prérequis que pour GTSRB : ~/.kaggle/kaggle.json.

    python scripts/download_gtsdb.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "gtsdb"
DATASET = "safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb"


def has_images() -> bool:
    return any(p.suffix.lower() in {".ppm", ".png", ".jpg"} for p in DEST.rglob("0*.*"))


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not (any(DEST.rglob("gt.txt")) and has_images()) and not list(DEST.glob("*.zip")):
        print(f"Téléchargement de {DATASET} (volumineux : images non compressées)...")
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET,
             "-p", str(DEST)],
            check=True,
        )
    # Décompression manuelle : la CLI Kaggle saute l'étape si le zip existe déjà
    for archive in DEST.glob("*.zip"):
        if not has_images():
            print(f"Décompression de {archive.name}...")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(DEST)
    gts = list(DEST.rglob("gt.txt"))
    if not gts or not has_images():
        gt_state = "trouvé" if gts else "absent"
        img_state = "trouvées" if has_images() else "absentes"
        sys.exit(f"Données incomplètes dans {DEST} : gt.txt {gt_state}, images {img_state}.")
    n = sum(1 for p in DEST.rglob("0*.*") if p.suffix.lower() in {".ppm", ".png", ".jpg"})
    print(f"OK : annotations dans {gts[0].parent}, {n} fichiers image.")
    print("Vous pouvez supprimer le .zip pour libérer de la place.")


if __name__ == "__main__":
    main()
