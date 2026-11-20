"""Télécharge GTSRB depuis Kaggle et le décompresse dans data/gtsrb/.

Prérequis : un compte Kaggle et le fichier ~/.kaggle/kaggle.json
(Kaggle > Settings > API > Create New Token).

    python scripts/download_data.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "gtsrb"
DATASET = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"


def main() -> None:
    if (DEST / "Train.csv").exists():
        print(f"Déjà présent : {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {DATASET} (~600 Mo)...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET,
         "-p", str(DEST), "--unzip"],
        check=True,
    )
    if not (DEST / "Train.csv").exists():
        sys.exit(f"Train.csv absent de {DEST} : vérifiez le contenu téléchargé.")
    print(f"OK : {DEST}")


if __name__ == "__main__":
    main()
