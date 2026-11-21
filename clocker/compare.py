"""Tableau comparatif des expériences (ablation) pour le rapport.

Lit reports/metrics.json et reports/<nom>/metrics.json, puis écrit
reports/comparaison.csv et reports/comparaison.md.

    python -m clocker.compare
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clocker import config

# Niveaux de perturbation repris dans le tableau (nom, indice dans PERTURBATIONS)
ROBUST_COLS = {
    "Flou σ=1,5": ("Flou gaussien (sigma)", 3),
    "Flou σ=2": ("Flou gaussien (sigma)", 4),
    "Bougé 7 px": ("Flou de bougé (px)", 3),
    "Bruit 0,1": ("Bruit gaussien (écart-type)", 3),
    "Sombre ×0,25": ("Assombrissement (facteur)", 4),
    "Contraste ×0,2": ("Faible contraste (facteur)", 3),
    "Contraste ×0,1": ("Faible contraste (facteur)", 4),
    "Ombre ×0,1": ("Éclairage inégal (facteur)", 4),
}


def collect() -> pd.DataFrame:
    files = [config.REPORTS_DIR / "metrics.json"]
    # Le modèle principal est dans reports/ : on ignore une ancienne copie dans reports/<nom>/
    files += sorted(f for f in config.REPORTS_DIR.glob("*/metrics.json")
                    if f.parent.name != config.MODEL_PATH.stem)
    rows = []
    for f in files:
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        row = {
            "Expérience": Path(m.get("modele", f.parent.name)).stem,
            "Taille": m.get("taille_entree", ""),
            "CLAHE": "oui" if m.get("clahe", True) else "non",
            "Débruit.": "oui" if m.get("debruitage", False) else "non",
            "Paramètres": m["nb_parametres"],
            "Accuracy test": round(100 * m["accuracy"], 2),
            "Macro-F1": round(100 * m["macro_f1"], 2),
            "Latence (ms)": m["latence_ms_image"],
        }
        rob = m.get("robustesse", {})
        for col, (key, idx) in ROBUST_COLS.items():
            if key in rob:
                row[col] = round(100 * rob[key]["accuracy"][idx], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = collect()
    if df.empty:
        print("Aucun metrics.json trouvé : lancez d'abord python -m clocker.evaluate")
        return
    df.to_csv(config.REPORTS_DIR / "comparaison.csv", index=False)
    (config.REPORTS_DIR / "comparaison.md").write_text(df.to_markdown(index=False),
                                                       encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
