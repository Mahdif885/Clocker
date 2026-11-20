# 🚦 Clocker — Reconnaissance de panneaux de signalisation routière

Projet de fin de module **INFO-CV401 — Vision par Ordinateur & Deep Learning** (Sujet 3).

Sous-système de type ADAS qui reconnaît les panneaux de limitation de vitesse et de danger, puis alerte le conducteur sur le tableau de bord. Pipeline : prétraitement CLAHE (canal L, espace Lab), classification par CNN Keras sur les 43 classes de [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), détection HSV/Hough en bonus, démo Streamlit.

Le plan de travail, la répartition des rôles et le planning sont dans **[PLAN.md](PLAN.md)**.

## Lancer la démo (1 commande)

```bash
streamlit run app/streamlit_app.py
```

(nécessite le modèle entraîné `models/clocker_cnn.keras`, voir ci-dessous)

## Installation

Python 3.10 ou plus récent.

```bash
git clone https://github.com/Mahdif885/Clocker.git
cd Clocker
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### Données

1. Créer un token API Kaggle (Kaggle → Settings → API → *Create New Token*) et placer `kaggle.json` dans `~/.kaggle/` (Windows : `C:\Users\<nom>\.kaggle\`).
2. Télécharger le dataset (~600 Mo) dans `data/gtsrb/` :

```bash
python scripts/download_data.py
```

## Utilisation

| Étape | Commande | Sortie |
|---|---|---|
| Prétraitement + cache | `python -m clocker.data` | `data/cache/gtsrb_32_clahe.npz` |
| Entraînement | `python -m clocker.train` | `models/clocker_cnn.keras`, `reports/figures/clocker_cnn_curves.png` |
| Évaluation | `python -m clocker.evaluate` | `reports/metrics.json`, matrice de confusion, robustesse |
| Prédiction CLI | `python -m clocker.inference image.png` | classe + confiance |
| Démo | `streamlit run app/streamlit_app.py` | interface web |
| Tests | `pytest` | — |
| Style PEP8 | `ruff check .` | — |

Expériences d'ablation pour le rapport :

```bash
python -m clocker.train --name sans_clahe --no-clahe
python -m clocker.train --name sans_augment --no-augment
python -m clocker.train --name deux_blocs --blocks 2
python -m clocker.train --name taille_64 --size 64
```

> Pas de GPU ? Le dépôt fonctionne tel quel sur Google Colab : `!git clone …`, `!pip install -r requirements.txt`, puis les mêmes commandes avec `!`.

## Structure

```
clocker/
  config.py         chemins et hyperparamètres
  labels.py         noms des 43 classes + catégories (vitesse, danger, …)
  preprocessing.py  CLAHE Lab, recadrage ROI, redimensionnement       (Membre A)
  data.py           lecture GTSRB, découpage train/val, cache .npz    (Membre A)
  detection.py      masques HSV rouge/bleu, Hough, formes             (Membre B)
  model.py          architecture CNN + augmentation                   (Membre C)
  train.py          entraînement, callbacks, courbes                  (Membre C)
  robustness.py     flou, bruit, assombrissement                      (Membre D)
  evaluate.py       métriques, matrice de confusion, FPS              (Membre D)
  inference.py      prédiction sur une image
app/streamlit_app.py  démo                                            (Membre D)
notebooks/            exploration (Jalon 1)
scripts/              téléchargement des données
reports/              métriques et figures générées (pour le rapport)
tests/                tests unitaires
```

## Équipe

| Membre | Rôle |
|---|---|
| _à compléter_ | A — Données & prétraitement |
| _à compléter_ | B — Détection / localisation |
| _à compléter_ | C — CNN & entraînement |
| _à compléter_ | D — Évaluation & démo |
