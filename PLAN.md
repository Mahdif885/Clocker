# Plan de projet — Clocker

**Module :** INFO-CV401 — Vision par Ordinateur & Deep Learning
**Sujet 3 :** Reconnaissance de panneaux de signalisation routière (ADAS)
**Dataset :** [GTSRB sur Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) — ~39 000 images d'entraînement, ~12 600 de test, 43 classes

---

## 1. Objectif

Concevoir le sous-système embarqué qui reconnaît les panneaux de limitation de vitesse et de danger et alerte le conducteur sur le tableau de bord.

Le pipeline :

```
Image ──► [Détection HSV / Hough] ──► ROI ──► [CLAHE Lab + resize 32×32] ──► [CNN Keras] ──► classe + confiance ──► alerte tableau de bord
             (bonus, Membre B)                    (Membre A)                     (Membre C)            (Membre D)
```

### Objectifs chiffrés

| Indicateur | Cible |
|---|---|
| Accuracy sur le jeu de test officiel GTSRB | ≥ 95 % |
| Macro-F1 (toutes classes) | ≥ 0,93 |
| Accuracy avec flou gaussien modéré (σ = 1) | ≥ 85 % |
| Temps d'inférence par image (CPU) | < 5 ms (≥ 200 FPS en classification seule) |
| Taille du modèle | < 5 Mo (contrainte embarquée) |

## 2. Répartition des rôles

Le guide impose une responsabilité technique claire par membre. À compléter avec les noms :

| Membre | Responsabilité | Fichiers principaux | Critère du barème |
|---|---|---|---|
| **A** — _Nom_ | Ingestion des données, exploration, CLAHE, augmentation | `clocker/data.py`, `clocker/preprocessing.py`, `notebooks/01_exploration.ipynb` | Démarche & Prétraitement (4 pts) |
| **B** — _Nom_ | Détection/localisation : masques HSV rouge/bleu, `cv2.HoughCircles`, contours | `clocker/detection.py` | Algorithmes (6 pts) — bonus |
| **C** — _Nom_ | Architecture CNN, entraînement, courbes, lutte contre l'overfitting | `clocker/model.py`, `clocker/train.py` | Algorithmes & Performance (6 pts) |
| **D** — _Nom_ | Évaluation (matrice de confusion, robustesse, FPS), démo Streamlit, coordination du rapport | `clocker/evaluate.py`, `app/streamlit_app.py` | Interface & Démo (4 pts) |

Si l'équipe compte 3 membres : B et D fusionnent (la détection devient une partie de la démo).

**Tout le monde :** qualité du code (PEP8, docstrings), rédaction de sa partie du rapport, ses diapositives et son temps de parole (≈ 3 min chacun).

## 3. Planning (6 semaines)

### Semaine 0 — Mise en place (cette semaine)
- [ ] Chacun clone le dépôt, crée l'environnement (`pip install -r requirements.txt`)
- [ ] Chacun configure sa clé API Kaggle et lance `python scripts/download_data.py`
- [ ] Remplir les noms dans le tableau des rôles ci-dessus
- [ ] Convention Git : une branche par fonctionnalité (`feat/clahe`, `feat/cnn`…), Pull Request relue par un autre membre avant fusion dans `main`

### Jalon 1 — Semaines 1-2 : Données & Prétraitement
- [ ] **A** Exploration : distribution des classes (déséquilibre : de ~210 à ~2 250 images par classe), tailles d'images (15 à 250 px), exemples par classe
- [ ] **A** Recadrage sur la ROI fournie dans `Train.csv`, redimensionnement 32×32
- [ ] **A** CLAHE sur le canal L (Lab) : comparer `clipLimit` ∈ {1, 2, 3, 4}, figures avant/après sur des images sombres
- [ ] **A** Découpage train/validation stratifié (85/15), sauvegarde en cache `.npz`
- [ ] **B** Premiers masques HSV rouge (deux plages de teinte : 0-10 et 170-180) et bleu, tests sur quelques photos de rue
- [ ] **C** Lecture : LeNet-5, architectures de référence sur GTSRB ; brouillon du CNN
- [ ] **D** Squelette de l'app Streamlit (upload d'image, affichage) avec un modèle factice
- **Livrable du jalon :** notebook d'exploration exécuté + cache de données prêt

### Jalon 2 — Semaines 3-4 : CNN & Entraînement
- [ ] **C** Modèle de base : 3 blocs Conv-BN-ReLU-MaxPool (32/64/128 filtres), Dropout, Dense 256, softmax 43
- [ ] **C** Entraînement Adam, `EarlyStopping` + `ReduceLROnPlateau`, courbes loss/accuracy
- [ ] **C** Expériences comparatives (une ligne chacune dans le rapport) : avec/sans CLAHE, avec/sans augmentation, 32×32 vs 64×64, 2 vs 3 blocs
- [ ] **A** Augmentation : petites rotations (±10°), zoom, translation, luminosité — **pas de flip horizontal** (il transforme « virage à gauche » en « virage à droite »)
- [ ] **A/C** Pondération des classes (`class_weight`) contre le déséquilibre
- [ ] **B** Détection de cercles par Hough + triangles par `approxPolyDP`, génération des ROI candidates
- [ ] **D** Brancher le vrai modèle dans Streamlit, top-3 des prédictions
- **Livrable du jalon :** `models/clocker_cnn.keras` avec ≥ 95 % sur la validation

### Jalon 3 — Semaines 5-6 : Évaluation, Démo, Rapport
- [ ] **D** Évaluation sur le jeu de test officiel : accuracy, précision/rappel/F1 par classe, matrice de confusion normalisée
- [ ] **D** Robustesse : courbes accuracy vs intensité pour flou gaussien, flou de mouvement, bruit gaussien, bruit sel/poivre, assombrissement
- [ ] **D** Mesure FPS / latence (classification seule et pipeline complet)
- [ ] **Tous** Analyse des cas d'échec : top-10 des confusions (ex. 30 ↔ 50 km/h, 80 ↔ fin de 80) avec images
- [ ] **B** Pipeline complet dans la démo : photo de rue → boîtes englobantes → classe
- [ ] **D** Alerte tableau de bord dans l'app (vitesse = bleu, danger = orange, interdiction = rouge)
- [ ] **Tous** Rapport PDF (6-10 pages), diapositives PDF, répétition chronométrée (12 + 5 + 3 min)

## 4. Livrables (rappel du guide)

| Livrable | Contenu | Responsable |
|---|---|---|
| Rapport technique PDF (6-10 p.) | Intro, méthodologie + schéma du pipeline, résultats (courbes, matrices, Accuracy, FPS), cas d'échec, conclusion + **tableau des contributions individuelles** | D coordonne, chacun écrit sa partie |
| Code source | Ce dépôt : code modulaire, `requirements.txt`, `README.md` avec démo en 1 commande | Tous |
| Support de soutenance | Diapositives PDF | Tous |

### Plan du rapport
1. Introduction — contexte ADAS, enjeux de sécurité, objectifs (½ p.)
2. Données — GTSRB, déséquilibre, variabilité de l'éclairage (1 p.)
3. Méthodologie — schéma du pipeline ; CLAHE (principe de l'égalisation adaptative limitée en contraste) ; détection HSV/Hough ; architecture CNN avec nombre de paramètres (2-3 p.)
4. Expérimentations & Résultats — courbes, tableau d'ablation, matrice de confusion, robustesse, FPS (2-3 p.)
5. Analyse des échecs (1 p.)
6. Conclusion & bilan individuel — tableau des contributions (½-1 p.)

## 5. Correspondance avec le barème

| Critère (pts) | Ce qui nous rapporte les points |
|---|---|
| Démarche & Prétraitement (4) | Justification de Lab vs RGB pour CLAHE, figures avant/après, recadrage ROI, HSV pour la détection |
| Algorithmes & Performance (6) | ≥ 95 % test, tableau d'ablation, métriques par classe, robustesse chiffrée |
| Interface & Démo live (4) | App Streamlit fluide, confiance + top-3, boîtes englobantes colorées, alerte tableau de bord |
| Qualité du code (3) | Modules séparés, PEP8 (`ruff`), docstrings, tests, démo en 1 commande |
| Soutenance (3) | 20 min chronométrées, temps de parole égal, chacun maîtrise sa partie |

## 6. Démo de soutenance (5 min)

1. Lancer `streamlit run app/streamlit_app.py`
2. Image nette de 50 km/h → prédiction + confiance
3. Même image, curseur de flou/bruit dans l'app → montrer quand le modèle décroche
4. Image sombre en contre-jour → avec/sans CLAHE
5. Photo de rue complète → détection + classification (bonus)

**Plan B :** préparer les images de démo dans `assets/demo/` et une capture vidéo au cas où l'ordinateur poserait problème.

## 7. Risques

| Risque | Parade |
|---|---|
| Entraînement trop lent sur les portables | 32×32, Google Colab (GPU gratuit), cache `.npz` |
| Confusions entre limitations de vitesse proches | Augmentation, 64×64 en option, analyse dédiée dans le rapport |
| La détection HSV/Hough est peu fiable sur des photos réelles | C'est un bonus : la démo principale reste la classification |
| Conflits Git | Une branche par fonctionnalité, PR petites et fréquentes |
| Fuite de données train/test | Le jeu de test officiel n'est utilisé **qu'une seule fois**, à la fin |
