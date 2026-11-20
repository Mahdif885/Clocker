"""Noms des 43 classes GTSRB et catégories utilisées pour l'alerte conducteur."""

CLASS_NAMES = {
    0: "Limitation 20 km/h",
    1: "Limitation 30 km/h",
    2: "Limitation 50 km/h",
    3: "Limitation 60 km/h",
    4: "Limitation 70 km/h",
    5: "Limitation 80 km/h",
    6: "Fin de limitation 80 km/h",
    7: "Limitation 100 km/h",
    8: "Limitation 120 km/h",
    9: "Interdiction de dépasser",
    10: "Interdiction de dépasser (poids lourds)",
    11: "Priorité à la prochaine intersection",
    12: "Route prioritaire",
    13: "Cédez le passage",
    14: "Stop",
    15: "Circulation interdite",
    16: "Accès interdit aux poids lourds",
    17: "Sens interdit",
    18: "Danger",
    19: "Virage dangereux à gauche",
    20: "Virage dangereux à droite",
    21: "Succession de virages",
    22: "Chaussée déformée",
    23: "Chaussée glissante",
    24: "Chaussée rétrécie à droite",
    25: "Travaux",
    26: "Feux tricolores",
    27: "Passage piétons",
    28: "Enfants",
    29: "Cyclistes",
    30: "Risque de neige ou verglas",
    31: "Passage d'animaux sauvages",
    32: "Fin de toutes les interdictions",
    33: "Obligation de tourner à droite",
    34: "Obligation de tourner à gauche",
    35: "Obligation d'aller tout droit",
    36: "Tout droit ou à droite",
    37: "Tout droit ou à gauche",
    38: "Contournement par la droite",
    39: "Contournement par la gauche",
    40: "Giratoire",
    41: "Fin d'interdiction de dépasser",
    42: "Fin d'interdiction de dépasser (poids lourds)",
}

SPEED_LIMIT = "Limitation de vitesse"
DANGER = "Danger"
PROHIBITION = "Interdiction"
MANDATORY = "Obligation"
PRIORITY = "Priorité"
END = "Fin d'interdiction"

_CATEGORIES = {
    SPEED_LIMIT: [0, 1, 2, 3, 4, 5, 7, 8],
    DANGER: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    PROHIBITION: [9, 10, 15, 16, 17],
    MANDATORY: [33, 34, 35, 36, 37, 38, 39, 40],
    PRIORITY: [12, 13, 14],
    END: [6, 32, 41, 42],
}

CLASS_CATEGORY = {cid: cat for cat, ids in _CATEGORIES.items() for cid in ids}

# Valeur de la limitation, pour l'affichage tableau de bord
SPEED_VALUES = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}


def class_name(class_id: int) -> str:
    """Retourne le nom lisible d'une classe."""
    return CLASS_NAMES.get(int(class_id), f"Classe {class_id}")


def class_category(class_id: int) -> str:
    """Retourne la catégorie (vitesse, danger, ...) d'une classe."""
    return CLASS_CATEGORY[int(class_id)]
