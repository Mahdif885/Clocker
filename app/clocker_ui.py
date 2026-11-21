# ruff: noqa: E501  (feuille de style CSS intégrée : lignes longues assumées)
"""Clocker — interface de démonstration (thème « béton et bitume », palette grise).

Lancement :
    python -m streamlit run app/clocker_ui.py

Cette interface n'utilise que les fonctions existantes du paquet `clocker`
(prétraitement, détection, labels, perturbations) : elle ne modifie rien.
L'ancienne démo `app/streamlit_app.py` reste disponible.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clocker import config  # noqa: E402
from clocker.detection import BLUE_RANGES, RED_RANGES, color_mask, crop, detect_signs  # noqa: E402
from clocker.labels import (  # noqa: E402
    DANGER,
    END,
    MANDATORY,
    PRIORITY,
    PROHIBITION,
    SPEED_LIMIT,
    SPEED_VALUES,
    class_category,
    class_name,
)
from clocker.preprocessing import finish, load_rgb, preprocess  # noqa: E402
from clocker.robustness import PERTURBATIONS  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette : gris béton et bitume + couleurs réelles de la signalisation
# --------------------------------------------------------------------------- #
ASPHALT = "#D4D7DB"     # fond de page : béton clair (grain par-dessus)
TARMAC = "#E6E8EB"      # cartes
KERB = "#B3B8BE"        # bordures, pistes des jauges
MARKING = "#26292D"     # texte principal : graphite
MUTED = "#5D636B"       # texte secondaire
GRAPHITE = "#454A51"    # accent neutre : onglet actif, liens, contours
YELLOW = "#F5C518"      # marquage jaune, actions, accent principal
RED = "#D7263D"         # bordure des panneaux de danger et d'interdiction
BLUE = "#1D5FBF"        # panneaux de direction et d'obligation
GREEN = "#1F7A4D"       # cartouche de route européenne

CATEGORY_COLORS = {
    SPEED_LIMIT: BLUE, DANGER: RED, PROHIBITION: RED,
    MANDATORY: BLUE, PRIORITY: YELLOW, END: MUTED,
}
BOX_RGB = {
    SPEED_LIMIT: (29, 95, 191), DANGER: (215, 38, 61), PROHIBITION: (215, 38, 61),
    MANDATORY: (29, 95, 191), PRIORITY: (245, 197, 24), END: (154, 161, 169),
}
SCALED_WITH_SIZE = {"Flou gaussien (sigma)", "Flou de bougé (px)"}

st.set_page_config(page_title="Clocker", page_icon="🛣️", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700;800&display=swap');

/* ---------- fond : enrobé avec grain ---------- */
.stApp {
  font-family: 'Barlow', 'Segoe UI', system-ui, sans-serif;
  color: %MARKING%;
  background-color: %ASPHALT%;
  background-image:
    radial-gradient(rgba(255,255,255,.45) 1px, transparent 1.2px),
    radial-gradient(rgba(0,0,0,.045) 1px, transparent 1.2px),
    radial-gradient(ellipse at 50% -10%, rgba(255,255,255,.55), transparent 55%);
  background-size: 4px 4px, 7px 7px, 100% 100%;
  background-position: 0 0, 2px 3px, 0 0;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.6rem; max-width: 1440px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700;
  color: %MARKING%;
}
[data-testid="stCaptionContainer"], .stCaption { color: %MUTED% !important; }

/* ---------- barre latérale : bas-côté ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #C9CDD2, #BEC2C8);
  border-right: 6px solid transparent;
  border-image: repeating-linear-gradient(180deg, #FFFFFF 0 26px, %GRAPHITE% 26px 44px) 6;
}
[data-testid="stSidebar"] h3 { font-size: 1.25rem; margin-top: .6rem; }

/* ---------- onglets : marquage au sol ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 2px dashed %KERB%; }
.stTabs [data-baseweb="tab"] { padding: .45rem .95rem; border-radius: 6px 6px 0 0; }
.stTabs [data-baseweb="tab"] p {
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.12rem;
  color: %MUTED%;
}
.stTabs [data-baseweb="tab"]:hover p { color: %MARKING%; }
.stTabs [aria-selected="true"] { background: rgba(69,74,81,.08); }
.stTabs [aria-selected="true"] p { color: %MARKING%; }
.stTabs [data-baseweb="tab-highlight"] { background: %GRAPHITE%; height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- en-tête : route en perspective ---------- */
.ck-hero { position: relative; display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
  min-height: 190px; border-radius: 14px; overflow: hidden; margin-bottom: 1.3rem;
  background: linear-gradient(180deg, #B9BEC4 0%, #A7ADB4 46%, #969CA3 60%, #7D838A 100%);
  box-shadow: 0 0 0 1px %KERB%, 0 14px 30px rgba(40,44,50,.18); }
.ck-hero-text { position: relative; z-index: 2; padding: 1.5rem 1.8rem;
  display: flex; flex-direction: column; justify-content: center; gap: .55rem;
  background: linear-gradient(90deg, rgba(222,225,229,.96) 55%, transparent); }
.ck-word { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 800;
  font-size: clamp(2.8rem, 5vw, 4.2rem); line-height: .9; letter-spacing: .01em; color: %MARKING%; }

.ck-sub { font-size: 1.05rem; color: #3C4147; max-width: 34ch; }
.ck-road { position: absolute; inset: 0 0 0 30%; perspective: 230px; perspective-origin: 50% 0%;
  overflow: hidden; }
.ck-road::before { /* lueur de l'horizon */
  content: ""; position: absolute; left: 0; right: 0; top: 30%; height: 40px;
  background: radial-gradient(ellipse at 50% 100%, rgba(255,255,255,.45), transparent 70%); }
.ck-lanes { position: absolute; left: 50%; bottom: -10px; width: 460px; height: 520px;
  margin-left: -230px; transform-origin: 50% 100%; transform: rotateX(72deg);
  background:
    linear-gradient(90deg, transparent 6%, #F4F4F0 6% 8%, transparent 8% 92%, #F4F4F0 92% 94%, transparent 94%),
    repeating-linear-gradient(180deg, #F4F4F0 0 38px, transparent 38px 80px) 50% 0 / 10px 100% no-repeat,
    #4E5258;
  animation: ck-drive 1.1s linear infinite; }
@keyframes ck-drive { from { background-position: 0 0, 50% 0, 0 0; }
                      to   { background-position: 0 0, 50% 80px, 0 0; } }
.ck-post { position: absolute; bottom: 14%; width: 10px; height: 46px; border-radius: 5px 5px 1px 1px;
  background: linear-gradient(180deg, #fff 0 18%, #111 18% 30%, #fff 30%); opacity: .9; }
.ck-post::after { content: ""; position: absolute; top: 4px; left: 2px; width: 6px; height: 4px;
  background: #FF9D1A; border-radius: 1px; box-shadow: 0 0 8px #FF9D1A; }

.ck-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .35rem; }
.ck-badge { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1rem;
  padding: .22rem .65rem .18rem; border-radius: 5px; color: #fff; white-space: nowrap; }
.ck-badge.e { background: %GREEN%; box-shadow: inset 0 0 0 2px %GREEN%, inset 0 0 0 4px #fff; }
.ck-badge.a { background: %RED%; box-shadow: inset 0 0 0 2px %RED%, inset 0 0 0 4px #fff; }
.ck-badge.n { background: %BLUE%; box-shadow: inset 0 0 0 2px %BLUE%, inset 0 0 0 4px #fff; }
.ck-badge.d { background: %YELLOW%; color: #111; box-shadow: inset 0 0 0 2px %YELLOW%, inset 0 0 0 4px #111; }

/* ---------- titres de section : plot réfléchissant ---------- */
.ck-h { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.3rem;
  color: %MARKING%; display: flex; align-items: center; gap: .55rem; margin: .2rem 0 .6rem; }
.ck-h::before { content: ""; width: 9px; height: 9px; border-radius: 2px; background: #FF9D1A;
  box-shadow: 0 0 0 2px rgba(255,255,255,.7), 0 0 6px rgba(255,157,26,.6); flex: none; }
.ck-lead { color: #3C4147; max-width: 95ch; }

/* ---------- panneau de résultat (tableau de bord) ---------- */
.ck-hud { border-radius: 16px; padding: 7px; background: %BLUE%;
  box-shadow: 0 16px 34px rgba(30,35,42,.22); }
.ck-hud.uncertain { background: repeating-linear-gradient(-45deg, %YELLOW% 0 16px, #33373C 16px 32px);
  box-shadow: 0 16px 34px rgba(30,35,42,.22);
  animation: ck-hazard 1s steps(2) 3; }
@keyframes ck-hazard { 50% { filter: brightness(.55); } }
.ck-hud-in { border: 3px solid #fff; border-radius: 11px; padding: 1.1rem 1.2rem 1rem;
  background: %BLUE%; display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 1.1rem 1.3rem;
  align-items: center; }
.ck-hud.uncertain .ck-hud-in { background: #4A4F56; }
.ck-sign { position: relative; width: 132px; height: 132px; border-radius: 12px; overflow: hidden;
  background: radial-gradient(circle at 50% 40%, #fff, #E9ECEF);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 0 3px rgba(255,255,255,.25), 0 0 28px rgba(255,255,255,.18); }
.ck-sign img { width: 112px; height: 112px; object-fit: contain;
  animation: ck-approach .9s cubic-bezier(.2,.8,.2,1) both; }
.ck-sign::after { /* reflet des phares sur le film rétroréfléchissant */
  content: ""; position: absolute; inset: -20%;
  background: linear-gradient(105deg, transparent 38%, rgba(255,255,255,.9) 50%, transparent 62%);
  transform: translateX(-110%); animation: ck-sweep 1.1s .55s ease-out forwards; }
@keyframes ck-approach { from { transform: scale(.3); filter: blur(5px) brightness(.35); opacity: .2; } }
@keyframes ck-sweep { to { transform: translateX(110%); } }
.ck-title { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 800;
  font-size: clamp(1.8rem, 2.5vw, 2.7rem); line-height: 1; color: #fff; overflow-wrap: normal;
  word-break: normal; hyphens: none; }
.ck-title.long { font-size: clamp(1.45rem, 1.8vw, 1.9rem); }
.ck-cat { color: rgba(255,255,255,.82); margin-top: .35rem; font-size: 1rem; }
.ck-gauge { grid-column: 1 / -1; display: flex; align-items: center; gap: 1rem;
  border-top: 1px solid rgba(255,255,255,.22); padding-top: .8rem; }
.ck-gauge svg { width: 168px; flex: none; }
.ck-gauge .needle { transform-origin: 100px 100px; animation: ck-needle 1.2s cubic-bezier(.3,1.4,.5,1) both; }
.ck-gauge .arc { animation: ck-arc 1.2s ease-out both; }
@keyframes ck-needle { from { transform: rotate(-90deg); } }
@keyframes ck-arc { from { stroke-dashoffset: 251.3; } }
.ck-verdict { color: #fff; font-size: 1rem; line-height: 1.35; }
.ck-verdict b { font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; display: block; }
.ck-hud.compact .ck-hud-in { padding: .8rem; gap: .8rem; }
.ck-hud.compact .ck-sign { width: 90px; height: 90px; }
.ck-hud.compact .ck-sign img { width: 76px; height: 76px; }
.ck-hud.compact .ck-title { font-size: 1.6rem; }
.ck-hud.compact .ck-gauge svg { width: 132px; }
.ck-hud.compact .ck-verdict { display: none; }
.ck-meta { color: %MUTED%; font-size: .88rem; margin-top: .7rem; }

/* ---------- hypothèses : jauges horizontales ---------- */
.ck-top { display: flex; flex-direction: column; gap: .7rem; }
.ck-row { display: grid; grid-template-columns: 40px minmax(0, 1fr) auto; gap: .65rem; align-items: center; }
.ck-row img { width: 40px; height: 40px; object-fit: contain; background: #fff; border-radius: 7px; padding: 3px;
  box-shadow: 0 0 0 1px %KERB%; }
.ck-row .lbl { font-size: .93rem; color: %MARKING%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ck-row .trk { height: 9px; background: %KERB%; border-radius: 5px; margin-top: .3rem; overflow: hidden; }
.ck-row .fil { height: 100%; border-radius: 5px; background: #8A9098;
  transform-origin: left; animation: ck-grow .8s cubic-bezier(.2,.8,.2,1) both; }
.ck-row:first-child .fil { background: %GRAPHITE%; }
.ck-row .pct { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.05rem;
  text-align: right; color: %MARKING%; white-space: nowrap; min-width: 54px; }
.ck-row:nth-child(2) .fil { animation-delay: .08s; } .ck-row:nth-child(3) .fil { animation-delay: .16s; }
.ck-row:nth-child(4) .fil { animation-delay: .24s; } .ck-row:nth-child(5) .fil { animation-delay: .32s; }
@keyframes ck-grow { from { transform: scaleX(0); } }

/* ---------- étapes : bornes kilométriques ---------- */
.ck-step { display: flex; align-items: center; gap: .6rem; margin-bottom: .45rem;
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.12rem; color: %MARKING%; }
.ck-pk { width: 30px; height: 40px; flex: none; border-radius: 15px 15px 3px 3px;
  background: linear-gradient(180deg, %RED% 0 40%, #F4F4F0 40%); color: #111;
  display: flex; align-items: flex-end; justify-content: center; padding-bottom: 3px;
  font-weight: 700; font-size: 1.05rem; box-shadow: 0 0 0 1px %KERB%, 0 3px 0 rgba(0,0,0,.18); }

/* ---------- cartes, panneaux d'information ---------- */
.ck-panel { background: %BLUE%; border-radius: 12px; padding: 5px; }
.ck-panel > div { border: 2px solid #fff; border-radius: 8px; padding: .8rem 1rem; color: #fff; }
.ck-panel small { opacity: .85; font-size: .9rem; }
.ck-panel b { display: block; font-family: 'Barlow Condensed', sans-serif; font-weight: 800;
  font-size: 2.1rem; line-height: 1.05; }
.ck-panel.green { background: %GREEN%; } .ck-panel.red { background: %RED%; }
.ck-panel.yellow { background: %YELLOW%; } .ck-panel.yellow > div { border-color: #111; color: #111; }
.ck-info { background: %TARMAC%; border: 1px solid %KERB%; border-left: 6px solid %GRAPHITE%;
  border-radius: 10px; padding: 1.1rem 1.3rem; color: #3C4147; }
.ck-info b { font-family: 'Barlow Condensed', sans-serif; font-size: 1.35rem; color: %MARKING%;
  display: block; margin-bottom: .25rem; }

.ck-chip { display: flex; gap: .6rem; align-items: center; padding: .45rem .5rem;
  border-radius: 8px; background: %TARMAC%; border: 1px solid %KERB%; margin-bottom: .4rem; }
.ck-chip img { width: 36px; height: 36px; object-fit: contain; border-radius: 5px; background: #fff; padding: 2px;
  box-shadow: 0 0 0 1px %KERB%; }
.ck-chip b { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.05rem; }
.ck-chip span { color: %MUTED%; font-size: .85rem; }

.ck-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(124px, 1fr)); gap: .8rem; }
.ck-cell { background: %TARMAC%; border: 1px solid %KERB%; border-radius: 10px; padding: .7rem .5rem;
  text-align: center; font-size: .84rem; line-height: 1.2; color: #3C4147;
  transition: transform .15s ease, border-color .15s ease; }
.ck-cell:hover { transform: translateY(-3px); border-color: %GRAPHITE%;
  box-shadow: 0 8px 18px rgba(30,35,42,.12); }
.ck-cell img { width: 62px; height: 62px; object-fit: contain; display: block; margin: 0 auto .5rem;
  background: #fff; border-radius: 8px; padding: 4px; }
.ck-cell small { color: %MUTED%; }

[data-testid="stImage"] img { border-radius: 10px; }

@media (max-width: 900px) {
  .ck-hero { grid-template-columns: 1fr; }
  .ck-road { inset: 0; opacity: .35; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""
for _k, _v in dict(ASPHALT=ASPHALT, TARMAC=TARMAC, KERB=KERB, MARKING=MARKING, MUTED=MUTED,
                   GRAPHITE=GRAPHITE, YELLOW=YELLOW, RED=RED, BLUE=BLUE, GREEN=GREEN).items():
    CSS = CSS.replace(f"%{_k}%", _v)
st.markdown(CSS, unsafe_allow_html=True)

def _night_theme() -> dict:
    return {"config": {
        "background": "transparent",
        "view": {"stroke": None},
        "font": "Barlow",
        "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": KERB,
                 "domainColor": KERB, "tickColor": KERB, "labelFontSize": 12,
                 "titleFontSize": 13},
        "legend": {"labelColor": MARKING, "labelFontSize": 13},
    }}


try:  # Altair ≥ 5.5
    alt.theme.register("ck_night", enable=True)(_night_theme)
except AttributeError:  # anciennes versions
    alt.themes.register("ck_night", _night_theme)
    alt.themes.enable("ck_night")

# --------------------------------------------------------------------------- #
# Chargements mis en cache
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model(name: str):
    import tensorflow as tf
    return tf.keras.models.load_model(config.MODELS_DIR / f"{name}.keras")


@st.cache_resource(show_spinner=False)
def get_cam_model(name: str):
    """Sous-modèle pour Grad-CAM : dernière couche de convolution + avant-dernière couche."""
    import tensorflow as tf
    model = get_model(name)
    conv = [lay for lay in model.layers if isinstance(lay, tf.keras.layers.Conv2D)][-1]
    head = model.layers[-1]  # Dense softmax : on recalcule les logits à la main
    sub = tf.keras.Model(model.inputs, [conv.output, model.layers[-2].output])
    w, b = head.get_weights()
    return sub, w, b, conv.name


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    return sorted(p.stem for p in config.MODELS_DIR.glob("*.keras"))


@st.cache_data(show_spinner=False)
def test_index() -> pd.DataFrame | None:
    csv = config.RAW_DIR / "Test.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["abs_path"] = df["Path"].apply(lambda p: str(config.RAW_DIR / p))
    return df


@st.cache_data(show_spinner=False)
def pictogram_uri(class_id: int) -> str:
    """Pictogramme officiel GTSRB (dossier Meta/), sinon un dessin SVG simplifié."""
    png = config.RAW_DIR / "Meta" / f"{class_id}.png"
    if png.exists():
        return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    svg = sign_svg(class_id)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def sign_svg(cid: int) -> str:
    """Dessin de secours quand le dossier Meta/ est absent."""
    cat = class_category(cid)
    v = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    if cid in SPEED_VALUES:
        n = SPEED_VALUES[cid]
        fs = 40 if n < 100 else 32
        return (v + f'<circle cx="50" cy="50" r="46" fill="{RED}"/>'
                f'<circle cx="50" cy="50" r="34" fill="#fff"/>'
                f'<text x="50" y="50" dy=".35em" text-anchor="middle" font-size="{fs}" '
                f'font-family="Barlow Condensed, Arial Narrow, sans-serif" font-weight="700" '
                f'fill="#111">{n}</text></svg>')
    if cid == 14:
        pts = " ".join(f"{50 + 46 * np.cos(np.pi / 8 + k * np.pi / 4):.1f},"
                       f"{50 + 46 * np.sin(np.pi / 8 + k * np.pi / 4):.1f}" for k in range(8))
        return (v + f'<polygon points="{pts}" fill="{RED}"/><text x="50" y="50" dy=".35em" '
                'text-anchor="middle" font-size="24" font-family="Arial" font-weight="700" '
                'fill="#fff">STOP</text></svg>')
    if cid == 13:
        return (v + f'<polygon points="4,8 96,8 50,94" fill="{RED}"/>'
                '<polygon points="20,17 80,17 50,73" fill="#fff"/></svg>')
    if cid == 12:
        return (v + '<polygon points="50,2 98,50 50,98 2,50" fill="#fff" stroke="#999"/>'
                f'<polygon points="50,18 82,50 50,82 18,50" fill="{YELLOW}"/></svg>')
    if cid == 17:
        return (v + f'<circle cx="50" cy="50" r="46" fill="{RED}"/>'
                '<rect x="20" y="42" width="60" height="16" fill="#fff"/></svg>')
    if cat == DANGER:
        return (v + f'<polygon points="50,6 96,92 4,92" fill="{RED}"/>'
                '<polygon points="50,24 82,84 18,84" fill="#fff"/></svg>')
    if cat == MANDATORY:
        return v + f'<circle cx="50" cy="50" r="46" fill="{BLUE}"/></svg>'
    if cat == END:
        return (v + '<circle cx="50" cy="50" r="46" fill="#fff" stroke="#888" stroke-width="3"/>'
                '<line x1="20" y1="80" x2="80" y2="20" stroke="#555" stroke-width="7"/></svg>')
    return (v + f'<circle cx="50" cy="50" r="46" fill="{RED}"/>'
            '<circle cx="50" cy="50" r="34" fill="#fff"/></svg>')


# --------------------------------------------------------------------------- #
# Traitement
# --------------------------------------------------------------------------- #
def predict_batch(model, x: np.ndarray) -> np.ndarray:
    return model(x, training=False).numpy()


def resize_like_pipeline(img: np.ndarray, size: int) -> np.ndarray:
    """Même redimensionnement que clocker.preprocessing.preprocess (affichage des étapes)."""
    interp = cv2.INTER_AREA if min(img.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def pixelated(img: np.ndarray, px: int = 224) -> np.ndarray:
    """Agrandit une petite image sans lissage : on voit ce que voit le réseau."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)


def largest_sign(image: np.ndarray):
    cands = detect_signs(image, use_hough=False)
    return max(cands, key=lambda c: c.w * c.h, default=None)


def grad_cam(name: str, x: np.ndarray, class_id: int) -> np.ndarray:
    """Carte Grad-CAM (Selvaraju et al., 2017) normalisée dans [0, 1]."""
    import tensorflow as tf
    sub, w, b, _ = get_cam_model(name)
    w_c = tf.constant(w[:, class_id])
    with tf.GradientTape() as tape:
        conv_out, feats = sub([x], training=False)
        logit = tf.reduce_sum(feats * w_c, axis=-1) + float(b[class_id])
    grads = tape.gradient(logit, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay_cam(base_rgb: np.ndarray, cam: np.ndarray, px: int = 224) -> np.ndarray:
    base = cv2.resize(base_rgb, (px, px), interpolation=cv2.INTER_CUBIC)
    heat = cv2.resize(cam, (px, px), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 0.5, heat, 0.5, 0)


def draw_box(img: np.ndarray, cand, color, label: str | None = None) -> np.ndarray:
    out = img.copy()
    t = max(2, int(round(min(img.shape[:2]) / 160)))
    x1, y1, x2, y2 = cand.box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
    if label:
        scale = max(0.45, min(img.shape[:2]) / 700)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t - 1))
        y0 = max(th + 6, y1)
        cv2.rectangle(out, (x1, y0 - th - 6), (x1 + tw + 6, y0), color, -1)
        cv2.putText(out, label, (x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return out


def fmt_pct(p: float) -> str:
    return f"{100 * p:.1f} %".replace(".", ",")


def short_name(cid: int) -> str:
    return f"{SPEED_VALUES[cid]} km/h" if cid in SPEED_VALUES else class_name(cid)


def read_table(path: Path) -> pd.DataFrame | None:
    """Lit un CSV de rapport ; None s'il est vide ou illisible."""
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    return None if df.empty else df


def to_rgb(file_bytes: bytes) -> np.ndarray | None:
    bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def thumb_uri(img: np.ndarray, px: int = 68) -> str:
    h, w = img.shape[:2]
    s = px / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""




# --------------------------------------------------------------------------- #
# Composants visuels
# --------------------------------------------------------------------------- #
def heading(text: str) -> None:
    st.markdown(f'<div class="ck-h">{text}</div>', unsafe_allow_html=True)


def gauge_svg(conf: float, threshold: float) -> str:
    """Jauge de confiance façon compteur de vitesse (0 à 100 %)."""
    cx, cy, arc_len = 100, 100, np.pi * 80
    ticks = []
    for k in range(11):
        a = np.pi * (1 - k / 10)
        r1 = 62 if k % 5 == 0 else 68
        x1, y1 = cx + r1 * np.cos(a), cy - r1 * np.sin(a)
        x2, y2 = cx + 74 * np.cos(a), cy - 74 * np.sin(a)
        w = 3 if k % 5 == 0 else 1.6
        ticks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="rgba(255,255,255,.75)" stroke-width="{w}"/>')
    ta = np.pi * (1 - threshold)
    tx1, ty1 = cx + 86 * np.cos(ta), cy - 86 * np.sin(ta)
    tx2, ty2 = cx + 98 * np.cos(ta), cy - 98 * np.sin(ta)
    angle = -90 + 180 * conf
    return f"""
<svg viewBox="0 0 200 138" role="img" aria-label="Confiance {fmt_pct(conf)}">
 <path d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="rgba(255,255,255,.18)"
       stroke-width="12" stroke-linecap="round"/>
 <path class="arc" d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="#fff" stroke-width="12"
       stroke-linecap="round" stroke-dasharray="{arc_len:.1f}"
       stroke-dashoffset="{arc_len * (1 - conf):.1f}"/>
 {''.join(ticks)}
 <line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="{YELLOW}"
       stroke-width="4" stroke-linecap="round"/>
 <g class="needle" style="transform: rotate({angle:.1f}deg)">
   <line x1="100" y1="100" x2="100" y2="34" stroke="{YELLOW}" stroke-width="4"
         stroke-linecap="round"/>
 </g>
 <circle cx="100" cy="100" r="8" fill="#111" stroke="{YELLOW}" stroke-width="3"/>
 <text x="100" y="134" text-anchor="middle" fill="#fff" font-size="24" font-weight="700"
       font-family="Barlow Condensed, Arial Narrow, sans-serif">{fmt_pct(conf)}</text>
</svg>"""


def hud_html(cid: int, conf: float, threshold: float, compact: bool = False) -> str:
    """Panneau de résultat : pictogramme, nom, jauge de confiance et décision d'alerte."""
    ok = conf >= threshold
    title = short_name(cid) if ok else "Panneau incertain"
    sub = class_category(cid) if ok else f"Meilleure hypothèse : {class_name(cid)}"
    if ok:
        verdict = ("<b>Alerte affichée</b>Le conducteur voit ce panneau sur le tableau "
                   "de bord.")
    else:
        verdict = (f"<b>Pas d'alerte</b>Confiance sous le seuil de {fmt_pct(threshold)} "
                   "(trait jaune).")
    classes = "ck-hud" + ("" if ok else " uncertain") + (" compact" if compact else "")
    return f"""
<div class="{classes}" role="status">
 <div class="ck-hud-in">
  <div class="ck-sign"><img src="{pictogram_uri(cid)}" alt="{class_name(cid)}"></div>
  <div><div class="ck-title{' long' if len(title) > 13 else ''}">{title}</div>
   <div class="ck-cat">{sub}</div></div>
  <div class="ck-gauge">{gauge_svg(conf, threshold)}<div class="ck-verdict">{verdict}</div></div>
 </div>
</div>"""


def topk_html(probs: np.ndarray, k: int = 5) -> str:
    rows = []
    for i in np.argsort(probs)[::-1][:k]:
        p = float(probs[i])
        rows.append(
            f'<div class="ck-row"><img src="{pictogram_uri(int(i))}" alt="">'
            f'<div><div class="lbl" title="{class_name(int(i))}">{class_name(int(i))}</div>'
            f'<div class="trk"><div class="fil" style="width:{max(p, .004) * 100:.1f}%">'
            f'</div></div></div><div class="pct">{fmt_pct(p)}</div></div>')
    return f'<div class="ck-top">{"".join(rows)}</div>'


def panel_html(label: str, value: str, color: str = "") -> str:
    return f'<div class="ck-panel {color}"><div><small>{label}</small><b>{value}</b></div></div>'


def info_html(title: str, text: str) -> str:
    return f'<div class="ck-info"><b>{title}</b>{text}</div>'


def hero_html(stats: dict | None) -> str:
    badges = ['<span class="ck-badge e">43 panneaux connus</span>']
    if stats:
        badges.append(f'<span class="ck-badge a">{fmt_pct(stats["accuracy"])} sur GTSRB</span>')
        badges.append(f'<span class="ck-badge n">{stats["latence_ms_image"]:.1f} ms par image'
                      .replace(".", ",") + "</span>")
        badges.append(f'<span class="ck-badge d">{stats["nb_parametres"] / 1000:.0f} k '
                      "paramètres</span>")
    posts = "".join(
        f'<div class="ck-post" style="left:{x}%;transform:scale({s})"></div>'
        for x, s in [(14, 1), (24, .7), (31, .5), (82, 1), (73, .7), (66, .5)])
    return f"""
<div class="ck-hero">
 <div class="ck-hero-text">
  <div class="ck-word">Clocker</div>
  <div class="ck-sub">Reconnaissance de panneaux routiers pour l'aide à la conduite</div>
  <div class="ck-badges">{''.join(badges)}</div>
 </div>
 <div class="ck-road" aria-hidden="true"><div class="ck-lanes"></div>{posts}</div>
</div>"""


def catalogue_html() -> str:
    cells = "".join(
        f'<div class="ck-cell"><img src="{pictogram_uri(i)}" alt="">{class_name(i)}'
        f'<br><small>classe {i}</small></div>' for i in range(config.NUM_CLASSES))
    return f'<div class="ck-grid">{cells}</div>'


def load_metrics(name: str) -> dict | None:
    main = config.MODEL_PATH.stem
    path = (config.REPORTS_DIR if name == main else config.REPORTS_DIR / name) / "metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Barre latérale : source, modèle, réglages
# --------------------------------------------------------------------------- #
models = available_models()
if not models:
    st.markdown(hero_html(None), unsafe_allow_html=True)
    st.markdown(info_html("Aucun modèle trouvé",
                          "Entraînez-en un avec <code>python -m clocker.train</code>, "
                          "puis rechargez cette page."), unsafe_allow_html=True)
    st.stop()

default_model = config.MODEL_PATH.stem if config.MODEL_PATH.stem in models else models[0]
demo_dir = ROOT / "assets" / "demo"
demo_files = sorted(p for p in demo_dir.glob("*") if p.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".webp"}) \
    if demo_dir.exists() else []
tests = test_index()

with st.sidebar:
    st.markdown("### Image à analyser")
    sources = ["Importer"]
    if tests is not None:
        sources.append("Exemples GTSRB")
    if demo_files:
        sources.append("Démo")
    source = st.segmented_control("Source", sources, default="Importer", required=True,
                                  label_visibility="collapsed")

    image, image_label = None, None
    if source == "Importer":
        ups = st.file_uploader("Photos de panneaux", accept_multiple_files=True,
                               type=["png", "jpg", "jpeg", "ppm", "bmp", "webp"])
        if ups:
            names = [u.name for u in ups]
            pick = st.selectbox("Image affichée", names, index=len(names) - 1) \
                if len(ups) > 1 else names[0]
            up = ups[names.index(pick)]
            image, image_label = to_rgb(up.getvalue()), up.name
            if image is None:
                st.error(f"{up.name} n'est pas une image lisible.")
    elif source == "Exemples GTSRB":
        cls = st.selectbox("Panneau", list(range(config.NUM_CLASSES)),
                           format_func=lambda i: f"{class_name(i)} (classe {i})", index=14)
        if st.button("Autre exemple", width="stretch"):
            st.session_state["ex_seed"] = st.session_state.get("ex_seed", 0) + 1
        rows = tests[tests.ClassId == cls]
        row = rows.sample(1, random_state=st.session_state.get("ex_seed", 0)).iloc[0]
        image, image_label = load_rgb(row.abs_path), f"GTSRB test, {Path(row.Path).name}"
    else:
        pick = st.selectbox("Image", [p.name for p in demo_files])
        image, image_label = load_rgb(demo_dir / pick), pick

    st.markdown("### Modèle")
    model_name = st.selectbox("Modèle", models, index=models.index(default_model),
                              label_visibility="collapsed")
    prep = config.load_prep(model_name)
    use_denoise = st.toggle("Débruitage bilatéral", value=prep["denoise"],
                            help="Valeur par défaut : réglage utilisé à l'entraînement.")
    use_clahe = st.toggle("CLAHE (canal L, espace Lab)", value=prep["clahe"])
    auto_crop = st.toggle("Recadrage automatique", value=True,
                          help="Cherche le panneau rouge ou bleu principal et le recadre "
                               "comme les images GTSRB avant de le classifier.")
    threshold = st.slider("Seuil d'alerte", 0.0, 1.0, 0.6, 0.05,
                          help="En dessous, le système n'alerte pas le conducteur.")

model = get_model(model_name)
size = model.input_shape[1]
st.markdown(hero_html(load_metrics(model_name)), unsafe_allow_html=True)

tab_rec, tab_pipe, tab_rob, tab_scene, tab_res, tab_cat = st.tabs([
    "Reconnaissance", "Étapes du traitement", "Robustesse", "Scène complète",
    "Résultats du modèle", "Panneaux reconnus",
])

# --------------------------------------------------------------------------- #
# Calcul principal (partagé par les onglets)
# --------------------------------------------------------------------------- #
result = None
if image is not None:
    cand = largest_sign(image) if auto_crop else None
    sign = crop(image, cand) if cand is not None else image
    x = preprocess(sign, size=size, use_clahe=use_clahe, denoise=use_denoise)[None]
    t0 = time.perf_counter()
    probs = predict_batch(model, x)[0]
    ms = 1000 * (time.perf_counter() - t0)
    cid = int(probs.argmax())
    result = dict(cand=cand, sign=sign, x=x, probs=probs, cid=cid,
                  conf=float(probs[cid]), ms=ms)

    key = hashlib.md5(image.tobytes()[:200000] + model_name.encode()).hexdigest()
    hist = st.session_state.setdefault("history", [])
    if not hist or hist[0]["key"] != key:
        hist.insert(0, dict(key=key, thumb=thumb_uri(sign), cid=cid, conf=result["conf"]))
        del hist[12:]

with st.sidebar:
    hist = st.session_state.get("history", [])
    if hist:
        st.markdown("### Journal de bord")
        st.markdown("".join(
            f'<div class="ck-chip"><img src="{h["thumb"]}" alt=""><div><b>'
            f'{short_name(h["cid"])}</b><br><span>{fmt_pct(h["conf"])}</span></div></div>'
            for h in hist), unsafe_allow_html=True)
        if st.button("Vider le journal", type="tertiary"):
            st.session_state["history"] = []
            st.rerun()

NO_IMAGE = info_html("Aucune image",
                     "Choisissez une image dans la barre de gauche pour ouvrir cet onglet.")

# --------------------------------------------------------------------------- #
# Onglet 1 : reconnaissance
# --------------------------------------------------------------------------- #
with tab_rec:
    if result is None:
        st.markdown(info_html(
            "Choisissez une image pour démarrer",
            "Importez une photo de panneau dans la barre de gauche, ou prenez un exemple du "
            "jeu de test GTSRB. Le modèle reconnaît les 43 panneaux de l'onglet "
            "<i>Panneaux reconnus</i> : un panneau absent de cette liste, comme le 110 km/h "
            "français, sera confondu avec le plus proche."), unsafe_allow_html=True)
    else:
        c_in, c_hud, c_top = st.columns([1, 1.6, 1.05], gap="large")
        with c_in:
            heading("Vue caméra")
            shown = image
            if result["cand"] is not None:
                shown = draw_box(image, result["cand"], (245, 197, 24))
            st.image(shown, width="stretch", caption=image_label)
            if auto_crop and result["cand"] is None:
                st.caption("Aucun panneau rouge ou bleu détecté : l'image entière est classée.")
            heading("Vu par le réseau")
            a, b = st.columns(2)
            a.image(pixelated(result["x"][0], 160), caption=f"Entrée {size}×{size}",
                    width="stretch")
            cam = grad_cam(model_name, result["x"], result["cid"])
            b.image(overlay_cam((result["x"][0] * 255).astype(np.uint8), cam, 160),
                    caption="Zones décisives", width="stretch")
        with c_hud:
            heading("Tableau de bord")
            st.markdown(hud_html(result["cid"], result["conf"], threshold),
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="ck-meta">Inférence en {result["ms"]:.1f} ms sur {size}×{size} px, '
                f'modèle {model_name}. La première prédiction est plus lente '
                f'(chargement de TensorFlow).</div>', unsafe_allow_html=True)
        with c_top:
            heading("Hypothèses")
            st.markdown(topk_html(result["probs"]), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Onglet 2 : étapes du traitement
# --------------------------------------------------------------------------- #
with tab_pipe:
    if result is None:
        st.markdown(NO_IMAGE, unsafe_allow_html=True)
    else:
        st.markdown('<p class="ck-lead">Chaque étape est exactement celle appliquée avant la '
                    'classification. Les images de 32×32 pixels sont agrandies sans lissage, '
                    'pour voir ce que voit le réseau.</p>', unsafe_allow_html=True)
        mask = color_mask(image, RED_RANGES) | color_mask(image, BLUE_RANGES)
        mask_view = image.copy()
        mask_view[mask == 0] = (mask_view[mask == 0] * 0.22).astype(np.uint8)
        resized = resize_like_pipeline(result["sign"], size)
        dn = finish(resized, use_clahe=False, denoise=True)
        cl = finish(resized, use_clahe=use_clahe, denoise=use_denoise)
        cam = grad_cam(model_name, result["x"], result["cid"])
        steps = [
            ("Image d'entrée", image, "Photo telle qu'importée."),
            ("Masques couleur", mask_view,
             "Pixels rouges et bleus en espace HSV ; le reste est assombri."),
            ("Recadrage", result["sign"],
             "Plus grande zone colorée, avec 10 % de marge comme dans GTSRB."
             if result["cand"] is not None else "Aucune zone trouvée : image entière."),
            ("Redimensionnement", pixelated(resized), f"{size}×{size} pixels, entrée du réseau."),
            ("Débruitage bilatéral", pixelated(dn),
             "Lisse le bruit en gardant les contours." + ("" if use_denoise
                                                          else " Désactivé pour ce modèle.")),
            ("CLAHE sur L (Lab)", pixelated(cl),
             "Contraste local rehaussé sans changer les couleurs."
             + ("" if use_clahe else " Désactivé : image après les étapes actives.")),
            ("Attention du modèle", overlay_cam((result["x"][0] * 255).astype(np.uint8), cam),
             "Grad-CAM : zones qui ont le plus pesé dans la décision."),
        ]
        for start in range(0, len(steps), 4):
            cols = st.columns(4, gap="medium")
            for col, (i, (title, img, text)) in zip(
                    cols, enumerate(steps[start:start + 4], start=start + 1)):
                with col:
                    st.markdown(f'<div class="ck-step"><div class="ck-pk">{i}</div>{title}</div>',
                                unsafe_allow_html=True)
                    st.image(img, width="stretch")
                    st.caption(text)

# --------------------------------------------------------------------------- #
# Onglet 3 : robustesse
# --------------------------------------------------------------------------- #
with tab_rob:
    if result is None:
        st.markdown(NO_IMAGE, unsafe_allow_html=True)
    else:
        st.markdown('<p class="ck-lead">La dégradation touche l\'image brute, avant le '
                    'prétraitement, comme sur une vraie caméra : même protocole que '
                    '<code>clocker.evaluate</code>.</p>', unsafe_allow_html=True)
        c_ctrl, c_view = st.columns([1, 2.1], gap="large")
        with c_ctrl:
            pert = st.selectbox("Dégradation", list(PERTURBATIONS))
            fn, levels = PERTURBATIONS[pert]
            lo, hi = float(min(levels)), float(max(levels))
            is_factor = pert.endswith("(facteur)")
            level = st.slider("Intensité", lo, hi, float(levels[0]),
                              step=(hi - lo) / 40, key=f"niveau_{pert}",
                              help="Pour les facteurs, 1 = image intacte ; plus petit = pire.")
        raw = resize_like_pipeline(result["sign"], size).astype(np.float32) / 255.0
        scale = size / 32

        def degrade(lv: float) -> np.ndarray:
            lv = lv * scale if pert in SCALED_WITH_SIZE else lv
            out = (np.clip(fn(raw, lv), 0, 1) * 255).astype(np.uint8)
            return finish(out, use_clahe=use_clahe, denoise=use_denoise)

        xp = degrade(level)
        pp = predict_batch(model, xp[None])[0]
        pc = int(pp.argmax())
        with c_view:
            a, b, c = st.columns([1, 1, 1.6])
            a.image(pixelated(raw), caption="Intacte", width="stretch")
            b.image(pixelated(xp), caption="Dégradée puis prétraitée", width="stretch")
            c.markdown(hud_html(pc, float(pp[pc]), threshold, compact=True),
                       unsafe_allow_html=True)

        sweep = np.linspace(levels[0], levels[-1], 21)
        ps = predict_batch(model, np.stack([degrade(float(lv)) for lv in sweep]))
        ref = result["cid"]
        keep = f"Toujours {short_name(ref)}"
        df = pd.DataFrame({
            "Intensité": sweep, "Probabilité": ps[:, ref],
            "Prédiction": [keep if p.argmax() == ref else "Autre panneau" for p in ps],
        })
        x_enc = alt.X("Intensité:Q", title=pert,
                      scale=alt.Scale(reverse=is_factor, domain=[lo, hi]),
                      axis=alt.Axis(tickCount=8, format="~g", grid=False))
        y_enc = alt.Y("Probabilité:Q", scale=alt.Scale(domain=[0, 1]),
                      axis=alt.Axis(format="%"), title=f"Probabilité de « {short_name(ref)} »")
        area = alt.Chart(df).mark_area(
            color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0, stops=[
                alt.GradientStop(color="rgba(69,74,81,0)", offset=0),
                alt.GradientStop(color="rgba(69,74,81,.22)", offset=1)]),
            interpolate="monotone").encode(x=x_enc, y=y_enc)
        line = alt.Chart(df).mark_line(color=GRAPHITE, strokeWidth=2.5,
                                       interpolate="monotone").encode(x=x_enc, y=y_enc)
        pts = alt.Chart(df).mark_circle(size=80, opacity=1).encode(
            x=x_enc, y=y_enc,
            color=alt.Color("Prédiction:N", scale=alt.Scale(
                domain=[keep, "Autre panneau"], range=[BLUE, RED]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("Intensité:Q", format=".2f"),
                     alt.Tooltip("Probabilité:Q", format=".1%"), "Prédiction:N"])
        rule = alt.Chart(pd.DataFrame({"x": [level]})).mark_rule(
            strokeDash=[6, 5], color=GRAPHITE, strokeWidth=2).encode(x="x:Q")
        heading("Jusqu'où le modèle tient-il sur cette image ?")
        st.altair_chart((area + line + pts + rule).properties(height=290),
                        width="stretch", theme=None)
        flips = df.index[df["Prédiction"] == "Autre panneau"]
        if len(flips):
            st.caption(f"La prédiction change à partir d'une intensité de "
                       f"{df['Intensité'][flips[0]]:.2f}.".replace(".", ",", 1))
        else:
            st.caption("La prédiction reste correcte sur toute la plage testée.")

# --------------------------------------------------------------------------- #
# Onglet 4 : scène complète
# --------------------------------------------------------------------------- #
with tab_scene:
    if result is None:
        st.markdown(info_html("Aucune image", "Choisissez une image dans la barre de gauche, "
                              "de préférence une photo de rue avec plusieurs panneaux."),
                    unsafe_allow_html=True)
    else:
        use_hough = st.toggle("Ajouter les cercles de Hough", value=False,
                              help="Trouve aussi des panneaux peu colorés, mais ajoute des "
                                   "fausses détections.")
        cands = detect_signs(image, use_hough=use_hough)
        found, boxed = [], image.copy()
        if cands:
            crops = [crop(image, c) for c in cands]
            xb = np.stack([preprocess(cr, size=size, use_clahe=use_clahe, denoise=use_denoise)
                           for cr in crops])
            pb = predict_batch(model, xb)
            for c, cr, p in zip(cands, crops, pb):
                k = int(p.argmax())
                if p[k] >= threshold:
                    found.append((c, cr, k, float(p[k])))
                    boxed = draw_box(boxed, c, BOX_RGB[class_category(k)],
                                     f"{short_name(k)} {100 * p[k]:.0f}%")
        c_img, c_list = st.columns([2, 1], gap="large")
        with c_img:
            heading("Détections")
            st.image(boxed, width="stretch",
                     caption=f"Zones candidates : {len(cands)}. Panneaux au-dessus du seuil "
                             f"d'alerte : {len(found)}.")
        with c_list:
            heading("Panneaux retenus")
            if not found:
                st.markdown(info_html("Aucun panneau retenu",
                                      "Baissez le seuil d'alerte ou ajoutez les cercles de "
                                      "Hough."), unsafe_allow_html=True)
            for c, cr, k, p in sorted(found, key=lambda f: -f[3]):
                st.markdown(
                    f'<div class="ck-chip"><img src="{pictogram_uri(k)}" alt="">'
                    f'<img src="{thumb_uri(cr)}" alt=""><div><b>{class_name(k)}</b><br>'
                    f'<span>{fmt_pct(p)}, zone {c.color}, forme {c.shape}</span></div></div>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Onglet 5 : résultats du modèle
# --------------------------------------------------------------------------- #
with tab_res:
    main = config.MODEL_PATH.stem
    rep_dir = config.REPORTS_DIR if model_name == main else config.REPORTS_DIR / model_name
    fig_dir = config.FIGURES_DIR if model_name == main else rep_dir / "figures"
    m = load_metrics(model_name)
    if m is None:
        st.markdown(info_html(f"Pas encore de résultats pour {model_name}",
                              f"Lancez <code>python -m clocker.evaluate --name {model_name}"
                              "</code>, puis rechargez la page."), unsafe_allow_html=True)
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(panel_html("Accuracy sur le test GTSRB", fmt_pct(m["accuracy"]), "green"),
                    unsafe_allow_html=True)
        k2.markdown(panel_html("Macro-F1 (43 classes)", fmt_pct(m["macro_f1"])),
                    unsafe_allow_html=True)
        k3.markdown(panel_html("Latence par image",
                               f"{m['latence_ms_image']:.1f} ms".replace(".", ",")),
                    unsafe_allow_html=True)
        k4.markdown(panel_html("Paramètres", f"{m['nb_parametres']:,}".replace(",", " "),
                               "yellow"), unsafe_allow_html=True)
        st.caption(f"{m['nb_images_test']} images de test jamais vues à l'entraînement.")

        comp = config.REPORTS_DIR / "comparaison.csv"
        comp_df = read_table(comp) if comp.exists() else None
        if comp_df is not None:
            heading("Comparaison des expériences")
            st.dataframe(comp_df, hide_index=True, width="stretch")

        figs = [("Courbes d'apprentissage", config.FIGURES_DIR / f"{model_name}_curves.png"),
                ("Matrice de confusion", fig_dir / "confusion_matrix.png"),
                ("Robustesse", fig_dir / "robustness.png"),
                ("Erreurs les plus confiantes", fig_dir / "failures.png")]
        figs = [(t, p) for t, p in figs if p.exists()]
        for i in range(0, len(figs), 2):
            cols = st.columns(2, gap="large")
            for col, (t, p) in zip(cols, figs[i:i + 2]):
                with col:
                    heading(t)
                    st.image(str(p), width="stretch")

        conf_csv = rep_dir / "top_confusions.csv"
        conf_df = read_table(conf_csv) if conf_csv.exists() else None
        if conf_df is not None:
            heading("Confusions les plus fréquentes")
            st.dataframe(conf_df, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Onglet 6 : catalogue
# --------------------------------------------------------------------------- #
with tab_cat:
    st.markdown('<p class="ck-lead">Le modèle choisit toujours parmi ces 43 panneaux du jeu '
                'allemand GTSRB. Un panneau absent de la liste est classé dans la catégorie la '
                'plus proche, parfois avec une forte confiance.</p>', unsafe_allow_html=True)
    st.markdown(catalogue_html(), unsafe_allow_html=True)
