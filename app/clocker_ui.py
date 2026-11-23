# ruff: noqa: E501  (feuille de style CSS intégrée : lignes longues assumées)
"""Clocker — interface de démonstration (barre latérale bleu nuit, contenu clair).

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
# Palette : barre latérale bleu nuit, contenu clair bleuté, couleurs de signalisation
# --------------------------------------------------------------------------- #
NAVY = "#0E1B33"        # barre latérale
ASPHALT = "#EEF2F7"     # fond de page
TARMAC = "#FFFFFF"      # cartes
KERB = "#D8E0EB"        # bordures, pistes des jauges
MARKING = "#14203A"     # texte principal
MUTED = "#5B6780"       # texte secondaire
GRAPHITE = "#2A3550"    # traits neutres (graphiques)
BLUE = "#1F63E0"        # accent principal, panneaux de direction
RED = "#E0243A"         # danger, interdiction
GREEN = "#1E9E57"       # validation
YELLOW = "#F7B500"      # priorité, avertissement

CATEGORY_COLORS = {
    SPEED_LIMIT: BLUE, DANGER: RED, PROHIBITION: RED,
    MANDATORY: BLUE, PRIORITY: YELLOW, END: MUTED,
}
BOX_RGB = {
    SPEED_LIMIT: (31, 99, 224), DANGER: (224, 36, 58), PROHIBITION: (224, 36, 58),
    MANDATORY: (31, 99, 224), PRIORITY: (247, 181, 0), END: (140, 150, 168),
}
SCALED_WITH_SIZE = {"Flou gaussien (sigma)", "Flou de bougé (px)"}

st.set_page_config(page_title="Clocker", page_icon="🛣️", layout="wide",
                   initial_sidebar_state="expanded")

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp { font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif; color: %MARKING%;
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(255,255,255,.9), transparent 60%),
    radial-gradient(900px 600px at 10% 110%, rgba(214,226,243,.8), transparent 60%),
    linear-gradient(115deg, transparent 38%, rgba(255,255,255,.55) 52%, transparent 68%),
    %ASPHALT%; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.2rem; padding-left: 2.2rem; padding-right: 2.2rem; max-width: 1480px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 700; letter-spacing: -.01em; }

/* ---------- état en haut à droite ---------- */
.ck-status { position: fixed; top: .95rem; right: 15rem; z-index: 1000001; display: flex;
  align-items: center; gap: .45rem; font-size: .8rem; font-weight: 600; letter-spacing: .04em;
  color: #2E7D4F; padding-right: .9rem; border-right: 1px solid %KERB%; }
.ck-status i { width: 8px; height: 8px; border-radius: 50%; background: %GREEN%;
  box-shadow: 0 0 0 3px rgba(30,158,87,.18); }

/* ---------- barre latérale ---------- */
[data-testid="stSidebar"] { background:
    radial-gradient(160% 55% at 110% -8%, rgba(120,160,230,.20) 0 38%, transparent 38.5%),
    radial-gradient(90% 40% at 0% 0%, rgba(47,116,240,.22), transparent 70%),
    linear-gradient(180deg, #11244A 0%, #0D1B36 45%, #0A1528 100%); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 [data-testid="stIconMaterial"],
[data-testid="stSidebar"] h4 span[role="img"] { font-size: 1.35rem !important; color: #fff; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #C9D3E4; }
[data-testid="stSidebar"][aria-expanded="true"] { width: 264px !important; min-width: 264px !important;
  max-width: 264px !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] p { font-size: .95rem; color: #EEF2F9; }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { font-size: 1.08rem; font-weight: 700;
  margin: .5rem 0 .5rem; color: #fff; }
[data-testid="stSidebar"] hr { border: none; border-top: 1px solid rgba(255,255,255,.10); margin: .9rem 0 .5rem; }
.ck-logo { text-align: center; padding: .4rem 0 1.1rem; border-bottom: 1px solid rgba(255,255,255,.09);
  margin-bottom: .4rem; }
.ck-logo svg { width: 64px; height: 39px; }
.ck-logo b { display: block; font-size: 1.75rem; font-weight: 800; color: #fff; letter-spacing: -.02em;
  line-height: 1.1; }
.ck-logo span { font-size: .84rem; color: #B9C5DA; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  position: relative; min-height: 124px; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: .35rem; padding: 1rem; cursor: pointer;
  background: rgba(255,255,255,.035); border: 1.5px dashed rgba(255,255,255,.30); border-radius: 14px;
  transition: border-color .15s ease, background .15s ease; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:focus-within {
  border-color: #6FA0FF; background: rgba(47,116,240,.10); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > span > button {
  position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; z-index: 2; cursor: pointer; }
[data-testid="stSidebar"] [data-testid="stFileChips"], [data-testid="stSidebar"] [data-testid="stFileChip"],
[data-testid="stSidebar"] [data-testid="stFileChipDeleteBtn"] { position: relative; z-index: 3; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]::before {
  content: ""; width: 34px; height: 30px; background: center / contain no-repeat
  url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 9.5a4 4 0 0 1-.5 7.97'/%3E%3Cpath d='m9 13 3-3 3 3M12 10v9'/%3E%3C/svg%3E"); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]::after {
  content: "Glisser-déposer une image\Aou cliquer pour parcourir"; white-space: pre-line;
  text-align: center; font-size: .9rem; font-weight: 600; line-height: 1.5; color: #EEF2F9; }
[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] {
  display: flex; flex-wrap: nowrap; gap: .5rem; }
[data-testid="stSidebar"] button[data-variant="segmented_control"] {
  flex: 1 1 auto; white-space: nowrap; border-radius: 10px !important; padding: .45rem .7rem;
  border: 1px solid rgba(255,255,255,.22) !important; background: rgba(255,255,255,.04); }
[data-testid="stSidebar"] button[data-variant="segmented_control"] p { font-size: .84rem; font-weight: 600; }
[data-testid="stSidebar"] button[data-variant="segmented_control"][aria-checked="true"] {
  background: %BLUE% !important; border-color: %BLUE% !important; color: #fff !important;
  box-shadow: 0 6px 16px rgba(31,99,224,.35); }
[data-testid="stSidebar"] button[data-variant="segmented_control"][aria-checked="true"] * { color: #fff !important; }
.ck-log { display: flex; align-items: center; gap: .6rem; color: %MARKING%;
  background: linear-gradient(90deg, #FFFFFF, #E9EDF4); border-radius: 12px; padding: .6rem .9rem;
  margin-bottom: .45rem; font-size: .92rem; font-weight: 600; box-shadow: 0 6px 14px rgba(0,0,0,.18); }
.ck-log i { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.ck-log img { width: 26px; height: 26px; object-fit: contain; border-radius: 5px; }
.ck-log span { margin-left: auto; color: %MUTED%; font-weight: 500; font-size: .8rem; }

/* ---------- onglets ---------- */
.stTabs [data-baseweb="tab-list"], [data-testid="stTabs"] [role="tablist"] {
  gap: 1.25rem; border-bottom: 1px solid %KERB%; }
.stTabs [data-baseweb="tab"], [data-testid="stTab"] { padding: .8rem .15rem .75rem; background: transparent; }
.stTabs [data-baseweb="tab"] p, [data-testid="stTab"] p { font-size: .98rem; font-weight: 500; white-space: nowrap;
  color: #36425E; display: flex; align-items: center; gap: .45rem; }
[data-testid="stTab"] p span[role="img"], .stTabs [data-baseweb="tab"] [data-testid="stIconMaterial"] {
  font-size: 1.35rem !important; color: #44506B; }
.stTabs [aria-selected="true"] p, [data-testid="stTab"][aria-selected="true"] p,
[data-testid="stTab"][aria-selected="true"] p span[role="img"] { color: %BLUE% !important; font-weight: 700; }
[data-testid="stTab"][aria-selected="true"] { box-shadow: inset 0 -3px 0 %BLUE%; }
[data-testid="stTab"]:hover p { color: %MARKING%; }
.stTabs [data-baseweb="tab-highlight"] { background: %BLUE%; height: 3px; border-radius: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- en-tête ---------- */
.ck-hero { position: relative; min-height: 222px; border-radius: 20px; overflow: hidden;
  margin: .3rem 0 1.4rem; background: linear-gradient(100deg, #FFFFFF 0%, #F3F6FB 45%, #E9EFF8 100%);
  box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 18px 40px rgba(20,40,80,.12),
              0 0 0 1px rgba(20,40,80,.05); }
.ck-hero-img { position: absolute; top: 0; right: 0; bottom: 0; width: 66%;
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.6) 14%, #000 32%);
          mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.6) 14%, #000 32%); }
.ck-hero-img svg, .ck-hero-img div { width: 100%; height: 100%; display: block; }
.ck-hero-img div { background-size: cover; background-position: center; }
.ck-hero-text { position: relative; z-index: 2; padding: 1.9rem 2.3rem; max-width: 470px; }
.ck-word { font-weight: 800; font-size: clamp(2.8rem, 4.4vw, 4.2rem); line-height: .95;
  letter-spacing: -.035em; color: #0B1630; }
.ck-sub { font-size: 1.22rem; line-height: 1.35; color: #1C2A48; margin-top: .7rem; font-weight: 500; }
.ck-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.1rem; }
.ck-badge { display: inline-flex; align-items: center; gap: .45rem; padding: .5rem .85rem;
  border-radius: 8px; color: #fff; font-weight: 700; font-size: .9rem; white-space: nowrap;
  box-shadow: 0 6px 14px rgba(20,40,80,.14); }
.ck-badge svg { width: 16px; height: 16px; flex: none; }
.ck-badge.e { background: %GREEN%; } .ck-badge.a { background: %RED%; }
.ck-badge.n { background: %BLUE%; } .ck-badge.d { background: %YELLOW%; color: #1B1B1B; }
.ck-lane { animation: ck-dash 1.6s linear infinite; }
@keyframes ck-dash { to { stroke-dashoffset: -60; } }

/* ---------- titres de section ---------- */
.ck-h { font-weight: 700; font-size: 1.08rem; color: %MARKING%; display: flex; align-items: center;
  gap: .55rem; margin: .2rem 0 .7rem; }
.ck-h::before { content: ""; width: 4px; height: 18px; border-radius: 3px; background: %BLUE%; flex: none; }
.ck-lead { color: #3B4763; max-width: 95ch; }

/* ---------- carte d'information ---------- */
.ck-info { position: relative; display: flex; gap: 1.2rem; align-items: flex-start; overflow: hidden;
  background: rgba(255,255,255,.85); border: 1px solid #E3E9F2; border-left: 6px solid %BLUE%;
  border-radius: 16px; padding: 1.6rem 1.8rem; box-shadow: 0 14px 34px rgba(20,40,80,.08); }
.ck-info .ic { width: 40px; height: 40px; flex: none; border-radius: 50%; background: #DCE7FB;
  display: flex; align-items: center; justify-content: center; }
.ck-info .ic svg { width: 26px; height: 26px; }
.ck-info b { display: block; font-size: 1.3rem; font-weight: 800; color: %MARKING%; margin-bottom: .3rem; }
.ck-info p { margin: 0; color: #3B4763; line-height: 1.6; max-width: 80ch; position: relative; z-index: 1; }
.ck-info .bg { position: absolute; right: 1.6rem; top: 50%; transform: translateY(-50%);
  width: 80px; height: 80px; opacity: .22; }

/* ---------- panneau de résultat (tableau de bord) ---------- */
.ck-hud { border-radius: 18px; padding: 7px; background: %BLUE%;
  box-shadow: 0 18px 40px rgba(31,99,224,.28); }
.ck-hud.uncertain { background: repeating-linear-gradient(-45deg, %YELLOW% 0 16px, #1E2A44 16px 32px);
  box-shadow: 0 18px 40px rgba(20,40,80,.2); animation: ck-hazard 1s steps(2) 3; }
@keyframes ck-hazard { 50% { filter: brightness(.6); } }
.ck-hud-in { border: 3px solid #fff; border-radius: 12px; padding: 1.1rem 1.2rem 1rem;
  background: %BLUE%; display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 1.1rem 1.3rem;
  align-items: center; }
.ck-hud.uncertain .ck-hud-in { background: #24314F; }
.ck-sign { position: relative; width: 112px; height: 112px; border-radius: 14px; overflow: hidden;
  background: #fff; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 0 3px rgba(255,255,255,.3), 0 10px 24px rgba(0,0,0,.18); }
.ck-sign img { width: 94px; height: 94px; object-fit: contain;
  animation: ck-approach .9s cubic-bezier(.2,.8,.2,1) both; }
.ck-sign::after { content: ""; position: absolute; inset: -20%;
  background: linear-gradient(105deg, transparent 38%, rgba(255,255,255,.9) 50%, transparent 62%);
  transform: translateX(-110%); animation: ck-sweep 1.1s .55s ease-out forwards; }
@keyframes ck-approach { from { transform: scale(.3); filter: blur(5px) brightness(.4); opacity: .2; } }
@keyframes ck-sweep { to { transform: translateX(110%); } }
.ck-title { font-weight: 800; font-size: clamp(1.7rem, 2.3vw, 2.4rem); line-height: 1.05; color: #fff;
  letter-spacing: -.02em; overflow-wrap: normal; word-break: normal; hyphens: none; }
.ck-title.long { font-size: clamp(1.3rem, 1.7vw, 1.7rem); }
.ck-cat { color: rgba(255,255,255,.85); margin-top: .35rem; font-size: .98rem; }
.ck-gauge { grid-column: 1 / -1; display: flex; align-items: center; gap: 1rem;
  border-top: 1px solid rgba(255,255,255,.22); padding-top: .8rem; }
.ck-gauge svg { width: 150px; flex: none; }
.ck-gauge .needle { transform-origin: 100px 100px; animation: ck-needle 1.2s cubic-bezier(.3,1.4,.5,1) both; }
.ck-gauge .arc { animation: ck-arc 1.2s ease-out both; }
@keyframes ck-needle { from { transform: rotate(-90deg); } }
@keyframes ck-arc { from { stroke-dashoffset: 251.3; } }
.ck-verdict { color: #fff; font-size: .95rem; line-height: 1.4; }
.ck-verdict b { font-size: 1.12rem; display: block; }
.ck-hud.compact .ck-hud-in { padding: .8rem; gap: .8rem; }
.ck-hud.compact .ck-sign { width: 88px; height: 88px; }
.ck-hud.compact .ck-sign img { width: 74px; height: 74px; }
.ck-hud.compact .ck-title { font-size: 1.45rem; }
.ck-hud.compact .ck-gauge svg { width: 132px; }
.ck-hud.compact .ck-verdict { display: none; }
.ck-meta { color: %MUTED%; font-size: .86rem; margin-top: .7rem; }

/* ---------- hypothèses ---------- */
.ck-top { display: flex; flex-direction: column; gap: .75rem; }
.ck-row { display: grid; grid-template-columns: 40px minmax(0, 1fr) auto; gap: .7rem; align-items: center;
  background: #fff; border: 1px solid #E3E9F2; border-radius: 12px; padding: .5rem .7rem; }
.ck-row img { width: 36px; height: 36px; object-fit: contain; }
.ck-row .lbl { font-size: .9rem; font-weight: 600; color: %MARKING%; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.ck-row .trk { height: 7px; background: #E6ECF5; border-radius: 5px; margin-top: .35rem; overflow: hidden; }
.ck-row .fil { height: 100%; border-radius: 5px; background: #AFBCD2;
  transform-origin: left; animation: ck-grow .8s cubic-bezier(.2,.8,.2,1) both; }
.ck-row:first-child { border-color: rgba(31,99,224,.35); box-shadow: 0 8px 20px rgba(31,99,224,.12); }
.ck-row:first-child .fil { background: %BLUE%; }
.ck-row .pct { font-weight: 700; font-size: .95rem; text-align: right; color: %MARKING%;
  white-space: nowrap; min-width: 54px; }
.ck-row:nth-child(2) .fil { animation-delay: .08s; } .ck-row:nth-child(3) .fil { animation-delay: .16s; }
.ck-row:nth-child(4) .fil { animation-delay: .24s; } .ck-row:nth-child(5) .fil { animation-delay: .32s; }
@keyframes ck-grow { from { transform: scaleX(0); } }

/* ---------- étapes : bornes kilométriques ---------- */
.ck-step { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; font-weight: 700;
  font-size: 1rem; color: %MARKING%; }
.ck-pk { width: 28px; height: 38px; flex: none; border-radius: 14px 14px 3px 3px;
  background: linear-gradient(180deg, %RED% 0 40%, #fff 40%); color: #111;
  display: flex; align-items: flex-end; justify-content: center; padding-bottom: 3px;
  font-weight: 800; font-size: .95rem; box-shadow: 0 0 0 1px %KERB%, 0 4px 10px rgba(20,40,80,.12); }

/* ---------- cartes ---------- */
.ck-panel { background: %BLUE%; border-radius: 14px; padding: 5px; box-shadow: 0 12px 26px rgba(20,40,80,.12); }
.ck-panel > div { border: 2px solid #fff; border-radius: 10px; padding: .8rem 1rem; color: #fff; }
.ck-panel small { opacity: .9; font-size: .86rem; }
.ck-panel b { display: block; font-weight: 800; font-size: 1.9rem; line-height: 1.1; letter-spacing: -.02em; }
.ck-panel.green { background: %GREEN%; } .ck-panel.red { background: %RED%; }
.ck-panel.yellow { background: %YELLOW%; } .ck-panel.yellow > div { border-color: #1B1B1B; color: #1B1B1B; }
.ck-chip { display: flex; gap: .6rem; align-items: center; padding: .5rem .6rem; border-radius: 12px;
  background: #fff; border: 1px solid #E3E9F2; margin-bottom: .45rem; }
.ck-chip img { width: 36px; height: 36px; object-fit: contain; border-radius: 6px; }
.ck-chip b { font-weight: 700; font-size: .95rem; }
.ck-chip span { color: %MUTED%; font-size: .82rem; }
.ck-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: .85rem; }
.ck-cell { background: #fff; border: 1px solid #E3E9F2; border-radius: 14px; padding: .8rem .5rem;
  text-align: center; font-size: .82rem; line-height: 1.25; color: #2C3856;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
.ck-cell:hover { transform: translateY(-3px); border-color: rgba(31,99,224,.4);
  box-shadow: 0 10px 22px rgba(20,40,80,.12); }
.ck-cell img { width: 60px; height: 60px; object-fit: contain; display: block; margin: 0 auto .5rem; }
.ck-cell small { color: %MUTED%; }
[data-testid="stMain"] [data-testid="stImage"] img { border-radius: 12px; }

@media (max-width: 900px) {
  .ck-hero-img { width: 100%; opacity: .35; }
  .ck-status { display: none; }
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
        "font": "Plus Jakarta Sans",
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


ICONS = {
    "check": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10.5" fill="#fff"/>'
             '<path d="m7.3 12.4 3.1 3.1 6.3-6.6" fill="none" stroke="#1E9E57" stroke-width="2.6" '
             'stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "alert": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10.5" fill="#fff"/>'
             '<rect x="10.7" y="5.8" width="2.6" height="8" rx="1.3" fill="#E0243A"/>'
             '<circle cx="12" cy="17.3" r="1.6" fill="#E0243A"/></svg>',
    "image": '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
             'stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"/>'
             '<circle cx="9" cy="10" r="1.8" fill="#fff"/><path d="m4 18 5.5-5 4 3.5 3-2.5L21 18"/></svg>',
    "db": '<svg viewBox="0 0 24 24" fill="none" stroke="#1B1B1B" stroke-width="2">'
          '<ellipse cx="12" cy="5.5" rx="7.5" ry="3"/><path d="M4.5 5.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 '
          '7.5-3v-6M4.5 11.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/></svg>',
    "info": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="#1F63E0"/>'
            '<rect x="10.8" y="10" width="2.4" height="7.5" rx="1.2" fill="#fff"/>'
            '<circle cx="12" cy="6.8" r="1.5" fill="#fff"/></svg>',
    "warn": '<svg viewBox="0 0 24 24" fill="none" stroke="#7F93B6" stroke-width="1.6" '
            'stroke-linejoin="round"><path d="M12 2.8 22 20H2z"/><path d="M12 9v5.2" '
            'stroke-linecap="round"/><circle cx="12" cy="17" r=".9" fill="#7F93B6"/></svg>',
}

LOGO = ('<svg viewBox="0 0 76 46" aria-hidden="true">'
        '<polygon points="4,44 31,3 36,3 21,44" fill="#fff"/>'
        '<polygon points="55,44 40,3 45,3 72,44" fill="#F7B500"/>'
        '<rect x="36.6" y="8" width="2.8" height="7" fill="#fff"/>'
        '<rect x="36.2" y="20" width="3.6" height="9" fill="#fff"/>'
        '<rect x="35.7" y="33" width="4.6" height="11" fill="#fff"/></svg>')


def info_html(title: str, text: str) -> str:
    return (f'<div class="ck-info"><div class="ic">{ICONS["info"]}</div>'
            f'<div><b>{title}</b><p>{text}</p></div>'
            f'<div class="bg" aria-hidden="true">{ICONS["warn"]}</div></div>')


def road_illustration() -> str:
    """Route au coucher du soleil (dessin vectoriel, aucune photo externe)."""
    trees = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'
        for x, y, r, c in [(40, 176, 16, "#2E4A2B"), (70, 170, 21, "#34532F"),
                           (105, 178, 13, "#2B4428"), (140, 174, 17, "#30502D"),
                           (178, 180, 10, "#2B4428"), (360, 183, 9, "#2E4A2B"),
                           (400, 180, 13, "#30502D"), (640, 182, 10, "#2E4A2B"),
                           (670, 176, 15, "#34532F"), (700, 178, 12, "#2B4428"),
                           (735, 170, 20, "#34532F"), (772, 175, 16, "#2E4A2B"),
                           (812, 166, 26, "#30502D"), (858, 172, 19, "#2E4A2B"),
                           (890, 162, 30, "#34532F")])
    posts = "".join(
        f'<g transform="translate({x} {y}) scale({k})"><rect x="-4" y="-34" width="8" height="34" '
        f'rx="3" fill="#F4F4F2"/><rect x="-4" y="-26" width="8" height="6" fill="#1B1B1B"/>'
        f'<rect x="-2.5" y="-31" width="5" height="3" fill="#FF8A1F"/></g>'
        for x, y, k in [(360, 276, 1.15), (418, 238, .8), (452, 218, .6), (478, 206, .45),
                        (690, 282, 1.25), (628, 242, .85), (594, 222, .62), (572, 209, .47)])
    return f"""
<svg viewBox="0 0 900 300" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
 <defs>
  <linearGradient id="ck-sky" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0" stop-color="#6C93C8"/><stop offset=".38" stop-color="#B9CBE4"/>
   <stop offset=".55" stop-color="#F1B97F"/><stop offset=".64" stop-color="#F6D2A0"/></linearGradient>
  <radialGradient id="ck-sun" cx="530" cy="186" r="150" gradientUnits="userSpaceOnUse">
   <stop offset="0" stop-color="#FFF1CF" stop-opacity=".95"/>
   <stop offset="1" stop-color="#FFF1CF" stop-opacity="0"/></radialGradient>
  <linearGradient id="ck-field" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0" stop-color="#7C8F4A"/><stop offset="1" stop-color="#4C6A2A"/></linearGradient>
  <linearGradient id="ck-tar" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0" stop-color="#8A8E95"/><stop offset="1" stop-color="#3B4048"/></linearGradient>
 </defs>
 <rect width="900" height="300" fill="url(#ck-sky)"/>
 <rect width="900" height="300" fill="url(#ck-sun)"/>
 <path d="M0 190 Q120 182 240 188 T480 186 T700 187 T900 184 V200 H0Z" fill="#3E5A35" opacity=".75"/>
 {trees}
 <rect y="192" width="900" height="108" fill="url(#ck-field)"/>
 <polygon points="516,191 528,191 900,300 250,300" fill="url(#ck-tar)"/>
 <polygon points="516,191 518.5,191 290,300 268,300" fill="#F4F4F2"/>
 <polygon points="526,191 528.5,191 885,300 860,300" fill="#F4F4F2"/>
 <line class="ck-lane" x1="522" y1="192" x2="588" y2="300" stroke="#F4F4F2" stroke-width="5"
       stroke-dasharray="26 34"/>
 {posts}
</svg>"""


def hero_media() -> str:
    """Photo personnelle si assets/hero.jpg|png|webp existe, sinon l'illustration."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        photo = ROOT / "assets" / f"hero.{ext}"
        if photo.exists():
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            uri = f"data:image/{mime};base64," + base64.b64encode(photo.read_bytes()).decode()
            return f'<div style="background-image:url({uri})"></div>'
    return road_illustration()


def hero_html(stats: dict | None) -> str:
    badges = [f'<span class="ck-badge e">{ICONS["check"]}43 panneaux connus</span>']
    if stats:
        badges.append(f'<span class="ck-badge a">{ICONS["alert"]}'
                      f'{fmt_pct(stats["accuracy"])} sur GTSRB</span>')
        badges.append(f'<span class="ck-badge n">{ICONS["image"]}'
                      + f'{stats["latence_ms_image"]:.1f} ms par image'.replace(".", ",")
                      + "</span>")
        badges.append(f'<span class="ck-badge d">{ICONS["db"]}'
                      f'{stats["nb_parametres"] / 1000:.0f} k paramètres</span>')
    return f"""
<div class="ck-hero">
 <div class="ck-hero-img">{hero_media()}</div>
 <div class="ck-hero-text">
  <div class="ck-word">Clocker</div>
  <div class="ck-sub">Reconnaissance de panneaux routiers<br>pour l'aide à la conduite</div>
  <div class="ck-badges">{''.join(badges)}</div>
 </div>
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
    st.markdown(f'<div class="ck-logo">{LOGO}<b>Clocker</b>'
                '<span>Analyse &amp; performance routière</span></div>', unsafe_allow_html=True)
    st.markdown("#### :material/bar_chart: Image à analyser")
    sources = ["Importer"]
    if tests is not None:
        sources.append("Exemples GTSRB")
    if demo_files:
        sources.append("Démo")
    source_icons = {"Importer": ":material/upload: Importer",
                    "Exemples GTSRB": "Exemples GTSRB", "Démo": ":material/play_arrow: Démo"}
    source = st.segmented_control("Source", sources, default="Importer", required=True,
                                  format_func=source_icons.get, label_visibility="collapsed")

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

    st.divider()
    st.markdown("#### :material/settings: Modèle")
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
st.markdown('<div class="ck-status"><i></i>MODÈLE PRÊT</div>', unsafe_allow_html=True)
st.markdown(hero_html(load_metrics(model_name)), unsafe_allow_html=True)

tab_rec, tab_pipe, tab_rob, tab_scene, tab_res, tab_cat = st.tabs([
    ":material/directions_car: Reconnaissance", ":material/settings: Étapes du traitement",
    ":material/shield: Robustesse", ":material/play_arrow: Scène complète",
    ":material/bar_chart: Résultats du modèle", ":material/image: Panneaux reconnus",
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
        st.divider()
        st.markdown("#### Journal de bord")
        st.markdown("".join(
            f'<div class="ck-log" style="color:'
            f'{CATEGORY_COLORS.get(class_category(h["cid"]), MARKING)}"><i style="background:'
            f'{CATEGORY_COLORS.get(class_category(h["cid"]), MUTED)}"></i>'
            f'{short_name(h["cid"])}<span>{fmt_pct(h["conf"])}</span></div>'
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
            "jeu de test GTSRB.<br>Le modèle reconnaît les 43 panneaux de l'onglet "
            "<i>Panneaux reconnus</i> : un panneau absent de cette liste, comme le 110 km/h "
            "français, sera confondu avec le plus proche."), unsafe_allow_html=True)
    else:
        c_in, c_hud, c_top = st.columns([0.95, 1.5, 1.25], gap="large")
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
                alt.GradientStop(color="rgba(31,99,224,0)", offset=0),
                alt.GradientStop(color="rgba(31,99,224,.22)", offset=1)]),
            interpolate="monotone").encode(x=x_enc, y=y_enc)
        line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2.5,
                                       interpolate="monotone").encode(x=x_enc, y=y_enc)
        pts = alt.Chart(df).mark_circle(size=80, opacity=1).encode(
            x=x_enc, y=y_enc,
            color=alt.Color("Prédiction:N", scale=alt.Scale(
                domain=[keep, "Autre panneau"], range=[GREEN, RED]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("Intensité:Q", format=".2f"),
                     alt.Tooltip("Probabilité:Q", format=".1%"), "Prédiction:N"])
        rule = alt.Chart(pd.DataFrame({"x": [level]})).mark_rule(
            strokeDash=[6, 5], color=MARKING, strokeWidth=2).encode(x="x:Q")
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
