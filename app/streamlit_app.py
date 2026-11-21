"""Démo Clocker : déposer une image et afficher le panneau prédit avec sa confiance.

Lancement :
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clocker import config  # noqa: E402
from clocker.detection import crop, detect_signs  # noqa: E402
from clocker.labels import DANGER, PROHIBITION, SPEED_LIMIT, SPEED_VALUES  # noqa: E402
from clocker.preprocessing import apply_clahe, denoise_bilateral  # noqa: E402
from clocker.robustness import gaussian_blur, gaussian_noise  # noqa: E402

ALERT_COLORS = {SPEED_LIMIT: "#1f6feb", DANGER: "#f0883e", PROHIBITION: "#da3633"}
BOX_COLORS = {SPEED_LIMIT: (31, 111, 235), DANGER: (240, 136, 62), PROHIBITION: (218, 54, 51)}

st.set_page_config(page_title="Clocker — Panneaux routiers", page_icon="🚦", layout="wide")


@st.cache_resource
def get_model():
    from clocker.inference import load_model
    return load_model(config.MODEL_PATH)


def perturb(img: np.ndarray, blur: float, noise: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = gaussian_noise(gaussian_blur(x, blur), noise)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def dashboard_alert(pred) -> None:
    color = ALERT_COLORS.get(pred.category, "#6e7781")
    big = f"{SPEED_VALUES[pred.class_id]} km/h" if pred.class_id in SPEED_VALUES else pred.name
    st.markdown(
        f"""<div style="border-left:8px solid {color};padding:14px 18px;border-radius:6px;
        background:rgba(127,127,127,0.08)">
        <div style="font-size:0.85rem;opacity:0.7">{pred.category.upper()}</div>
        <div style="font-size:2rem;font-weight:700">{big}</div>
        <div>{pred.name} — confiance <b>{pred.confidence:.1%}</b></div></div>""",
        unsafe_allow_html=True,
    )
    if pred.confidence < 0.6:
        st.warning("Confiance faible : l'alerte ne serait pas affichée au conducteur.")


st.title("🚦 Clocker — Reconnaissance de panneaux de signalisation")

with st.sidebar:
    st.header("Paramètres")
    mode = st.radio("Type d'image", ["Panneau recadré", "Scène complète (détection)"])
    prep = config.load_prep(config.MODEL_PATH.stem)
    use_denoise = st.checkbox("Débruitage bilatéral", value=prep["denoise"])
    use_clahe = st.checkbox("Prétraitement CLAHE", value=prep["clahe"])
    st.subheader("Test de robustesse")
    blur = st.slider("Flou gaussien (sigma)", 0.0, 4.0, 0.0, 0.25)
    noise = st.slider("Bruit gaussien (écart-type)", 0.0, 0.3, 0.0, 0.01)

if not config.MODEL_PATH.exists():
    st.error(
        f"Modèle introuvable ({config.MODEL_PATH.relative_to(config.ROOT_DIR)}). "
        "Entraînez-le d'abord : `python -m clocker.train`"
    )
    st.stop()

uploaded = st.file_uploader("Déposez une image", type=["png", "jpg", "jpeg", "ppm", "bmp"])
if uploaded is None:
    st.info("Déposez une image de panneau pour lancer la prédiction.")
    st.stop()

buf = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Image illisible.")
    st.stop()
image = perturb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), blur, noise)

from clocker.inference import predict_image  # noqa: E402

model = get_model()

if mode == "Panneau recadré":
    t0 = time.perf_counter()
    pred = predict_image(model, image, use_clahe=use_clahe, denoise=use_denoise)
    ms = 1000 * (time.perf_counter() - t0)

    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.image(image, caption="Image d'entrée", width="stretch")
    shown_prep = denoise_bilateral(image) if use_denoise else image
    shown_prep = apply_clahe(shown_prep) if use_clahe else shown_prep
    c2.image(shown_prep, caption="Après prétraitement", width="stretch")
    with c3:
        dashboard_alert(pred)
        st.caption(f"Temps d'inférence : {ms:.1f} ms")
        st.subheader("Top 3")
        for cid, name, p in pred.top_k:
            st.progress(p, text=f"{cid} — {name} : {p:.1%}")
else:
    cands = detect_signs(image)
    shown = image.copy()
    results = []
    for cand in cands:
        pred = predict_image(model, crop(image, cand), use_clahe=use_clahe, denoise=use_denoise)
        if pred.confidence < 0.6:
            continue
        results.append((cand, pred))
        color = BOX_COLORS.get(pred.category, (60, 200, 60))
        x1, y1, x2, y2 = cand.box
        cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
        cv2.putText(shown, f"{pred.class_id} {pred.confidence:.0%}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    st.image(shown, caption=f"{len(cands)} candidats, {len(results)} panneaux retenus",
             width="stretch")
    for _, pred in sorted(results, key=lambda r: -r[1].confidence):
        dashboard_alert(pred)
