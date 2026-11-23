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
from clocker.detection import crop, detect_signs, main_sign  # noqa: E402
from clocker.labels import DANGER, PROHIBITION, SPEED_LIMIT, SPEED_VALUES  # noqa: E402
from clocker.preprocessing import apply_clahe, denoise_bilateral  # noqa: E402
from clocker.robustness import gaussian_blur, gaussian_noise  # noqa: E402

ALERT_COLORS = {SPEED_LIMIT: "#1f6feb", DANGER: "#f0883e", PROHIBITION: "#da3633"}
BOX_COLORS = {SPEED_LIMIT: (31, 111, 235), DANGER: (240, 136, 62), PROHIBITION: (218, 54, 51)}

st.set_page_config(page_title="Clocker — Panneaux routiers", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")

def _hero_road_data_uri() -> str:
    asset = Path(__file__).resolve().parent.parent / "assets" / "ui" / "clocker_hero_road.jpg"
    try:
        import base64
        return "data:image/jpeg;base64," + base64.b64encode(asset.read_bytes()).decode("ascii")
    except OSError:
        return ""


st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{font-family:'DM Sans',sans-serif;color:#102447;background:radial-gradient(ellipse at 80% 0%,rgba(90,145,235,.15),transparent 34%),linear-gradient(135deg,#f8faff,#edf3fc 55%,#e8effb)}
.block-container{max-width:1500px;padding:1.4rem 2rem 3rem}h1,h2,h3{font-family:'Manrope',sans-serif!important;color:#102447!important;letter-spacing:-.03em}
[data-testid="stSidebar"]{background:linear-gradient(165deg,#10294d,#0b1c36 65%,#102b50);border-right:1px solid rgba(255,255,255,.12)}[data-testid="stSidebar"] *{color:#edf4ff!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:rgba(255,255,255,.1);border-radius:10px}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section{background:rgba(255,255,255,.07);border:1px dashed rgba(175,205,255,.55);border-radius:14px}
.stButton button{border:0;border-radius:10px;background:linear-gradient(135deg,#2678f5,#1555ca);color:white;font-weight:700;box-shadow:0 6px 18px rgba(25,103,226,.2)}
.ck-legacy-hero{display:grid;grid-template-columns:1.05fr .95fr;align-items:center;overflow:hidden;min-height:245px;margin:.3rem 0 1.5rem;border-radius:22px;background:linear-gradient(110deg,#fbfdff 0%,#dce9fb 48%,#879fbe 100%);box-shadow:0 18px 44px rgba(31,65,119,.15),inset 0 0 0 1px white}
.ck-legacy-copy{padding:2rem 2.3rem;position:relative;z-index:2}.ck-legacy-title{font:800 clamp(3.5rem,5vw,5.1rem)/.95 'Manrope',sans-serif;letter-spacing:-.07em;color:#102447}
.ck-legacy-sub{font-size:1.12rem;color:#294467;margin-top:.7rem;max-width:38ch}.ck-legacy-road{min-height:245px;position:relative;overflow:hidden;background:linear-gradient(90deg,rgba(220,233,251,.15),transparent 30%),linear-gradient(180deg,#a7bad3,#52647e)}
.ck-legacy-road:before{content:'';position:absolute;left:50%;bottom:-110px;width:420px;height:450px;transform:translateX(-50%) perspective(250px) rotateX(58deg);background:linear-gradient(90deg,transparent 7%,white 7.5% 8.5%,transparent 9% 91%,white 91.5% 92.5%,transparent 93%),repeating-linear-gradient(180deg,white 0 30px,transparent 30px 72px) 50% 0/9px 100% no-repeat,linear-gradient(90deg,#303947,#4b5667 50%,#303947);box-shadow:0 0 30px rgba(15,30,55,.3)}
.ck-legacy-metrics{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.1rem}.ck-legacy-pill{padding:.45rem .75rem;border-radius:9px;background:#1769e8;color:#fff;font-weight:700;font-size:.85rem}.ck-legacy-pill.green{background:#159b68}.ck-legacy-pill.gold{background:#f5b82e;color:#17243a}
.ck-legacy-result{padding:1.25rem 1.4rem;border-radius:16px;border:1px solid #d9e5f5;border-left:6px solid #1769e8;background:linear-gradient(110deg,#fff,#f4f8ff);box-shadow:0 12px 30px rgba(31,65,119,.08);color:#294467}
[data-testid="stImage"] img{border-radius:13px;box-shadow:0 7px 18px rgba(20,48,88,.08)}
@media(max-width:850px){.block-container{padding:1rem}.ck-legacy-hero{grid-template-columns:1fr}.ck-legacy-road{min-height:130px}}
</style>
""",unsafe_allow_html=True)


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
        f"""<div class="ck-legacy-result" style="border-left-color:{color}">
        <div style="font-size:.8rem;letter-spacing:.08em;font-weight:800;color:#66758d">{pred.category.upper()}</div>
        <div style="font:800 2.2rem 'Manrope',sans-serif;color:#102447;margin:.2rem 0">{big}</div>
        <div style="color:#294467">{pred.name} — confiance <b>{pred.confidence:.1%}</b></div></div>""",
        unsafe_allow_html=True,
    )
    if pred.confidence < 0.6:
        st.warning("Confiance faible : l'alerte ne serait pas affichée au conducteur.")


st.markdown("""
<div class="ck-legacy-hero"><div class="ck-legacy-copy"><div class="ck-legacy-title">Clocker</div>
<div class="ck-legacy-sub">Reconnaissance de panneaux routiers<br>pour l'aide à la conduite</div>
<div class="ck-legacy-metrics"><span class="ck-legacy-pill green">43 panneaux GTSRB</span>
<span class="ck-legacy-pill">98,99 % de précision</span><span class="ck-legacy-pill gold">823 k paramètres</span></div></div>
<div class="ck-legacy-road"></div></div>
""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="ck-brand">
      <svg class="ck-brand-mark" viewBox="0 0 100 68" aria-hidden="true">
        <path d="M8 60 L45 4 L55 4 L28 60 Z" fill="#F5F8FF"/>
        <path d="M56 4 L92 60 L76 60 L51 4 Z" fill="#F4B72E"/>
        <path d="M49 15 L54 15 L58 23 L46 23 Z" fill="#172b4d"/>
        <path d="M41 31 L61 31 L66 40 L36 40 Z" fill="#172b4d"/>
        <path d="M31 49 L70 49 L76 60 L24 60 Z" fill="#172b4d"/>
      </svg>
      <div class="ck-brand-name">Clocker</div>
      <div class="ck-brand-tag">Analyse &amp; performance routière</div>
    </div>
    """, unsafe_allow_html=True)
    st.header("Paramètres")
    mode = st.radio(
        "Type d'image",
        ["Automatique", "Panneau recadré", "Scène complète (détection)"],
        help="Automatique : cherche le panneau rouge/bleu principal et le recadre avant "
             "de classifier (comme les images GTSRB). Recadré : l'image entière est le panneau.",
    )
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

if mode in ("Automatique", "Panneau recadré"):
    sign = image
    if mode == "Automatique":
        cand = main_sign(image)
        if cand is None:
            st.info("Aucun panneau rouge/bleu détecté : classification de l'image entière.")
        else:
            sign = crop(image, cand)
    t0 = time.perf_counter()
    pred = predict_image(model, sign, use_clahe=use_clahe, denoise=use_denoise)
    ms = 1000 * (time.perf_counter() - t0)

    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.image(image, caption="Image d'entrée", width="stretch")
    shown_prep = denoise_bilateral(sign) if use_denoise else sign
    shown_prep = apply_clahe(shown_prep) if use_clahe else shown_prep
    c2.image(shown_prep, caption="Panneau recadré, après prétraitement"
             if sign is not image else "Après prétraitement", width="stretch")
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

_legacy_road_uri = _hero_road_data_uri()
st.markdown(f"""
<style>
.ck-legacy-hero {{
  position:relative;isolation:isolate;display:grid;grid-template-columns:1fr 1.12fr;
  align-items:center;overflow:hidden;min-height:248px;margin:.2rem 0 1.35rem;
  border-radius:19px;background-color:#dce7f8;
  background-image:linear-gradient(90deg,#f8faff 0%,rgba(246,249,255,.98) 31%,
    rgba(238,244,255,.84) 44%,rgba(235,242,255,.08) 65%,transparent 100%),url("{_legacy_road_uri}");
  background-size:100% 100%,cover;background-position:center,center 56%;
  box-shadow:0 16px 36px rgba(44,72,125,.17),inset 0 0 0 1px rgba(255,255,255,.9);
}}
.ck-legacy-copy{{padding:1.8rem 2.25rem;position:relative;z-index:2}}
.ck-legacy-title{{font:800 clamp(3.6rem,5.5vw,5.25rem)/.92 'Manrope',sans-serif;letter-spacing:-.075em;color:#101d3d}}
.ck-legacy-sub{{font-size:1.1rem;line-height:1.4;color:#172849;margin-top:.55rem;max-width:36ch}}
.ck-legacy-road{{display:none}}
.ck-legacy-metrics{{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}}
.ck-legacy-pill{{padding:.48rem .72rem;border-radius:8px;background:#1769f5;color:white;font-weight:700;font-size:.82rem;box-shadow:0 5px 12px rgba(16,34,68,.12)}}
.ck-legacy-pill.green{{background:#0eaa68}}.ck-legacy-pill.gold{{background:#ffbd20;color:#15213b}}
.ck-brand{{text-align:center;padding:1rem .3rem 1.25rem;margin:-.3rem 0 .8rem;border-bottom:1px solid rgba(222,234,255,.15)}}
.ck-brand-mark{{width:62px;height:45px;margin:0 auto .35rem;display:block}}
.ck-brand-name{{font:800 1.65rem/1 'Manrope',sans-serif;color:#fff;letter-spacing:-.055em}}
.ck-brand-tag{{font:500 .72rem 'DM Sans',sans-serif;color:#b9c9e5;margin-top:.45rem}}
.ck-legacy-result{{border-radius:14px!important}}
@media(max-width:950px){{.ck-legacy-hero{{grid-template-columns:1fr;min-height:285px;background-position:center,65% center}}.ck-legacy-copy{{padding:1.4rem}}}}
</style>
""",unsafe_allow_html=True)
