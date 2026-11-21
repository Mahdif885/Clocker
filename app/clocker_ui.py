"""Clocker — interface de démonstration complète.

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
# Identité visuelle : couleurs de la signalisation routière
# --------------------------------------------------------------------------- #
BLUE = "#0B4EA2"      # panneaux de direction / obligation
RED = "#C1121F"       # bordure des panneaux de danger et d'interdiction
YELLOW = "#F2B705"    # route prioritaire
ASPHALT = "#2B2F35"   # texte, bandeau
CONCRETE = "#F3F4F5"  # fond
GREY = "#8A9099"      # fins d'interdiction, éléments secondaires

CATEGORY_COLORS = {
    SPEED_LIMIT: BLUE, DANGER: RED, PROHIBITION: RED,
    MANDATORY: BLUE, PRIORITY: "#B88900", END: GREY,
}
BOX_BGR = {  # couleurs des boîtes (OpenCV, RGB car les images sont en RGB)
    SPEED_LIMIT: (11, 78, 162), DANGER: (193, 18, 31), PROHIBITION: (193, 18, 31),
    MANDATORY: (11, 78, 162), PRIORITY: (242, 183, 5), END: (138, 144, 153),
}
SCALED_WITH_SIZE = {"Flou gaussien (sigma)", "Flou de bougé (px)"}

st.set_page_config(page_title="Clocker", page_icon="🚸", layout="wide",
                   initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
.stApp {{ font-family: 'Barlow', system-ui, sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family: 'Barlow Condensed', 'Barlow', sans-serif; font-weight: 600;
  letter-spacing: .01em; color: {ASPHALT};
}}
.block-container {{ padding-top: 3.4rem; max-width: 1400px; }}

/* En-tête : nom + bande d'asphalte avec marquage au sol */
.ck-head {{ display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
.ck-word {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: 3rem; line-height: 1; color: {ASPHALT}; }}
.ck-sub {{ font-size: 1.05rem; color: #5A6068; }}
.ck-road {{ height: 14px; margin: .7rem 0 1.2rem; border-radius: 2px;
  background:
    linear-gradient(90deg, #fff 0 46px, transparent 46px 80px) 0 50% / 80px 3px repeat-x,
    {ASPHALT}; }}

/* Panneau de résultat, inspiré des panneaux de direction */
.ck-plate {{ background: {BLUE}; border-radius: 14px; padding: 7px; color: #fff; }}
.ck-plate.uncertain {{ background: #5F656D; }}
.ck-plate-in {{ border: 3px solid #fff; border-radius: 9px; padding: 1.1rem 1.2rem;
  display: flex; flex-wrap: wrap; gap: 1.1rem 1.2rem; align-items: center; }}
.ck-text {{ flex: 1 1 170px; min-width: 0; overflow-wrap: break-word; }}
.ck-sign {{ flex: 0 0 124px; height: 124px; background: #fff; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; }}
.ck-sign img, .ck-sign svg {{ max-width: 108px; max-height: 108px; }}
.ck-name {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: clamp(1.6rem, 2.3vw, 2.3rem); line-height: 1.05; }}
.ck-plate.compact .ck-plate-in {{ padding: .8rem; gap: .7rem; }}
.ck-plate.compact .ck-sign {{ flex-basis: 84px; height: 84px; }}
.ck-plate.compact .ck-sign img {{ max-width: 72px; max-height: 72px; }}
.ck-plate.compact .ck-name {{ font-size: 1.5rem; }}
.ck-plate.compact .ck-note-in {{ display: none; }}
.ck-cat {{ font-size: 1rem; opacity: .85; margin-top: .25rem; }}
.ck-conf {{ margin-top: .8rem; font-size: .95rem; }}
.ck-meter {{ height: 8px; background: rgba(255,255,255,.25); border-radius: 4px;
  margin-top: .35rem; overflow: hidden; }}
.ck-meter > div {{ height: 100%; background: #fff; border-radius: 4px; }}
.ck-note {{ color: #5A6068; font-size: .9rem; margin-top: .5rem; }}

/* Petits panneaux (journal, scène, catalogue) */
.ck-chip {{ display: flex; gap: .6rem; align-items: center; padding: .35rem 0;
  border-bottom: 1px solid #DDE0E4; }}
.ck-chip img {{ width: 34px; height: 34px; object-fit: contain; }}
.ck-chip b {{ font-weight: 600; }}
.ck-chip span {{ color: #5A6068; font-size: .85rem; }}
.ck-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr));
  gap: .8rem; }}
.ck-cell {{ background: #fff; border: 1px solid #DDE0E4; border-radius: 8px;
  padding: .6rem; text-align: center; font-size: .82rem; line-height: 1.2; }}
.ck-cell img, .ck-cell svg {{ width: 56px; height: 56px; object-fit: contain;
  display: block; margin: 0 auto .4rem; }}
.ck-cell small {{ color: {GREY}; }}
.ck-step {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 600;
  font-size: 1.05rem; color: {ASPHALT}; margin-bottom: .2rem; }}
.ck-step em {{ font-style: normal; color: {BLUE}; margin-right: .3rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


def plate_html(cid: int, conf: float, threshold: float, compact: bool = False) -> str:
    ok = conf >= threshold
    title = short_name(cid) if ok else "Panneau incertain"
    sub = class_category(cid) if ok else f"Meilleure hypothèse : {class_name(cid)}"
    note = ("Alerte affichée au conducteur" if ok
            else f"Confiance sous le seuil de {fmt_pct(threshold)} : aucune alerte")
    return f"""
<div class="ck-plate{'' if ok else ' uncertain'}{' compact' if compact else ''}" role="status">
 <div class="ck-plate-in">
  <div class="ck-sign"><img src="{pictogram_uri(cid)}" alt="{class_name(cid)}"></div>
  <div class="ck-text">
   <div class="ck-name">{title}</div>
   <div class="ck-cat">{sub}</div>
   <div class="ck-conf">Confiance {fmt_pct(conf)}<div class="ck-meter">
     <div style="width:{100 * conf:.1f}%"></div></div></div>
   <div class="ck-cat ck-note-in" style="margin-top:.6rem">{note}</div>
  </div>
 </div>
</div>"""


def topk_chart(probs: np.ndarray, k: int = 5) -> alt.Chart:
    order = np.argsort(probs)[::-1][:k]
    df = pd.DataFrame({
        "Panneau": [f"{class_name(i)}  ({i})" for i in order],
        "Probabilité": [float(probs[i]) for i in order],
        "rang": list(range(k)),
    })
    base = alt.Chart(df).encode(
        y=alt.Y("Panneau:N", sort=alt.SortField("rang"), title=None,
                axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        x=alt.X("Probabilité:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title=None, grid=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=22).encode(
        color=alt.condition(alt.datum.rang == 0, alt.value(BLUE), alt.value("#B9BEC5")),
        tooltip=[alt.Tooltip("Panneau:N"), alt.Tooltip("Probabilité:Q", format=".2%")],
    )
    text = base.mark_text(align="left", dx=4, fontSize=12, color=ASPHALT).encode(
        text=alt.Text("Probabilité:Q", format=".1%"))
    return (bars + text).properties(height=k * 38).configure_view(stroke=None)


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
# En-tête
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="ck-head"><div class="ck-word">Clocker</div>'
    '<div class="ck-sub">Reconnaissance de panneaux routiers pour l\'aide à la conduite</div></div>'
    '<div class="ck-road" aria-hidden="true"></div>',
    unsafe_allow_html=True,
)

models = available_models()
if not models:
    st.error("Aucun modèle dans le dossier models. Entraînez-en un avec "
             "`python -m clocker.train`, puis relancez cette page.")
    st.stop()

# --------------------------------------------------------------------------- #
# Barre latérale : source, modèle, réglages
# --------------------------------------------------------------------------- #
default_model = config.MODEL_PATH.stem if config.MODEL_PATH.stem in models else models[0]
demo_dir = ROOT / "assets" / "demo"
demo_files = sorted(p for p in demo_dir.glob("*") if p.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}) if demo_dir.exists() else []
tests = test_index()

with st.sidebar:
    st.subheader("Image à analyser")
    sources = ["Importer"]
    if tests is not None:
        sources.append("Exemples GTSRB")
    if demo_files:
        sources.append("Images de démo")
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

    st.subheader("Modèle")
    model_name = st.selectbox("Modèle", models, index=models.index(default_model),
                              label_visibility="collapsed")
    prep = config.load_prep(model_name)
    use_denoise = st.toggle("Débruitage bilatéral", value=prep["denoise"],
                            help="Réglage utilisé à l'entraînement de ce modèle par défaut.")
    use_clahe = st.toggle("CLAHE (canal L, espace Lab)", value=prep["clahe"])
    auto_crop = st.toggle("Recadrage automatique", value=True,
                          help="Cherche le panneau rouge ou bleu principal et le recadre "
                               "comme les images GTSRB avant de le classifier.")
    threshold = st.slider("Seuil d'alerte", 0.0, 1.0, 0.6, 0.05,
                          help="En dessous, le système n'alerte pas le conducteur.")

model = get_model(model_name)
size = model.input_shape[1]

tab_rec, tab_pipe, tab_rob, tab_scene, tab_res, tab_cat = st.tabs([
    "Reconnaissance", "Étapes du traitement", "Robustesse", "Scène complète",
    "Résultats du modèle", "Panneaux reconnus",
])


def catalogue_html() -> str:
    cells = "".join(
        f'<div class="ck-cell"><img src="{pictogram_uri(i)}" alt="">{class_name(i)}'
        f'<br><small>classe {i}</small></div>' for i in range(config.NUM_CLASSES))
    return f'<div class="ck-grid">{cells}</div>'


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

    # Journal de la session (une entrée par image et par modèle)
    key = hashlib.md5(image.tobytes()[:200000] + model_name.encode()).hexdigest()
    hist = st.session_state.setdefault("history", [])
    if not hist or hist[0]["key"] != key:
        hist.insert(0, dict(key=key, thumb=thumb_uri(sign), cid=cid,
                            conf=result["conf"], label=image_label))
        del hist[12:]

with st.sidebar:
    hist = st.session_state.get("history", [])
    if hist:
        st.subheader("Journal")
        st.markdown("".join(
            f'<div class="ck-chip"><img src="{h["thumb"]}" alt=""><div><b>'
            f'{short_name(h["cid"])}</b><br><span>{fmt_pct(h["conf"])}</span></div></div>'
            for h in hist), unsafe_allow_html=True)
        if st.button("Vider le journal", type="tertiary"):
            st.session_state["history"] = []
            st.rerun()

# --------------------------------------------------------------------------- #
# Onglet 1 : reconnaissance
# --------------------------------------------------------------------------- #
with tab_rec:
    if result is None:
        st.markdown("#### Choisissez une image pour commencer")
        st.write("Importez une photo de panneau dans le panneau de gauche, ou prenez un "
                 "exemple du jeu de test GTSRB. Le modèle reconnaît les 43 panneaux de "
                 "l'onglet *Panneaux reconnus* : un panneau absent de cette liste, comme "
                 "le 110 km/h français, sera confondu avec le plus proche.")
    else:
        c_in, c_plate, c_top = st.columns([1, 1.45, 1.1], gap="large")
        with c_in:
            shown = image
            if result["cand"] is not None:
                shown = draw_box(image, result["cand"], (11, 78, 162))
            st.image(shown, width="stretch", caption=image_label)
            if auto_crop and result["cand"] is None:
                st.caption("Aucun panneau rouge ou bleu détecté : l'image entière est classifiée.")
        with c_plate:
            st.markdown(plate_html(result["cid"], result["conf"], threshold),
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="ck-note">Inférence en {result["ms"]:.1f} ms sur {size}×{size} px '
                f'avec le modèle {model_name}. La première prédiction est plus lente '
                f'(chargement de TensorFlow).</div>', unsafe_allow_html=True)
        with c_top:
            st.markdown("#### Hypothèses du modèle")
            st.altair_chart(topk_chart(result["probs"]), width="stretch")
            st.markdown("#### Ce que voit le réseau")
            a, b = st.columns(2)
            a.image(pixelated(result["x"][0], 160), caption=f"Entrée {size}×{size}")
            cam = grad_cam(model_name, result["x"], result["cid"])
            b.image(overlay_cam((result["x"][0] * 255).astype(np.uint8), cam, 160),
                    caption="Zones décisives")

# --------------------------------------------------------------------------- #
# Onglet 2 : étapes du traitement
# --------------------------------------------------------------------------- #
with tab_pipe:
    if result is None:
        st.info("Choisissez d'abord une image dans le panneau de gauche.")
    else:
        st.write("Chaque étape ci-dessous est exactement celle appliquée avant la "
                 "classification. Les images de 32×32 pixels sont agrandies sans lissage.")
        mask = color_mask(image, RED_RANGES) | color_mask(image, BLUE_RANGES)
        mask_view = image.copy()
        mask_view[mask == 0] = (mask_view[mask == 0] * 0.25).astype(np.uint8)
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
        for row_start in range(0, len(steps), 4):
            cols = st.columns(4, gap="medium")
            for col, (i, (title, img, text)) in zip(
                    cols, enumerate(steps[row_start:row_start + 4], start=row_start + 1)):
                with col:
                    st.markdown(f'<div class="ck-step"><em>{i}</em>{title}</div>',
                                unsafe_allow_html=True)
                    st.image(img, width="stretch")
                    st.caption(text)

# --------------------------------------------------------------------------- #
# Onglet 3 : robustesse
# --------------------------------------------------------------------------- #
with tab_rob:
    if result is None:
        st.info("Choisissez d'abord une image dans le panneau de gauche.")
    else:
        st.write("La dégradation est appliquée à l'image brute, avant le prétraitement, "
                 "comme sur une vraie caméra (même protocole que `clocker.evaluate`).")
        names = list(PERTURBATIONS)
        c_ctrl, c_view = st.columns([1, 2], gap="large")
        with c_ctrl:
            pert = st.selectbox("Dégradation", names)
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
            a, b, c = st.columns([1, 1, 1.5])
            a.image(pixelated(raw), caption="Intacte", width="stretch")
            b.image(pixelated(xp), caption="Dégradée puis prétraitée", width="stretch")
            with c:
                st.markdown(plate_html(pc, float(pp[pc]), threshold, compact=True),
                            unsafe_allow_html=True)

        sweep = np.linspace(levels[0], levels[-1], 21)
        batch = np.stack([degrade(float(lv)) for lv in sweep])
        ps = predict_batch(model, batch)
        ref = result["cid"]
        df = pd.DataFrame({
            "Intensité": sweep,
            "Probabilité": ps[:, ref],
            "Prédiction": [("Toujours " + short_name(ref)) if p.argmax() == ref
                           else "Autre panneau" for p in ps],
        })
        x_enc = alt.X("Intensité:Q", title=pert,
                      scale=alt.Scale(reverse=is_factor, domain=[lo, hi]),
                      axis=alt.Axis(tickCount=8, format="~g", grid=False))
        line = alt.Chart(df).mark_line(color=GREY).encode(x=x_enc, y="Probabilité:Q")
        pts = alt.Chart(df).mark_circle(size=70).encode(
            x=x_enc,
            y=alt.Y("Probabilité:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                    title=f"Probabilité de « {short_name(ref)} »"),
            color=alt.Color("Prédiction:N", scale=alt.Scale(
                domain=[f"Toujours {short_name(ref)}", "Autre panneau"], range=[BLUE, RED]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("Intensité:Q", format=".2f"),
                     alt.Tooltip("Probabilité:Q", format=".1%"), "Prédiction:N"],
        )
        rule = alt.Chart(pd.DataFrame({"x": [level]})).mark_rule(
            strokeDash=[4, 3], color=ASPHALT).encode(x="x:Q")
        st.markdown("#### Jusqu'où le modèle tient-il sur cette image ?")
        st.altair_chart((line + pts + rule).properties(height=280).configure_view(stroke=None),
                        width="stretch")
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
        st.info("Choisissez d'abord une image dans le panneau de gauche, "
                "de préférence une photo de rue avec plusieurs panneaux.")
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
                    boxed = draw_box(boxed, c, BOX_BGR[class_category(k)],
                                     f"{short_name(k)} {100 * p[k]:.0f}%")
        c_img, c_list = st.columns([2, 1], gap="large")
        with c_img:
            st.image(boxed, width="stretch",
                     caption=f"Zones candidates : {len(cands)}. Panneaux au-dessus du seuil "
                             f"d'alerte : {len(found)}.")
        with c_list:
            st.markdown("#### Panneaux retenus")
            if not found:
                st.write("Aucun panneau au-dessus du seuil. Baissez le seuil d'alerte ou "
                         "essayez avec les cercles de Hough.")
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
    mfile = rep_dir / "metrics.json"
    if not mfile.exists():
        st.info(f"Pas encore de résultats pour {model_name}. Lancez "
                f"`python -m clocker.evaluate --name {model_name}`.")
    else:
        try:
            m = json.loads(mfile.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            st.warning(f"{mfile.name} est illisible : relancez l'évaluation de {model_name}.")
            st.stop()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy (test GTSRB)", fmt_pct(m["accuracy"]))
        k2.metric("Macro-F1", fmt_pct(m["macro_f1"]))
        k3.metric("Latence par image", f"{m['latence_ms_image']:.1f} ms".replace(".", ","))
        k4.metric("Paramètres", f"{m['nb_parametres']:,}".replace(",", " "))
        st.caption(f"{m['nb_images_test']} images de test jamais vues à l'entraînement.")

        comp = config.REPORTS_DIR / "comparaison.csv"
        comp_df = read_table(comp) if comp.exists() else None
        if comp_df is not None:
            st.markdown("#### Comparaison des expériences")
            st.dataframe(comp_df, hide_index=True, width="stretch")

        figs = [("Courbes d'apprentissage", config.FIGURES_DIR / f"{model_name}_curves.png"),
                ("Matrice de confusion", fig_dir / "confusion_matrix.png"),
                ("Robustesse", fig_dir / "robustness.png"),
                ("Erreurs les plus confiantes", fig_dir / "failures.png")]
        figs = [(t, p) for t, p in figs if p.exists()]
        for i in range(0, len(figs), 2):
            cols = st.columns(2, gap="large")
            for col, (t, p) in zip(cols, figs[i:i + 2]):
                col.markdown(f"#### {t}")
                col.image(str(p), width="stretch")

        conf_csv = rep_dir / "top_confusions.csv"
        conf_df = read_table(conf_csv) if conf_csv.exists() else None
        if conf_df is not None:
            st.markdown("#### Confusions les plus fréquentes")
            st.dataframe(conf_df, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Onglet 6 : catalogue
# --------------------------------------------------------------------------- #
with tab_cat:
    st.write("Le modèle choisit toujours parmi ces 43 panneaux du jeu allemand GTSRB. "
             "Un panneau absent de la liste est classé dans la catégorie la plus proche, "
             "parfois avec une forte confiance.")
    st.markdown(catalogue_html(), unsafe_allow_html=True)

