"""Démo en direct : reconnaissance de panneaux sur un flux vidéo.

Sources possibles :
    python scripts/live.py                                   # webcam du PC
    python scripts/live.py --source http://192.168.1.23:8080/video   # téléphone (IP Webcam)
    python scripts/live.py --source trajet.mp4               # vidéo enregistrée

Téléphone Android : installer l'application « IP Webcam », mettre le téléphone et le
PC sur le même Wi-Fi, appuyer sur « Start server » et utiliser l'adresse affichée
en bas de l'écran du téléphone, suivie de /video.

Touches : Q ou Échap = quitter, P = pause, S = capture d'écran (reports/live/).

Le tableau de bord à droite imite celui d'une voiture :
  - la dernière limitation de vitesse confirmée reste affichée jusqu'à la suivante
    (ou jusqu'à un panneau de fin de limitation) ;
  - les autres panneaux (danger, stop...) s'affichent quelques secondes ;
  - un panneau n'est confirmé qu'après avoir été vu sur plusieurs images
    consécutives (filtrage temporel contre les fausses alertes).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from clocker import config  # noqa: E402
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
from clocker.scene import recognise  # noqa: E402

PANEL_W = 300
GRAPHITE = (69, 74, 81)
CONCRETE = (212, 215, 219)
CAT_RGB = {SPEED_LIMIT: (29, 95, 191), DANGER: (215, 38, 61), PROHIBITION: (215, 38, 61),
           MANDATORY: (29, 95, 191), PRIORITY: (245, 197, 24), END: (154, 161, 169)}
END_OF_LIMIT = {6, 32}       # fin de limitation 80, fin de toutes les interdictions
ALERT_SECONDS = 3.0


# --------------------------------------------------------------------------- #
# Lecture du flux
# --------------------------------------------------------------------------- #
class LatestFrameReader:
    """Lit un flux en continu dans un thread et ne garde que l'image la plus récente.

    Indispensable pour un flux réseau (téléphone) : sinon les images s'accumulent
    dans un tampon et l'affichage prend de plus en plus de retard.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap, self.frame, self.ok = cap, None, True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.ok:
            ok, frame = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:
                self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            frame, self.frame = self.frame, None
        return (frame is not None or self.ok), frame


def open_source(src: str) -> tuple[cv2.VideoCapture, bool]:
    """Renvoie (capture, est_un_fichier)."""
    is_file = Path(src).exists()
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        msg = f"Impossible d'ouvrir la source « {src} »."
        if src.startswith("http"):
            msg += ("\n - Le téléphone et le PC sont-ils sur le même réseau Wi-Fi ?"
                    "\n - Le serveur IP Webcam est-il démarré (« Start server ») ?"
                    "\n - L'adresse se termine-t-elle par /video ? Ex. : "
                    "http://192.168.1.23:8080/video"
                    "\n - Testez l'adresse dans un navigateur du PC.")
        sys.exit(msg)
    return cap, is_file


# --------------------------------------------------------------------------- #
# Dessin
# --------------------------------------------------------------------------- #
def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


FONTS = {k: load_font(*v) for k, v in {
    "small": (15, False), "body": (18, False), "bold": (20, True), "big": (34, True),
}.items()}


def pictogram(class_id: int, size: int) -> Image.Image:
    """Pictogramme officiel GTSRB (data/gtsrb/Meta), sinon un disque coloré."""
    path = config.RAW_DIR / "Meta" / f"{class_id}.png"
    tile = Image.new("RGB", (size, size), (255, 255, 255))
    if path.exists():
        pic = Image.open(path).convert("RGBA")
        pic.thumbnail((size - 10, size - 10))
        tile.paste(pic, ((size - pic.width) // 2, (size - pic.height) // 2), pic)
    else:
        d = ImageDraw.Draw(tile)
        d.ellipse((6, 6, size - 6, size - 6), outline=CAT_RGB[class_category(class_id)],
                  width=max(4, size // 9))
        d.text((size // 2, size // 2), str(SPEED_VALUES.get(class_id, class_id)),
               fill=(20, 20, 20), font=FONTS["big"], anchor="mm")
    return tile


_PICTO_CACHE: dict[tuple[int, int], Image.Image] = {}


def picto(class_id: int, size: int) -> Image.Image:
    key = (class_id, size)
    if key not in _PICTO_CACHE:
        _PICTO_CACHE[key] = pictogram(class_id, size)
    return _PICTO_CACHE[key]


def short(cid: int) -> str:
    return f"{SPEED_VALUES[cid]} km/h" if cid in SPEED_VALUES else class_name(cid)


def render(frame_rgb: np.ndarray, dets: list, confirmed: set[int], limit: int | None,
           alert: tuple[int, float] | None, fps: float, model_name: str,
           paused: bool) -> np.ndarray:
    """Image affichée : vidéo avec boîtes + tableau de bord à droite."""
    h = frame_rgb.shape[0]
    video = Image.fromarray(frame_rgb)
    d = ImageDraw.Draw(video)
    lw = max(2, h // 240)
    for det in dets:
        x1, y1, x2, y2 = det.box
        ok = det.class_id in confirmed
        color = CAT_RGB.get(class_category(det.class_id), (60, 200, 60)) if ok else (230, 230, 230)
        d.rectangle((x1, y1, x2, y2), outline=color, width=lw if ok else 1)
        if ok:
            label = f" {short(det.class_id)}  {det.conf:.0%} "
            tw = d.textlength(label, font=FONTS["body"])
            d.rectangle((x1, y1 - 24, x1 + tw, y1), fill=color)
            d.text((x1, y1 - 22), label, fill=(255, 255, 255), font=FONTS["body"])

    panel = Image.new("RGB", (PANEL_W, h), CONCRETE)
    p = ImageDraw.Draw(panel)
    # bande de marquage au sol sur le bord gauche du panneau
    for y in range(0, h, 44):
        p.rectangle((0, y, 5, y + 26), fill=GRAPHITE)
    x0, y = 22, 18
    p.text((x0, y), "Clocker", fill=(38, 41, 45), font=FONTS["big"])
    y += 46
    p.text((x0, y), "Limitation en cours", fill=GRAPHITE, font=FONTS["bold"])
    y += 30
    if limit is not None:
        panel.paste(picto(limit, 150), (x0, y))
    else:
        p.rounded_rectangle((x0, y, x0 + 150, y + 150), radius=12, outline=(150, 155, 162),
                            width=2)
        p.text((x0 + 75, y + 75), "aucune", fill=(110, 115, 122), font=FONTS["body"],
               anchor="mm")
    y += 170
    p.text((x0, y), "Dernière alerte", fill=GRAPHITE, font=FONTS["bold"])
    y += 30
    if alert is not None:
        cid, _ = alert
        color = CAT_RGB.get(class_category(cid), GRAPHITE)
        p.rounded_rectangle((x0 - 4, y - 4, PANEL_W - 18, y + 98), radius=10, fill=color)
        panel.paste(picto(cid, 90), (x0 + 2, y + 2))
        name = class_name(cid)
        lines, line = [], ""
        for word in name.split():
            trial = (line + " " + word).strip()
            if p.textlength(trial, font=FONTS["small"]) > PANEL_W - 140:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        for k, ln in enumerate(lines[:4]):
            p.text((x0 + 102, y + 6 + 20 * k), ln, fill=(255, 255, 255), font=FONTS["small"])
    else:
        p.text((x0, y), "aucune", fill=(110, 115, 122), font=FONTS["body"])
    y += 120
    p.text((x0, y), f"{fps:5.1f} images/s", fill=(38, 41, 45), font=FONTS["bold"])
    p.text((x0, y + 28), f"Modèle : {model_name}", fill=GRAPHITE, font=FONTS["small"])
    if paused:
        p.text((x0, y + 52), "EN PAUSE (P)", fill=(215, 38, 61), font=FONTS["bold"])
    p.text((x0, h - 30), "Q quitter   P pause   S capture", fill=GRAPHITE, font=FONTS["small"])
    return np.hstack([np.asarray(video), np.asarray(panel)])


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Reconnaissance de panneaux en direct")
    ap.add_argument("--source", default="0",
                    help="0 = webcam, URL du téléphone (…/video) ou fichier vidéo")
    ap.add_argument("--model", default=config.MODEL_PATH.stem)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="confiance minimale pour afficher un panneau (plus haut = moins "
                         "de fausses alertes)")
    ap.add_argument("--confirm", type=int, default=2,
                    help="nombre d'images, parmi les 3 dernières, où le panneau doit être vu")
    ap.add_argument("--hough", action="store_true", help="ajoute les cercles de Hough")
    ap.add_argument("--width", type=int, default=960, help="largeur de traitement en pixels")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotation de l'image (téléphone tenu à la verticale : 90 ou 270)")
    ap.add_argument("--record", default=None, help="enregistre la démo dans ce fichier .mp4")
    ap.add_argument("--no-window", action="store_true", help="sans fenêtre (tests)")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()

    import tensorflow as tf

    model_path = config.MODELS_DIR / f"{a.model}.keras"
    if not model_path.exists():
        sys.exit(f"Modèle introuvable : {model_path}")
    model = tf.keras.models.load_model(model_path)
    prep = config.load_prep(a.model)
    cap, is_file = open_source(a.source)
    reader = None if is_file else LatestFrameReader(cap)

    history: deque[set[int]] = deque(maxlen=3)
    limit: int | None = None
    alert: tuple[int, float] | None = None
    writer, fps, n, paused, last = None, 0.0, 0, False, None
    shots = config.REPORTS_DIR / "live"
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(a.rotate)
    print("Démo en direct lancée. Q pour quitter.")

    while True:
        t0 = time.perf_counter()
        if not paused or last is None:
            if reader is None:
                ok, frame = cap.read()
            else:
                ok, frame = reader.read()
                if ok and frame is None:  # pas encore de nouvelle image
                    time.sleep(0.005)
                    continue
            if not ok or frame is None:
                break
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            scale = a.width / frame.shape[1]
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = recognise(model, rgb, prep["clahe"], prep["denoise"], a.threshold, a.hough)

            # Filtrage temporel : confirmé si vu dans `confirm` des 3 dernières images
            history.append({dt.class_id for dt in dets})
            counts: dict[int, int] = {}
            for s in history:
                for c in s:
                    counts[c] = counts.get(c, 0) + 1
            confirmed = {c for c, k in counts.items() if k >= a.confirm}
            now = time.time()
            for cid in confirmed:
                if cid in SPEED_VALUES:
                    limit = cid
                elif cid in END_OF_LIMIT:
                    limit, alert = None, (cid, now)
                else:
                    alert = (cid, now)
            if alert is not None and now - alert[1] > ALERT_SECONDS:
                alert = None
            last = (rgb, dets, confirmed)

        rgb, dets, confirmed = last
        view = render(rgb, dets, confirmed, limit, alert, fps, a.model, paused)
        bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        if a.record:
            if writer is None:
                Path(a.record).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                         (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)

        if not a.no_window:
            cv2.imshow("Clocker - reconnaissance en direct", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("s"):
                shots.mkdir(parents=True, exist_ok=True)
                out = shots / f"capture_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(out), bgr)
                print(f"Capture : {out}")

        dt = time.perf_counter() - t0
        fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / max(dt, 1e-6)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break

    if reader is not None:
        reader.ok = False
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Vidéo enregistrée : {a.record}")
    if not a.no_window:
        cv2.destroyAllWindows()
    print(f"{n} images traitées, {fps:.1f} images/s en moyenne (lissée).")


if __name__ == "__main__":
    main()
