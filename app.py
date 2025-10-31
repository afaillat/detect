import os
import cv2
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List

# --- vidéo intégrée ---
# pip install streamlit-webrtc av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# =========================
# Branding (change le titre plus tard)
APP_TITLE = "Titre de l’application"
TAGLINE = "Détection d’objets en temps réel (webcam intégrée)"
# =========================

st.set_page_config(page_title=f"{APP_TITLE}", layout="centered")
st.title(f"🧠 {APP_TITLE}")
st.caption(TAGLINE)

# ---- cache du modèle YOLO (une seule fois) ----
@st.cache_resource(show_spinner=True)
def load_model():
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.35  # seuil de confiance
    model.iou = 0.45
    return model

# Pour WebRTC sur localhost (pas besoin de TURN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Classes qu’on met en avant (tu peux affiner)
HIGHLIGHT = {"book", "cell phone", "bottle", "cup", "laptop", "scissors"}

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float

def draw_boxes(frame: np.ndarray, dets: List[Detection]) -> np.ndarray:
    for d in dets:
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{d.label} {d.conf:.2f}",
            (d.x1, max(0, d.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame

# ---------- Transformer vidéo (process chaque frame) ----------
class YoloTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # inference
        results = self.model(img, size=640)
        df = results.pandas().xyxy[0]

        dets: List[Detection] = []
        for _, r in df.iterrows():
            name = str(r["name"])
            if HIGHLIGHT and name not in HIGHLIGHT:
                # si tu veux tout afficher, commente les 2 lignes suivantes
                pass
            dets.append(
                Detection(
                    x1=int(r["xmin"]),
                    y1=int(r["ymin"]),
                    x2=int(r["xmax"]),
                    y2=int(r["ymax"]),
                    label=name,
                    conf=float(r["confidence"]),
                )
            )
        out = draw_boxes(img, dets)
        return out

# --------- UI boutons Start/Stop ----------
if "run" not in st.session_state:
    st.session_state.run = False

c1, c2 = st.columns(2)
if c1.button("▶️  Lancer", use_container_width=True):
    st.session_state.run = True
if c2.button("⏹  Arrêter", use_container_width=True):
    st.session_state.run = False

st.markdown("---")

# --------- Zone vidéo intégrée ----------
if st.session_state.run:
    st.info("🟢 Détection en cours…")
    webrtc_streamer(
        key="smartdesk-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=YoloTransformer,  # dessine boîtes + labels
        async_processing=True,
    )
else:
    st.info("⚪ En attente. Clique sur **Lancer** pour activer la caméra.")