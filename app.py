import os
import cv2
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# =========================
# Branding
APP_TITLE = "ExamSafe AI"

st.set_page_config(page_title=f"{APP_TITLE}", layout="centered")

st.title(APP_TITLE)
st.markdown("""
**ExamSafe AI** is an intelligent proctoring assistant that uses real-time object detection  
to enhance exam integrity. The system automatically recognizes prohibited items such as  
cell phones, laptops or books during examinations.

Click **▶️ Start** to activate live monitoring.  
If any forbidden object is detected, an alert will be displayed immediately.
""")
# =========================

# ---- YOLOv5 Model Cache ----
@st.cache_resource(show_spinner=True)
def load_model():
    import torch
    # essaie de charger un modèle un peu plus précis (yolov5m)
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
    except Exception:
        # fallback vers yolov5s si m indisponible
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.25  # plus tolérant pour petites détections
    model.iou = 0.45
    return model

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

HIGHLIGHT = {"book", "cell phone", "bottle", "cup", "laptop", "scissors","person"}

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
        color = (0, 255, 0)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(
            frame,
            f"{d.label} {d.conf:.2f}",
            (d.x1, max(0, d.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame

# ---------- Video Transformer ----------
class YoloTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.convertScaleAbs(img, alpha=1.15, beta=15)

        results = self.model(img, size=960)
        df = results.pandas().xyxy[0]

        dets: List[Detection] = []
        for _, r in df.iterrows():
            name = str(r["name"])
            if HIGHLIGHT and name not in HIGHLIGHT:
                continue
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

# --------- UI Buttons ----------
if "run" not in st.session_state:
    st.session_state.run = False

c1, c2 = st.columns(2)
if c1.button("▶️ Start", use_container_width=True):
    st.session_state.run = True
if c2.button("⏹ Stop", use_container_width=True):
    st.session_state.run = False

st.markdown("---")

# --------- Video Stream ----------
if st.session_state.run:
    st.info("🟢 Detecting…")
    webrtc_streamer(
        key="smartdesk-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        video_transformer_factory=YoloTransformer,
        async_processing=True,
    )
else:
    st.info("⚪ Waiting. Click **Start** to activate the camera.")