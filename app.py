import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from datetime import datetime
import plotly.graph_objects as go
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Surveillance", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.big-title {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.card {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    margin: 10px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚨 AI Smart Surveillance Dashboard</p>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
weapon_model = YOLO("model/weapon_model.pt")
crowd_model = YOLO("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

# ✅ ONLY ALLOW THESE WEAPONS
WEAPON_CLASSES = ["pistol", "knife"]

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Settings")
mode = st.sidebar.radio("Mode", ["Upload Video", "Webcam"])
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)

# ---------------- LAYOUT ----------------
left, center, right = st.columns([1,2,1])
frame_placeholder = center.empty()
chart_placeholder = left.empty()
status_placeholder = right.empty()

# ---------------- DATA ----------------
live_data = {
    "Time": [],
    "Alerts": []
}

# ---------------- PROCESS FRAME ----------------
def process_frame(frame):

    people_count = 0
    weapon_names = []

    # =========================
    # 👥 CROWD DETECTION
    # =========================
    c_res = crowd_model.predict(frame, conf=confidence, verbose=False)

    if c_res and c_res[0].boxes is not None:
        for box in c_res[0].boxes:
            if int(box.cls[0]) == 0:  # person
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # =========================
    # 🔫 WEAPON DETECTION (FIXED)
    # =========================
    w_res = weapon_model.predict(frame, conf=confidence, verbose=False)

    if w_res and w_res[0].boxes is not None:
        for box in w_res[0].boxes:

            cls_id = int(box.cls[0])
            label = w_res[0].names[cls_id].lower().strip()
            conf = float(box.conf[0])

            # ✅ FILTER ONLY KNIFE + PISTOL
            if label not in WEAPON_CLASSES:
                continue

            weapon_names.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # =========================
    # 📊 RIGHT PANEL UI
    # =========================
    if weapon_names:
        status_placeholder.markdown(
            f'<div class="card">👥 People: {people_count}<br>🔫 Weapon: {", ".join(set(weapon_names))}</div>',
            unsafe_allow_html=True
        )
    else:
        status_placeholder.markdown(
            f'<div class="card">👥 People: {people_count}<br>🟢 No Weapon</div>',
            unsafe_allow_html=True
        )

    # =========================
    # 📈 GRAPH
    # =========================
    timestamp = datetime.now().strftime("%H:%M:%S")
    live_data["Time"].append(timestamp)
    live_data["Alerts"].append(1 if weapon_names else 0)

    return frame

# ---------------- GRAPH ----------------
def update_chart():
    if len(live_data["Time"]) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=live_data["Time"],
            y=live_data["Alerts"],
            mode='lines+markers',
            name="Weapon Alerts"
        ))
        fig.update_layout(
            title="Weapon Detection Over Time",
            plot_bgcolor='#111',
            paper_bgcolor='#111',
            font=dict(color='white')
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# ---------------- VIDEO ----------------
if mode == "Upload Video":

    uploaded_file = st.file_uploader("📂 Upload Video", type=["mp4","avi"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = process_frame(frame)
            frame = cv2.resize(frame, (480, 300))

            frame_placeholder.image(frame, channels="BGR")
            update_chart()

            time.sleep(0.03)

        cap.release()

# ---------------- WEBCAM ----------------
if mode == "Webcam":

    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)
            frame = cv2.resize(frame, (480, 300))

            frame_placeholder.image(frame, channels="BGR")
            update_chart()

            time.sleep(0.03)

        cap.release()
        cv2.destroyAllWindows()