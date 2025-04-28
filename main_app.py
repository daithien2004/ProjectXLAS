import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import joblib
import tempfile

# ============ CẤU HÌNH WEBRTC =============
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": ["turn:global.relay.metered.ca:80"],
            "username": "e19de69abf4542dbf3820d61",
            "credential": "9/VaW0S2PKXSyoF9",
        },
        {
            "urls": ["turn:global.relay.metered.ca:443"],
            "username": "e19de69abf4542dbf3820d61",
            "credential": "9/VaW0S2PKXSyoF9",
        },
        {
            "urls": ["turns:global.relay.metered.ca:443?transport=tcp"],
            "username": "e19de69abf4542dbf3820d61",
            "credential": "9/VaW0S2PKXSyoF9",
        }
    ]
})

# ============ STREAMLIT SETUP =============
st.set_page_config(page_title="Nhận dạng khuôn mặt", layout="wide")
st.title("📸 ĐỒ ÁN XỬ LÝ ẢNH - Nhận dạng khuôn mặt")
st.markdown("---")

st.sidebar.title("👨‍🎓 Thông tin sinh viên")
st.sidebar.markdown("**Họ tên:** Quảng Đại Thiện")
st.sidebar.markdown("**MSSV:** 22110426")
st.sidebar.markdown("---")
option = st.sidebar.selectbox("🎯 Chọn chế độ:", 
                               ["Nhận dạng webcam", "Nhận dạng video mẫu", "Upload video nhận dạng"])

# ============ LOAD MODEL =============
svc = joblib.load('model/svc.pkl')
mydict = ['AnhKhoa', 'DaiThien', 'Loc', 'LyHung', 'TamTue']
color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  

detector = cv2.FaceDetectorYN.create(
    "model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(
    "model/face_recognition_sface_2021dec.onnx", "")

# ============ HÀM XỬ LÝ FRAME =============
def process_frame(frame):
    img = frame.copy()
    h, w, _ = img.shape
    detector.setInputSize((w, h))

    faces = detector.detect(img)
    if faces[1] is not None:
        for face in faces[1]:
            aligned = recognizer.alignCrop(img, face)
            feat = recognizer.feature(aligned)
            pred = svc.predict(feat)
            label = mydict[pred[0]]
            coords = face[:4].astype(np.int32)
            cv2.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color[pred[0]], 2)
            cv2.putText(img, label, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[pred[0]], 2)
    return img

# ============ XỬ LÝ VIDEO STREAM =============
class RealTimeProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============ MAIN ==================
if option == "Nhận dạng webcam":
    st.subheader("📷 Nhận dạng khuôn mặt từ Webcam")
    webrtc_streamer(
        key="realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=RealTimeProcessor,
        async_processing=True,
    )

elif option == "Nhận dạng video mẫu":
    st.subheader("🎞️ Nhận dạng từ video mẫu")
    video_path = "nhandien.mp4"
    video_file = open(video_path, 'rb')
    st.video(video_file.read())

    cap = cv2.VideoCapture(video_path)
    FRAME_WINDOW = st.image([])

    run = st.button("▶️ Bắt đầu nhận dạng")
    stop = st.button("⏹️ Dừng lại")

    if run:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            FRAME_WINDOW.image(frame, channels="BGR")
            if stop:
                break
        cap.release()

elif option == "Upload video nhận dạng":
    st.subheader("📤 Upload video từ máy tính")

    uploaded_file = st.file_uploader("Tải lên video (mp4, avi, mov...)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        FRAME_WINDOW = st.image([])

        run_upload = st.button("▶️ Bắt đầu nhận dạng")
        stop_upload = st.button("⏹️ Dừng lại")

        if run_upload:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = process_frame(frame)
                FRAME_WINDOW.image(frame, channels="BGR")
                if stop_upload:
                    break
            cap.release()
