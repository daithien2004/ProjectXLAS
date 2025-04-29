import streamlit as st
import cv2
import numpy as np
import joblib
import av
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode

class FaceRecognizer:
    def __init__(self):
        """Khởi tạo mô hình nhận dạng khuôn mặt."""
        self.svc = joblib.load('model/svc.pkl')
        self.mydict = ['AnhKhoa', 'DaiThien', 'Loc', 'LyHung', 'TamTue']
        self.color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        self.detector = cv2.FaceDetectorYN.create(
            "model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(
            "model/face_recognition_sface_2021dec.onnx", "")

    def process_frame(self, frame):
        """
        Xử lý một frame video để nhận dạng khuôn mặt.
        Args:
            frame: Frame ảnh (numpy array, BGR).
        Returns:
            Frame đã được xử lý với các khuôn mặt được nhận dạng.
        """
        img = frame.copy()
        h, w, _ = img.shape
        self.detector.setInputSize((w, h))
        faces = self.detector.detect(img)
        if faces[1] is not None:
            for face in faces[1]:
                aligned = self.recognizer.alignCrop(img, face)
                feat = self.recognizer.feature(aligned)
                pred = self.svc.predict(feat)
                label = self.mydict[pred[0]]
                coords = face[:4].astype(np.int32)
                cv2.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), self.color[pred[0]], 2)
                cv2.putText(img, label, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color[pred[0]], 2)
        return img

    def process_webcam(self, rtc_configuration):
        """Xử lý video từ webcam."""
        class RealTimeProcessor:
            def __init__(self, recognizer):
                self.recognizer = recognizer

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = self.recognizer.process_frame(img)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="realtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: RealTimeProcessor(self),
            async_processing=True,
        )

    def process_video(self, video_path, is_uploaded=False):
        """
        Xử lý video từ file mẫu hoặc file upload.
        Args:
            video_path: Đường dẫn file video hoặc đối tượng file upload.
            is_uploaded (bool): True nếu là file upload.
        """
        if is_uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_path.read())
                video_path = tfile.name
        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        FRAME_WINDOW = st.image([])
        run = st.button("▶️ Bắt đầu nhận dạng")
        stop = st.button("⏹️ Dừng lại")

        if run:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame)
                FRAME_WINDOW.image(frame, channels="BGR")
                if stop:
                    break
            cap.release()