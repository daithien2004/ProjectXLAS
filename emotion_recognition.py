# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import LeakyReLU
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import time
import os

class EmotionRecognizer:
    def __init__(self):
        """Khởi tạo mô hình nhận diện cảm xúc."""
        self.class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emoji_dict = {
            'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😄',
            'neutral': '😐', 'sad': '😢', 'surprise': '😲'
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.face_cascade.empty():
            st.error("Không thể tải Haar Cascade classifier!")
        self.model = self.load_model()

    @st.cache_resource
    def load_model(_self):
        """Tải mô hình nhận diện cảm xúc."""
        model_path = "emotion_model_simple.h5"  # Giữ như bạn cung cấp, xác nhận file ở thư mục gốc
        if not os.path.exists(model_path):
            st.error(f"Không tìm thấy file mô hình: {model_path}")
            return None
        try:
            return tf.keras.models.load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình: {str(e)}")
            return None

    def preprocess_image(self, image):
        """Xử lý ảnh đầu vào cho mô hình."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            return normalized.reshape(1, 48, 48, 1)
        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {str(e)}")
            return None

    def display_prediction(self, prediction, placeholder):
        """Hiển thị danh sách 7 cảm xúc và highlight cảm xúc nổi bật"""
        placeholder.empty()
        
        if prediction is None:
            with placeholder:
                st.warning("Không phát hiện khuôn mặt!")
            return
        
        with placeholder.container():
            # Tìm cảm xúc có xác suất cao nhất
            top_index = np.argmax(prediction[0])
            top_emotion = self.class_labels[top_index]
            top_prob = prediction[0][top_index] * 100
            
            # Hiển thị danh sách 7 cảm xúc
            st.markdown("### 📊 Xác suất các cảm xúc:")
            
            # Tạo 2 cột để hiển thị đẹp hơn
            col1, col2 = st.columns(2)
            
            for i, label in enumerate(self.class_labels):
                prob = prediction[0][i] * 100
                emoji = self.emoji_dict[label]
                
                # Highlight cảm xúc có xác suất cao nhất
                if i == top_index:
                    display_text = f"""
                    <div style='background-color:#f0f8ff; padding:8px; border-radius:5px; 
                                border-left:4px solid #1E90FF; margin-bottom:5px;'>
                        {emoji} <strong>{label.capitalize()}</strong>: 
                        <span style='color:#1E90FF;'>{prob:.2f}% ★</span>
                    </div>
                    """
                else:
                    display_text = f"{emoji} {label.capitalize()}: `{prob:.2f}%`"
                
                # Phân bố đều vào 2 cột
                if i < 4:  # 3 cảm xúc đầu hiển thị ở cột 1
                    col1.markdown(display_text, unsafe_allow_html=True)
                else:       # 4 cảm xúc sau hiển thị ở cột 2
                    col2.markdown(display_text, unsafe_allow_html=True)
            
            # Hiển thị cảm xúc nổi bật bên dưới
            st.markdown("---")
            st.markdown(
                f"""
                <div style='text-align: center; background-color:#f0f8ff; padding:15px; 
                            border-radius:10px; margin-top:15px;'>
                    <h3>🎯 CẢM XÚC NỔI BẬT</h3>
                    <div style='font-size:24px;'>
                        {self.emoji_dict[top_emotion]} <strong>{top_emotion.upper()}</strong> 
                        - <span style='color:#1E90FF;'>{top_prob:.2f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    def process_frame(self, frame):
        """Xử lý một khung hình để nhận diện cảm xúc."""
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        prediction = None

        if len(faces) == 0:
            return img, None

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            input_face = self.preprocess_image(face)
            if input_face is not None:
                prediction = self.model.predict(input_face, verbose=0)
                label = self.class_labels[np.argmax(prediction)]
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(
                    img,
                    label.capitalize(),
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        return img, prediction

    def process_webcam(self, rtc_configuration):
        """Xử lý video từ webcam."""
        class EmotionProcessor(VideoProcessorBase):
            def __init__(self):
                self.recognizer = EmotionRecognizer()
                self.latest_prediction = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                processed_img, prediction = self.recognizer.process_frame(img)
                self.latest_prediction = prediction
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        # Hiển thị giao diện
        col1, col2 = st.columns([1, 3])
        with col1:
            result_placeholder = st.empty()
        with col2:
            # Khởi tạo WebRTC trong cột 2
            ctx = webrtc_streamer(
                key="emotion-webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                video_processor_factory=EmotionProcessor,
                async_processing=True,
                media_stream_constraints={"video": True, "audio": False}
            )

            # Vòng lặp cập nhật UI
            while ctx.state.playing:
                if ctx.video_processor and ctx.video_processor.latest_prediction is not None:
                    # Hiển thị kết quả
                    self.display_prediction(
                        ctx.video_processor.latest_prediction,
                        result_placeholder
                    )
                time.sleep(0.05)  # Giảm CPU usage

    def process_video(self, video_path, is_uploaded=False):
        """Xử lý video từ file mẫu hoặc upload."""
        if self.model is None:
            st.error("Không thể chạy video vì mô hình không được tải!")
            return

        if is_uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_path.read())
            tfile.close()
            video_path = tfile.name
        else:
            if not os.path.exists(video_path):
                st.error(f"File video '{video_path}' không tồn tại!")
                return

        st.video(video_path, autoplay=False)

        if "video_running" not in st.session_state:
            st.session_state.video_running = False

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("▶️ Bắt đầu nhận dạng"):
                st.session_state.video_running = True
            if st.button("⏹️ Dừng lại"):
                st.session_state.video_running = False
        with col2:
            frame_window = st.image([])

        prob_placeholder = col1.empty()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Không thể mở file video: {video_path}")
            if is_uploaded:
                time.sleep(0.5)
                try:
                    os.remove(video_path)
                except PermissionError:
                    st.warning("Không thể xóa file video tạm thời vì nó đang được sử dụng.")
            return

        last_update = 0
        while cap.isOpened() and st.session_state.video_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame, prediction = self.process_frame(frame)
            frame_window.image(frame, channels="BGR")
            if prediction is not None:
                current_time = time.time()
                if current_time - last_update >= 0.2:
                    self.display_prediction(prediction, prob_placeholder)
                    last_update = current_time
            time.sleep(0.05)

        cap.release()
        time.sleep(0.5)
        if is_uploaded:
            for _ in range(3):
                try:
                    os.remove(video_path)
                    break
                except PermissionError:
                    time.sleep(1)
                    st.warning("Đang thử xóa file video tạm thời...")
            else:
                st.warning("Không thể xóa file video tạm thời vì nó đang được sử dụng.")

    def process_image(self):
        """Xử lý ảnh upload."""
        if self.model is None:
            st.error("Không thể xử lý ảnh vì mô hình không được tải!")
            return

        uploaded = st.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(img, caption="Ảnh đã tải lên", channels="BGR")
            input_img = self.preprocess_image(img)
            if input_img is not None:
                prediction = self.model.predict(input_img, verbose=0)
                prob_placeholder = st.empty()
                self.display_prediction(prediction, prob_placeholder)
            else:
                st.warning("Không phát hiện khuôn mặt trong ảnh!")