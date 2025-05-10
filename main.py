import streamlit as st
from config import (
    RTC_CONFIGURATION, CHAPTER3_IMAGE_MAP, CHAPTER3_FUNCTIONS, IMAGE_FOLDER_CH3,
    CHAPTER4_IMAGE_MAP, CHAPTER4_FUNCTIONS, CHAPTER4_IMAGE_DESCRIPTIONS, IMAGE_FOLDER_CH4,
    CHAPTER9_IMAGE_MAP, CHAPTER9_FUNCTIONS, CHAPTER9_IMAGE_DESCRIPTIONS, IMAGE_FOLDER_CH9
)
from ui import setup_sidebar, display_image_processing_ui
from face_recognition import FaceRecognizer
from emotion_recognition import EmotionRecognizer
from fruit_recognition import load_fruit_model, process_fruit_recognition
import os

@st.cache_resource
def load_face_recognizer():
    return FaceRecognizer()

@st.cache_resource
def load_emotion_recognizer():
    return EmotionRecognizer()

@st.cache_resource
def load_fruit_model_cached():
    return load_fruit_model()

def main():
    st.set_page_config(page_title="Xử lý ảnh số", layout="wide")
    st.title("📸 ĐỒ ÁN XỬ LÝ ẢNH")
    with st.spinner("Đang khởi tạo mô hình..."):
        face_recognizer = load_face_recognizer()
        emotion_recognizer = load_emotion_recognizer()
        fruit_model = load_fruit_model_cached()
    st.markdown("---")

    main_task, sub_task = setup_sidebar()

    if main_task == "Nhận dạng khuôn mặt" and sub_task != "Chọn tác vụ":
        st.header("🔍 Nhận dạng khuôn mặt")
        if sub_task == "Nhận dạng từ Webcam":
            st.subheader("📷 Nhận dạng từ Webcam")
            face_recognizer.process_webcam(RTC_CONFIGURATION)
        elif sub_task == "Nhận dạng từ video mẫu":
            st.subheader("🎞️ Nhận dạng từ video mẫu")
            if os.path.exists("nhandien.mp4"):
                face_recognizer.process_video("nhandien.mp4")
            else:
                st.error("File video mẫu 'nhandien.mp4' không tồn tại!")
        elif sub_task == "Nhận dạng từ video upload":
            st.subheader("📤 Nhận dạng từ video upload")
            uploaded_file = st.file_uploader("Tải lên video", type=["mp4", "avi", "mov"])
            if uploaded_file:
                face_recognizer.process_video(uploaded_file, is_uploaded=True)

    elif main_task == "Nhận diện cảm xúc" and sub_task != "Chọn tác vụ":
        st.header("😊 Nhận diện cảm xúc")
        if sub_task == "Nhận diện từ Webcam":
            st.subheader("📷 Nhận diện từ Webcam")
            emotion_recognizer.process_webcam(RTC_CONFIGURATION)
        elif sub_task == "Nhận diện từ video upload":
            st.subheader("📤 Nhận diện từ video upload")
            uploaded_file = st.file_uploader("Tải lên video", type=["mp4", "avi", "mov"])
            if uploaded_file:
                emotion_recognizer.process_video(uploaded_file, is_uploaded=True)
        elif sub_task == "Nhận diện từ ảnh upload":
            st.subheader("🖼️ Nhận diện từ ảnh upload")
            emotion_recognizer.process_image()

    elif main_task == "Nhận dạng trái cây" and sub_task == "Tải lên ảnh":
        if fruit_model is not None:
            process_fruit_recognition(fruit_model)
        else:
            st.error("Không thể tải mô hình nhận dạng trái cây. Kiểm tra file trai_cay.onnx.")

    elif main_task == "Xử lý ảnh Chapter 3" and sub_task != "Chọn tác vụ":
        display_image_processing_ui(
            chapter="Chapter 3",
            task=sub_task,
            image_map=CHAPTER3_IMAGE_MAP,
            functions=CHAPTER3_FUNCTIONS,
            image_folder=IMAGE_FOLDER_CH3,
            session_key="chapter3"
        )

    elif main_task == "Xử lý ảnh Chapter 4" and sub_task != "Chọn tác vụ":
        display_image_processing_ui(
            chapter="Chapter 4",
            task=sub_task,
            image_map=CHAPTER4_IMAGE_MAP,
            functions=CHAPTER4_FUNCTIONS,
            image_folder=IMAGE_FOLDER_CH4,
            session_key="chapter4",
            image_descriptions=CHAPTER4_IMAGE_DESCRIPTIONS
        )

    elif main_task == "Xử lý ảnh Chapter 9" and sub_task != "Chọn tác vụ":
        display_image_processing_ui(
            chapter="Chapter 9",
            task=sub_task,
            image_map=CHAPTER9_IMAGE_MAP,
            functions=CHAPTER9_FUNCTIONS,
            image_folder=IMAGE_FOLDER_CH9,
            session_key="chapter9",
            image_descriptions=CHAPTER9_IMAGE_DESCRIPTIONS
        )

    else:
        st.header("🔍 Nhận dạng khuôn mặt")
        st.subheader("📷 Nhận dạng từ Webcam")
        face_recognizer.process_webcam(RTC_CONFIGURATION)

if __name__ == "__main__":
    main()