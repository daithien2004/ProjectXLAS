import streamlit as st
import cv2
import numpy as np
import os
from config import (
    CHAPTER3_IMAGE_MAP, CHAPTER3_FUNCTIONS, CHAPTER4_FUNCTIONS,
    CHAPTER9_IMAGE_MAP, CHAPTER9_FUNCTIONS, CHAPTER9_IMAGE_DESCRIPTIONS,
    FACE_OPTIONS, IMAGE_FOLDER_CH3, IMAGE_FOLDER_CH9
)
import chapter3
import chapter4
import chapter9

def setup_sidebar():
    """
    Thiết lập sidebar với thông tin sinh viên và menu chọn tác vụ.
    Returns: (main_task, sub_task) - Tác vụ chính và tác vụ chi tiết.
    """
    st.sidebar.title("👨‍🎓 Thông tin sinh viên")
    st.sidebar.markdown("**Họ tên:** Quảng Đại Thiện")
    st.sidebar.markdown("**MSSV:** 22110426")
    st.sidebar.markdown("---")

    # Khởi tạo session state
    if 'main_task' not in st.session_state:
        st.session_state.main_task = "Nhận dạng khuôn mặt"
    if 'sub_task' not in st.session_state:
        st.session_state.sub_task = "Nhận dạng từ Webcam"

    # Radio button để chọn tác vụ chính
    st.sidebar.header("🔍 Chọn loại tác vụ")
    main_task = st.sidebar.radio(
        "Loại tác vụ:",
        ["Nhận dạng khuôn mặt", "Nhận dạng trái cây", "Xử lý ảnh Chapter 3", "Xử lý ảnh Chapter 4", "Xử lý ảnh Chapter 9"],
        key="main_task_selector"
    )

    # Đồng bộ main_task ngay lập tức
    if main_task != st.session_state.main_task:
        st.session_state.main_task = main_task
        # Đặt sub_task về giá trị mặc định cho main_task mới
        default_sub_tasks = {
            "Nhận dạng khuôn mặt": "Nhận dạng từ Webcam",
            "Nhận dạng trái cây": "Tải lên ảnh",
            "Xử lý ảnh Chapter 3": "Chọn tác vụ",
            "Xử lý ảnh Chapter 4": "Chọn tác vụ",
            "Xử lý ảnh Chapter 9": "Chọn tác vụ"
        }
        st.session_state.sub_task = default_sub_tasks[main_task]

    # Combobox cho tác vụ chi tiết
    sub_task = st.session_state.sub_task
    with st.sidebar.container():
        if main_task == "Nhận dạng khuôn mặt":
            st.sidebar.header("🔍 Tác vụ nhận dạng")
            sub_task = st.sidebar.selectbox(
                "Chọn tác vụ nhận dạng:",
                FACE_OPTIONS,
                index=FACE_OPTIONS.index(st.session_state.sub_task) if st.session_state.sub_task in FACE_OPTIONS else 0,
                key="face_sub_task_selector"
            )
        elif main_task == "Nhận dạng trái cây":
            st.sidebar.header("🍎 Tác vụ nhận dạng trái cây")
            sub_task = st.sidebar.selectbox(
                "Chọn tác vụ:",
                ["Tải lên ảnh"],
                index=0,
                key="fruit_sub_task_selector"
            )
        elif main_task == "Xử lý ảnh Chapter 3":
            st.sidebar.header("🖼️ Tác vụ Chapter 3")
            sub_task = st.sidebar.selectbox(
                "Chọn tác vụ xử lý ảnh:",
                list(CHAPTER3_FUNCTIONS.keys()),
                index=list(CHAPTER3_FUNCTIONS.keys()).index(st.session_state.sub_task) if st.session_state.sub_task in CHAPTER3_FUNCTIONS else 0,
                format_func=lambda x: CHAPTER3_FUNCTIONS[x],
                key="chapter3_sub_task_selector"
            )
        elif main_task == "Xử lý ảnh Chapter 4":
            st.sidebar.header("🖼️ Tác vụ Chapter 4")
            sub_task = st.sidebar.selectbox(
                "Chọn tác vụ xử lý ảnh:",
                list(CHAPTER4_FUNCTIONS.keys()),
                index=list(CHAPTER4_FUNCTIONS.keys()).index(st.session_state.sub_task) if st.session_state.sub_task in CHAPTER4_FUNCTIONS else 0,
                format_func=lambda x: CHAPTER4_FUNCTIONS[x],
                key="chapter4_sub_task_selector"
            )
        elif main_task == "Xử lý ảnh Chapter 9":
            st.sidebar.header("🖼️ Tác vụ Chapter 9")
            sub_task = st.sidebar.selectbox(
                "Chọn tác vụ xử lý ảnh:",
                list(CHAPTER9_FUNCTIONS.keys()),
                index=list(CHAPTER9_FUNCTIONS.keys()).index(st.session_state.sub_task) if st.session_state.sub_task in CHAPTER9_FUNCTIONS else 0,
                format_func=lambda x: CHAPTER9_FUNCTIONS[x],
                key="chapter9_sub_task_selector"
            )

    # Cập nhật sub_task
    st.session_state.sub_task = sub_task

    return main_task, sub_task

def display_image_processing_ui(chapter, task, image_map, functions, image_folder, session_key, image_descriptions=None):
    """
    Hiển thị giao diện xử lý ảnh cho một chapter cụ thể.
    """
    st.header(f"🖼️ Xử lý ảnh số - {chapter}")

    if image_descriptions and task in image_descriptions:
        st.write(f"**Loại ảnh khuyến nghị**: {image_descriptions[task]}")

    use_default_image = st.checkbox("Sử dụng ảnh mặc định", value=True, key=f"{session_key}_default_image")

    img = None
    if use_default_image:
        if task in image_map:
            image_path = os.path.join(image_folder, image_map[task])
            try:
                is_grayscale = task not in ["NegativeColor", "HistEqualColor"] if chapter == "Chapter 3" else True
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
                if img is None:
                    st.error(f"Không tìm thấy file ảnh: {image_path}")
                    return
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh mặc định: {str(e)}")
                st.warning(f"Vui lòng kiểm tra file {image_path} trong thư mục {image_folder}.")
                return
    else:
        uploaded_image = st.file_uploader(
            "📤 Tải lên ảnh để xử lý",
            type=["jpg", "jpeg", "png", "tif"],
            help="Hỗ trợ ảnh JPG, PNG, TIFF",
            key=f"{session_key}_uploader"
        )
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            is_grayscale = task not in ["NegativeColor", "HistEqualColor"] if chapter == "Chapter 3" else True
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)

    if img is not None:
        try:
            module = {"Chapter 3": chapter3, "Chapter 4": chapter4, "Chapter 9": chapter9}
            processed_img = getattr(module[chapter], task)(img)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc", use_column_width=True, channels="BGR" if img.ndim == 3 else "GRAY")
            with col2:
                st.image(processed_img, caption=f"Kết quả ({functions[task]})", use_column_width=True, channels="BGR" if processed_img.ndim == 3 else "GRAY")

            if st.button(f"💾 Tải xuống ảnh đã xử lý ({chapter})"):
                _, encoded_img = cv2.imencode('.png', processed_img)
                st.download_button(
                    label="Nhấn để tải xuống",
                    data=encoded_img.tobytes(),
                    file_name=f"processed_{task}.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            st.warning(f"Kiểm tra định dạng ảnh đầu vào (xám/màu) phù hợp với chức năng đã chọn.")