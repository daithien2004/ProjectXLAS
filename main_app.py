import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import joblib
import tempfile
import chapter3
import chapter4
import chapter9
import os

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
st.set_page_config(page_title="Xử lý ảnh số", layout="wide")
st.title("📸 ĐỒ ÁN XỬ LÝ ẢNH")
st.markdown("---")

st.sidebar.title("👨‍🎓 Thông tin sinh viên")
st.sidebar.markdown("**Họ tên:** Quảng Đại Thiện")
st.sidebar.markdown("**MSSV:** 22110426")
st.sidebar.markdown("---")

# Khởi tạo session state để quản lý trạng thái combobox
if 'face_task' not in st.session_state:
    st.session_state.face_task = "Nhận dạng từ Webcam"  # Mặc định là Webcam
if 'chapter3_task' not in st.session_state:
    st.session_state.chapter3_task = "Chọn tác vụ"
if 'chapter4_task' not in st.session_state:
    st.session_state.chapter4_task = "Chọn tác vụ"
if 'chapter9_task' not in st.session_state:
    st.session_state.chapter9_task = "Chọn tác vụ"

# ============ COMBOBOX NHẬN DẠNG KHUÔN MẶT =============
st.sidebar.header("🔍 Nhận dạng khuôn mặt")
face_options = ["Chọn tác vụ", "Nhận dạng từ Webcam", "Nhận dạng từ video mẫu", "Nhận dạng từ video upload"]
face_task = st.sidebar.selectbox(
    "Chọn tác vụ nhận dạng:",
    face_options,
    index=face_options.index(st.session_state.face_task),
    key="face_task_selector"
)

# Cập nhật trạng thái khi chọn tác vụ Nhận dạng khuôn mặt
if face_task != st.session_state.face_task:
    st.session_state.face_task = face_task
    if face_task != "Chọn tác vụ":
        st.session_state.chapter3_task = "Chọn tác vụ"  # Đặt lại Chapter 3
        st.session_state.chapter4_task = "Chọn tác vụ"  # Đặt lại Chapter 4
        st.session_state.chapter9_task = "Chọn tác vụ"  # Đặt lại Chapter 9

# ============ COMBOBOX XỬ LÝ ẢNH CHAPTER 3 =============
st.sidebar.header("🖼️ Xử lý ảnh Chapter 3")
chapter3_functions = {
    "Chọn tác vụ": "Chọn tác vụ",
    "Negative": "Ảnh âm bản (xám)",
    "NegativeColor": "Ảnh âm bản (màu)",
    "Logarit": "Biến đổi Logarit",
    "Power": "Biến đổi lũy thừa",
    "PiecewisetLine": "Biến đổi tuyến tính từng khúc",
    "Histogram": "Hiển thị histogram",
    "Hist_equal": "Cân bằng histogram (xám)",
    "HistEqualColor": "Cân bằng histogram (màu)",
    "LocalHist": "Cân bằng histogram cục bộ",
    "HistStat": "Lọc theo thống kê histogram",
    "Sharpening": "Làm nét ảnh (Laplacian)",
    "SharpeningMask": "Làm nét ảnh (SharpeningMask)",
    "Gradient": "Biên ảnh (Gradient)"
}
chapter3_task = st.sidebar.selectbox(
    "Chọn tác vụ xử lý ảnh (Chapter 3):",
    list(chapter3_functions.keys()),
    index=list(chapter3_functions.keys()).index(st.session_state.chapter3_task),
    format_func=lambda x: chapter3_functions[x],
    key="chapter3_task_selector"
)

# Cập nhật trạng thái khi chọn tác vụ Chapter 3
if chapter3_task != st.session_state.chapter3_task:
    st.session_state.chapter3_task = chapter3_task
    if chapter3_task != "Chọn tác vụ":
        st.session_state.face_task = "Chọn tác vụ"  # Đặt lại Nhận dạng khuôn mặt
        st.session_state.chapter4_task = "Chọn tác vụ"  # Đặt lại Chapter 4
        st.session_state.chapter9_task = "Chọn tác vụ"  # Đặt lại Chapter 9
        st.rerun()

# Ánh xạ tác vụ Chapter 3 tới file ảnh
chapter3_image_map = {
    "Negative": "3.1.tif",
    "NegativeColor": "3.12.tif",
    "Logarit": "3.2.tif",
    "Power": "3.3.tif",
    "PiecewisetLine": "3.5.jpg",
    "Histogram": "3.6.tif",
    "Hist_equal": "3.7.tif",
    "HistEqualColor": "3.8.tif",
    "LocalHist": "3.9.tif",
    "HistStat": "3.10.tif",
    "Sharpening": "3.11.tif",
    "SharpeningMask": "3.12.tif",
    "Gradient": "3.13.tif" 
}
image_folder = "PictureForChapter3"  # Thư mục chứa ảnh

# ============ COMBOBOX XỬ LÝ ẢNH CHAPTER 4 =============
st.sidebar.header("🖼️ Xử lý ảnh Chapter 4")
chapter4_functions = {
    "Chọn tác vụ": "Chọn tác vụ",
    "Spectrum": "Phổ tần số",
    "RemoveMoire": "Loại bỏ nhiễu Moire",
    "RemoveInterference": "Loại bỏ nhiễu giao thoa",
    "RemoveMoireSimple": "Loại bỏ nhiễu Moire (đơn giản)",
    "RemoveInferenceFilter": "Loại bỏ nhiễu giao thoa (bộ lọc)",
    "DrawInferenceFilter": "Vẽ bộ lọc giao thoa",
    "CreateMotion": "Tạo hiệu ứng chuyển động",
    "CreateDemotion": "Khử hiệu ứng chuyển động"
}
chapter4_task = st.sidebar.selectbox(
    "Chọn tác vụ xử lý ảnh (Chapter 4):",
    list(chapter4_functions.keys()),
    index=list(chapter4_functions.keys()).index(st.session_state.chapter4_task),
    format_func=lambda x: chapter4_functions[x],
    key="chapter4_task_selector"
)

# Cập nhật trạng thái khi chọn tác vụ Chapter 4
if chapter4_task != st.session_state.chapter4_task:
    st.session_state.chapter4_task = chapter4_task
    if chapter4_task != "Chọn tác vụ":
        st.session_state.face_task = "Chọn tác vụ"  # Đặt lại Nhận dạng khuôn mặt
        st.session_state.chapter3_task = "Chọn tác vụ"  # Đặt lại Chapter 3
        st.session_state.chapter9_task = "Chọn tác vụ"  # Đặt lại Chapter 9

# ============ COMBOBOX XỬ LÝ ẢNH CHAPTER 9 =============
st.sidebar.header("🖼️ Xử lý ảnh Chapter 9")
chapter9_functions = {
    "Chọn tác vụ": "Chọn tác vụ",
    "Erosion": "Xói mòn",
    "Dilation": "Giãn nở",
    "Boundary": "Trích biên",
    "Counter": "Vẽ đường viền (Contours)"
}
chapter9_task = st.sidebar.selectbox(
    "Chọn tác vụ xử lý ảnh (Chapter 9):",
    list(chapter9_functions.keys()),
    index=list(chapter9_functions.keys()).index(st.session_state.chapter9_task),
    format_func=lambda x: chapter9_functions[x],
    key="chapter9_task_selector"
)

# Cập nhật trạng thái khi chọn tác vụ Chapter 9
if chapter9_task != st.session_state.chapter9_task:
    st.session_state.chapter9_task = chapter9_task
    if chapter9_task != "Chọn tác vụ":
        st.session_state.face_task = "Chọn tác vụ"  # Đặt lại Nhận dạng khuôn mặt
        st.session_state.chapter3_task = "Chọn tác vụ"  # Đặt lại Chapter 3
        st.session_state.chapter4_task = "Chọn tác vụ"  # Đặt lại Chapter 4

# ============ LOGIC XỬ LÝ =============
# Hiển thị giao diện dựa trên tác vụ được chọn
if st.session_state.chapter3_task != "Chọn tác vụ":
    # Hiển thị giao diện Xử lý ảnh Chapter 3
    st.header("🖼️ Xử lý ảnh số - Chapter 3")

    # Checkbox để chọn giữa ảnh mặc định và upload thủ công
    use_default_image = st.checkbox("Sử dụng ảnh mặc định", value=True, key="chapter3_default_image")

    if use_default_image:
        # Tải ảnh mặc định từ thư mục
        image_path = os.path.join(image_folder, chapter3_image_map[st.session_state.chapter3_task])
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR if st.session_state.chapter3_task in ["NegativeColor", "HistEqualColor"] else cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh mặc định: {str(e)}")
            st.warning(f"Vui lòng kiểm tra file {image_path} trong thư mục {image_folder}.")
            img = None
    else:
        # Tùy chọn upload ảnh thủ công
        uploaded_image = st.file_uploader(
            "📤 Tải lên ảnh để xử lý",
            type=["jpg", "jpeg", "png", "tif"],
            help="Hỗ trợ ảnh JPG, PNG, TIFF",
            key="chapter3_uploader"
        )
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR if st.session_state.chapter3_task in ["NegativeColor", "HistEqualColor"] else cv2.IMREAD_GRAYSCALE)
        else:
            img = None

    if img is not None:
        # Xử lý ảnh
        try:
            processed_img = getattr(chapter3, st.session_state.chapter3_task)(img)

            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc", use_column_width=True)
            with col2:
                st.image(processed_img,
                         caption=f"Kết quả ({chapter3_functions[st.session_state.chapter3_task]})",
                         use_column_width=True)

            # Tùy chọn tải xuống ảnh đã xử lý
            if st.button("💾 Tải xuống ảnh đã xử lý (Chapter 3)"):
                _, encoded_img = cv2.imencode('.png', processed_img)
                st.download_button(
                    label="Nhấn để tải xuống",
                    data=encoded_img.tobytes(),
                    file_name=f"processed_{st.session_state.chapter3_task}.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            st.warning(f"Kiểm tra lại định dạng ảnh đầu vào (xám/màu) phù hợp với chức năng đã chọn.")

elif st.session_state.chapter4_task != "Chọn tác vụ":
    # Hiển thị giao diện Xử lý ảnh Chapter 4
    st.header("🖼️ Xử lý ảnh số - Chapter 4")

    uploaded_image = st.file_uploader(
        "📤 Tải lên ảnh để xử lý",
        type=["jpg", "jpeg", "png", "tif"],
        help="Hỗ trợ ảnh JPG, PNG, TIFF (ảnh xám được khuyến nghị)",
        key="chapter4_uploader"
    )

    if uploaded_image is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Chapter 4 yêu cầu ảnh xám

        # Xử lý ảnh
        try:
            processed_img = getattr(chapter4, st.session_state.chapter4_task)(img)

            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc", use_column_width=True)
            with col2:
                st.image(processed_img,
                         caption=f"Kết quả ({chapter4_functions[st.session_state.chapter4_task]})",
                         use_column_width=True)

            # Tùy chọn tải xuống ảnh đã xử lý
            if st.button("💾 Tải xuống ảnh đã xử lý (Chapter 4)"):
                _, encoded_img = cv2.imencode('.png', processed_img)
                st.download_button(
                    label="Nhấn để tải xuống",
                    data=encoded_img.tobytes(),
                    file_name=f"processed_{st.session_state.chapter4_task}.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            st.warning("Kiểm tra lại định dạng ảnh đầu vào (ảnh xám được yêu cầu).")

elif st.session_state.chapter9_task != "Chọn tác vụ":
    # Hiển thị giao diện Xử lý ảnh Chapter 9
    st.header("🖼️ Xử lý ảnh số - Chapter 9")

    uploaded_image = st.file_uploader(
        "📤 Tải lên ảnh để xử lý",
        type=["jpg", "jpeg", "png", "tif"],
        help="Hỗ trợ ảnh JPG, PNG, TIFF (ảnh nhị phân hoặc xám được khuyến nghị)",
        key="chapter9_uploader"
    )

    if uploaded_image is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Chapter 9 yêu cầu ảnh xám

        # Xử lý ảnh
        try:
            processed_img = getattr(chapter9, st.session_state.chapter9_task)(img)

            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc", use_column_width=True)
            with col2:
                st.image(processed_img,
                         caption=f"Kết quả ({chapter9_functions[st.session_state.chapter9_task]})",
                         use_column_width=True)

            # Tùy chọn tải xuống ảnh đã xử lý
            if st.button("💾 Tải xuống ảnh đã xử lý (Chapter 9)"):
                _, encoded_img = cv2.imencode('.png', processed_img)
                st.download_button(
                    label="Nhấn để tải xuống",
                    data=encoded_img.tobytes(),
                    file_name=f"processed_{st.session_state.chapter9_task}.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            st.warning("Kiểm tra lại định dạng ảnh đầu vào (ảnh nhị phân hoặc xám được yêu cầu).")

elif st.session_state.face_task != "Chọn tác vụ":
    # Hiển thị giao diện Nhận dạng khuôn mặt
    st.header("🔍 Nhận dạng khuôn mặt")

    # Load model nhận dạng khuôn mặt
    svc = joblib.load('model/svc.pkl')
    mydict = ['AnhKhoa', 'DaiThien', 'Loc', 'LyHung', 'TamTue']
    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    detector = cv2.FaceDetectorYN.create(
        "model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(
        "model/face_recognition_sface_2021dec.onnx", "")

    # Hàm xử lý frame
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

    # Xử lý video stream
    class RealTimeProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process_frame(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Xử lý các tác vụ nhận dạng
    if st.session_state.face_task == "Nhận dạng từ Webcam":
        st.subheader("📷 Nhận dạng từ Webcam")
        webrtc_streamer(
            key="realtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=RealTimeProcessor,
            async_processing=True,
        )

    elif st.session_state.face_task == "Nhận dạng từ video mẫu":
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

    elif st.session_state.face_task == "Nhận dạng từ video upload":
        st.subheader("📤 Nhận dạng từ video upload")
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

else:
    # Hiển thị giao diện mặc định (Nhận dạng từ Webcam)
    st.header("🔍 Nhận dạng khuôn mặt")
    st.subheader("📷 Nhận dạng từ Webcam")

    # Load model nhận dạng khuôn mặt
    svc = joblib.load('model/svc.pkl')
    mydict = ['AnhKhoa', 'DaiThien', 'Loc', 'LyHung', 'TamTue']
    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    detector = cv2.FaceDetectorYN.create(
        "model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(
        "model/face_recognition_sface_2021dec.onnx", "")

    # Hàm xử lý frame
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

    # Xử lý video stream
    class RealTimeProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process_frame(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=RealTimeProcessor,
        async_processing=True,
    )