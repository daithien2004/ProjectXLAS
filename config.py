from streamlit_webrtc import RTCConfiguration
import cv2

# Cấu hình WebRTC
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

# Ánh xạ tác vụ và file ảnh cho Chapter 3
CHAPTER3_IMAGE_MAP = {
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

CHAPTER3_FUNCTIONS = {
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

# Ánh xạ tác vụ và file ảnh cho Chapter 4
CHAPTER4_FUNCTIONS = {
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

# Ánh xạ tác vụ và file ảnh cho Chapter 9
CHAPTER9_IMAGE_MAP = {
    "Erosion": "9.1.tif",
    "Dilation": "9.2.tif",
    "Boundary": "9.3.tif",
    "Counter": "9.4.tif"
}

CHAPTER9_FUNCTIONS = {
    "Chọn tác vụ": "Chọn tác vụ",
    "Erosion": "Xói mòn",
    "Dilation": "Giãn nở",
    "Boundary": "Trích biên",
    "Counter": "Vẽ đường viền (Contours)"
}

CHAPTER9_IMAGE_DESCRIPTIONS = {
    "Erosion": "Ảnh nhị phân với vùng sáng rõ (ví dụ: văn bản, hình khối).",
    "Dilation": "Ảnh nhị phân tương tự, để thấy hiệu quả giãn nở.",
    "Boundary": "Ảnh nhị phân với vùng sáng có biên rõ (ví dụ: hình tròn, hình vuông).",
    "Counter": "Ảnh nhị phân với nhiều đối tượng (để vẽ nhiều đường viền)."
}

# Các hằng số cho nhận dạng trái cây
FRUIT_CLASSES = ['Buoi', 'DuaHau', 'SauRieng', 'Tao', 'ThanhLong']  # Đúng với mô hình
FRUIT_INPUT_WIDTH = 640
FRUIT_INPUT_HEIGHT = 640
FRUIT_SCORE_THRESHOLD = 0.4  # Giảm để tăng phát hiện
FRUIT_NMS_THRESHOLD = 0.45
FRUIT_CONFIDENCE_THRESHOLD = 0.3  # Giảm để tăng phát hiện
FRUIT_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FRUIT_FONT_SCALE = 0.7
FRUIT_THICKNESS = 1
FRUIT_COLORS = {
    'BLACK': (0, 0, 0),
    'BLUE': (255, 178, 50),
    'YELLOW': (0, 255, 255),
    'RED': (0, 0, 255)
}

# Các hằng số khác
IMAGE_FOLDER_CH3 = "PictureForChapter3"
IMAGE_FOLDER_CH9 = "PictureForChapter9"
FACE_OPTIONS = ["Chọn tác vụ", "Nhận dạng từ Webcam", "Nhận dạng từ video mẫu", "Nhận dạng từ video upload"]