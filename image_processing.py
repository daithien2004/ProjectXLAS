# Giả định các hàm xử lý ảnh đã được định nghĩa trong chapter3.py, chapter4.py, chapter9.py
from chapter3 import (
    Negative, NegativeColor, Logarit, Power, PiecewisetLine, Histogram,
    Hist_equal, HistEqualColor, LocalHist, HistStat, Sharpening, SharpeningMask, Gradient
)
from chapter4 import (
    Spectrum, RemoveMoire, RemoveInterference, RemoveMoireSimple,
    RemoveInferenceFilter, DrawInferenceFilter, CreateMotion, CreateDemotion,CreateDemotionNoise
)
from chapter9 import Erosion, Dilation, Boundary, Counter

# Có thể thêm các hàm tiện ích nếu cần
def validate_image(img, is_grayscale=True):
    """
    Kiểm tra ảnh đầu vào.
    Args:
        img: Ảnh numpy array.
        is_grayscale (bool): Yêu cầu ảnh xám hay không.
    Returns:
        bool: True nếu ảnh hợp lệ, False nếu không.
    """
    if img is None:
        return False
    if is_grayscale and img.ndim != 2:
        return False
    if not is_grayscale and img.ndim != 3:
        return False
    return True