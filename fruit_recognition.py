import streamlit as st
import cv2
import numpy as np
from PIL import Image
from config import (
    FRUIT_CLASSES, FRUIT_INPUT_WIDTH, FRUIT_INPUT_HEIGHT,
    FRUIT_SCORE_THRESHOLD, FRUIT_NMS_THRESHOLD, FRUIT_CONFIDENCE_THRESHOLD,
    FRUIT_FONT_FACE, FRUIT_FONT_SCALE, FRUIT_THICKNESS, FRUIT_COLORS
)

@st.cache_resource
def load_fruit_model():
    """Tải mô hình nhận dạng trái cây."""
    try:
        net = cv2.dnn.readNet("./Source/NhanDienTraiCay/trai_cay.onnx")
        return net
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình trai_cay.onnx: {str(e)}")
        return None

def draw_label(im, label, x, y):
    """Vẽ nhãn lên ảnh tại vị trí (x, y)."""
    text_size = cv2.getTextSize(label, FRUIT_FONT_FACE, FRUIT_FONT_SCALE, FRUIT_THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), FRUIT_COLORS['BLACK'], cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FRUIT_FONT_FACE, FRUIT_FONT_SCALE, FRUIT_COLORS['YELLOW'], FRUIT_THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    """Tiền xử lý ảnh trước khi đưa vào mô hình."""
    try:
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (FRUIT_INPUT_WIDTH, FRUIT_INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs
    except Exception as e:
        st.error(f"Lỗi trong quá trình tiền xử lý: {str(e)}")
        return None

def post_process(input_image, outputs):
    """Hậu xử lý kết quả từ mô hình để vẽ hộp và nhãn, trả về ảnh và danh sách phát hiện."""
    if outputs is None:
        st.warning("Không nhận được kết quả từ mô hình.")
        return input_image, []

    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / FRUIT_INPUT_WIDTH
    y_factor = image_height / FRUIT_INPUT_HEIGHT

    # Thu thập các phát hiện
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= FRUIT_CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > FRUIT_SCORE_THRESHOLD and class_id < len(FRUIT_CLASSES):
                confidences.append(float(confidence))
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    if not boxes:
        st.warning("Không phát hiện được trái cây nào trong ảnh.")
        return input_image, []

    # Áp dụng Non-Maximum Suppression
    try:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, FRUIT_CONFIDENCE_THRESHOLD, FRUIT_NMS_THRESHOLD)
        if isinstance(indices, tuple):
            indices = indices[0]  # OpenCV cũ trả về tuple
        indices = indices.flatten() if indices.ndim > 1 else indices

        if len(indices) == 0:
            st.warning("Không có đối tượng nào vượt qua NMS.")
            return input_image, []

        # Danh sách các phát hiện
        detections = []
        for i in indices:
            if i < len(class_ids) and i < len(confidences):
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                cv2.rectangle(input_image, (left, top), (left + width, top + height), FRUIT_COLORS['BLUE'], 3 * FRUIT_THICKNESS)
                label = "{}:{:.2f}".format(FRUIT_CLASSES[class_ids[i]], confidences[i])
                draw_label(input_image, label, left, top)
                detections.append({
                    "class": FRUIT_CLASSES[class_ids[i]],
                    "confidence": confidences[i],
                    "box": box
                })
            else:
                st.warning(f"Chỉ số {i} không hợp lệ trong indices.")

        return input_image, detections
    except Exception as e:
        st.error(f"Lỗi trong quá trình hậu xử lý: {str(e)}")
        return input_image, []

def process_fruit_recognition(net):
    """Xử lý nhận dạng trái cây từ ảnh tải lên."""
    if net is None:
        st.error("Không thể tải mô hình nhận dạng trái cây. Vui lòng kiểm tra file trai_cay.onnx.")
        return

    st.header("🍎 Nhận dạng trái cây")
    st.write("**Mô tả**: Nhận dạng 5 loại trái cây (Bưởi, Dưa Hấu, Sầu Riêng, Táo, Thanh Long) từ ảnh tải lên.")

    img_file_buffer = st.file_uploader("📤 Tải lên ảnh chứa trái cây", type=["bmp", "png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        # Hiển thị ảnh gốc
        try:
            image = Image.open(img_file_buffer)
            frame = np.array(image)
            if frame.ndim == 2:  # Ảnh xám
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            st.image(image, caption="Ảnh gốc", use_column_width=True)
        except Exception as e:
            st.error(f"Lỗi khi đọc ảnh: {str(e)}")
            return

        if st.button("🔍 Nhận dạng"):
            # Xử lý ảnh
            detections = pre_process(frame, net)
            if detections is not None:
                img, detected_fruits = post_process(frame.copy(), detections)

                # Thêm thông tin hiệu suất
                t, _ = net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                cv2.putText(img, label, (20, 40), FRUIT_FONT_FACE, FRUIT_FONT_SCALE, FRUIT_COLORS['RED'], FRUIT_THICKNESS, cv2.LINE_AA)

                # Hiển thị kết quả
                st.image(img, caption="Kết quả nhận dạng", use_column_width=True, channels="BGR")

                # Hiển thị danh sách trái cây được phát hiện
                if detected_fruits:
                    st.subheader("Trái cây được phát hiện:")
                    fruit_counts = {fruit: 0 for fruit in FRUIT_CLASSES}
                    for detection in detected_fruits:
                        fruit_counts[detection["class"]] += 1
                    for fruit, count in fruit_counts.items():
                        if count > 0:
                            st.write(f"- {fruit}: {count} trái")
                    total_fruits = sum(fruit_counts.values())
                    st.write(f"**Tổng số trái cây**: {total_fruits}")
                else:
                    st.warning("Không phát hiện được trái cây nào trong ảnh.")

                # Tùy chọn tải xuống ảnh đã xử lý
                if st.button("💾 Tải xuống ảnh đã xử lý"):
                    _, encoded_img = cv2.imencode('.png', img)
                    st.download_button(
                        label="Nhấn để tải xuống",
                        data=encoded_img.tobytes(),
                        file_name="processed_fruit.png",
                        mime="image/png"
                    )