import cv2
import numpy as np
from PIL import Image

def draw_bounding_boxes(image_pil, boxes, scores):
    """
    Vẽ bounding boxes và điểm confidence score lên ảnh bản vẽ.
    
    Args:
        image_pil (PIL.Image): Ảnh bản vẽ gốc.
        boxes (list): Danh sách các bounding box [x1, y1, x2, y2].
        scores (list): Danh sách điểm confidence tương ứng.
        
    Returns:
        PIL.Image: Ảnh đã được vẽ bounding boxes.
    """
    # Chuyển từ định dạng PIL (RGB) sang NumPy array của OpenCV (BGR)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # 1. Vẽ khung chữ nhật (Màu đỏ BGR: 0, 0, 255, độ dày: 2)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 2. Tạo nhãn text chứa confidence score (làm tròn 2 chữ số)
        label = f"{score:.2f}"
        
        # 3. Tính toán kích thước khối text để tạo background cho chữ dễ đọc
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Vẽ background màu đỏ cho text
        cv2.rectangle(image_cv, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
        
        # Ghi chữ màu trắng lên trên background đỏ
        cv2.putText(image_cv, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Chuyển ngược lại từ định dạng OpenCV (BGR) sang PIL (RGB)
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))


def format_results_to_json(boxes, scores):
    """
    Chuyển đổi kết quả danh sách box và score thành định dạng JSON chuẩn.
    
    Args:
        boxes (list): Danh sách các bounding box [x1, y1, x2, y2].
        scores (list): Danh sách điểm confidence tương ứng.
        
    Returns:
        list of dicts: Kết quả đã được format để xuất ra UI hoặc API.
    """
    results = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        results.append({
            "id": i + 1,
            "bbox": {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2
            },
            "confidence_score": round(score, 4)
        })
    return results