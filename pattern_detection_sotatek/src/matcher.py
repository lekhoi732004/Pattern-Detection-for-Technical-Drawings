import cv2
import numpy as np
import torch
import torchvision.ops as ops
from PIL import Image

class DeepFeatureMatcher:
    """
    Lớp xử lý cốt lõi cho hệ thống nhận diện Pattern bản vẽ kỹ thuật (Zero-shot).
    Sử dụng thuật toán Template Matching kết hợp với các kỹ thuật xử lý ảnh (Morphology)
    và Dynamic Thresholding để đạt độ thu hồi (Recall) tối đa.
    """
    def __init__(self, device='cpu'):
        self.device = device
        print("[INFO] Khởi tạo hệ thống Template Matching - Maximum Recall (Độ quét sâu)")

    def _preprocess_pil_to_cv(self, image_pil):
        """
        Chuyển đổi ảnh từ định dạng PIL (thường dùng trong UI/Web) sang NumPy array (OpenCV).
        Đồng thời xử lý kênh Alpha (nền trong suốt) để tránh lỗi nhiễu đen.
        """
        # Nếu ảnh có kênh Alpha (RGBA - ảnh PNG nền trong suốt)
        if image_pil.mode == 'RGBA':
            background = Image.new("RGB", image_pil.size, (255, 255, 255))
            # Dán ảnh lên nền trắng tinh để khử nền trong suốt
            background.paste(image_pil, mask=image_pil.split()[3])
            image_cv = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2GRAY)
        else:
            # Ảnh bình thường (RGB/L), chuyển thẳng sang Grayscale (Ảnh xám)
            image_cv = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
        return image_cv

    def _crop_to_content(self, image_cv):
        """
        Tự động cắt xén (Auto-crop) bỏ các khoảng trắng thừa xung quanh Pattern mẫu,
        giúp thuật toán chỉ tập trung vào đặc trưng hình học của linh kiện.
        """
        # Nhị phân hóa: Chuyển nền trắng (240-255) thành đen, nét mực đen thành trắng
        _, thresh = cv2.threshold(image_cv, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Tìm tất cả các điểm ảnh (pixels) thuộc nét vẽ
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image_cv
            
        # Tìm khung chữ nhật nhỏ nhất bao quanh toàn bộ nét vẽ
        x, y, w, h = cv2.boundingRect(coords)
        padding = 4 # Thêm lề (padding) 4 pixel để pattern có không gian "thở"
        
        # Đảm bảo padding không bị tràn ra ngoài giới hạn kích thước gốc của ảnh
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_cv.shape[1] - x, w + padding * 2)
        h = min(image_cv.shape[0] - y, h + padding * 2)
        
        # Trả về ảnh đã được cắt gọt chuẩn xác
        return image_cv[y:y+h, x:x+w]

    def detect(self, drawing_pil, pattern_pil, auto_thresh=True, manual_threshold=0.55, iou_thresh=0.1, scales=None):
        """
        Hàm thực thi chính: Quét Pattern trên bản vẽ (Drawing) đa tỉ lệ, đa góc độ.
        """
        # =====================================================================
        # 1. TIỀN XỬ LÝ (PREPROCESSING)
        # =====================================================================
        drawing_cv = self._preprocess_pil_to_cv(drawing_pil)
        pattern_cv = self._preprocess_pil_to_cv(pattern_pil)
        pattern_cv = self._crop_to_content(pattern_cv)
        
        # Chuyển nét vẽ thành màu trắng, nền thành màu đen (Binarization)
        # Mục đích: Chuẩn hóa nét vẽ giữa bản vẽ và ảnh mẫu, loại bỏ nhiễu màu
        _, d_bin = cv2.threshold(drawing_cv, 128, 255, cv2.THRESH_BINARY_INV)
        _, p_bin = cv2.threshold(pattern_cv, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Dùng Kernel 2x2 làm đậm nét vẽ lên 1 pixel 
        # Mục đích: Bù trừ sai số khi nét vẽ trên bản vẽ mỏng/dày hơn so với ảnh mẫu
        kernel = np.ones((2, 2), np.uint8)
        d_bin = cv2.dilate(d_bin, kernel, iterations=1)
        p_bin = cv2.dilate(p_bin, kernel, iterations=1)
        
        # =====================================================================
        # 2. KHÔNG GIAN TÌM KIẾM (SEARCH SPACE: SCALES & ANGLES)
        # =====================================================================
        # Thiết lập 80 bước thu phóng (Scale) từ 10% đến 150% để vét mọi kích thước
        if scales is None:
            scales = np.linspace(0.1, 1.5, 80)
            
        all_boxes = []
        all_scores = []
        angles = [0, 90, 180, 270] # Quét 4 hướng trực giao cơ bản
        
        # Hạ baseline ban đầu xuống mức rất thấp (0.40) để thu thập toàn bộ ứng viên tiềm năng
        baseline_thresh = 0.40 if auto_thresh else manual_threshold
        
        for angle in angles:
            # Xoay Pattern theo góc chỉ định
            if angle == 0:
                rotated_pattern = p_bin
            elif angle == 90:
                rotated_pattern = cv2.rotate(p_bin, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_pattern = cv2.rotate(p_bin, cv2.ROTATE_180)
            elif angle == 270:
                rotated_pattern = cv2.rotate(p_bin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            h_rot, w_rot = rotated_pattern.shape
            
            # Trượt Pattern trên bản vẽ qua từng mức Scale
            for scale in scales:
                width = int(w_rot * scale)
                height = int(h_rot * scale)
                
                # Bỏ qua nếu kích thước scale quá nhỏ hoặc phình to hơn cả bản vẽ
                if width < 10 or height < 10 or width > d_bin.shape[1] or height > d_bin.shape[0]:
                    continue
                    
                resized_pattern = cv2.resize(rotated_pattern, (width, height), interpolation=cv2.INTER_AREA)
                
                # Cốt lõi: Dùng Normalized Cross-Correlation (TM_CCOEFF_NORMED) để chấm điểm giống nhau
                res = cv2.matchTemplate(d_bin, resized_pattern, cv2.TM_CCOEFF_NORMED)
                
                # Lọc thô: Chỉ giữ lại các vị trí có điểm số >= baseline_thresh (0.40)
                loc = np.where(res >= baseline_thresh)
                for pt in zip(*loc[::-1]):
                    x1, y1 = int(pt[0]), int(pt[1])
                    x2, y2 = x1 + width, y1 + height
                    score = float(res[pt[1], pt[0]])
                    
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)
                    
        if not all_boxes:
            return [], []
            
        # =====================================================================
        # 3. LỌC TRÙNG LẶP (NON-MAXIMUM SUPPRESSION - NMS)
        # =====================================================================
        # Chuyển dữ liệu sang PyTorch Tensors để tận dụng hàm NMS siêu tốc
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        
        # NMS sẽ gộp hàng ngàn bounding boxes đè lên nhau, chỉ giữ lại box có điểm cao nhất
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_thresh)
        
        nms_boxes = boxes_tensor[keep_indices].tolist()
        nms_scores = scores_tensor[keep_indices].tolist()
        
        # Trả về ngay lập tức nếu người dùng chọn mode Thủ công (Manual)
        if not auto_thresh:
            return nms_boxes, nms_scores
            
        # =====================================================================
        # 4. ADVANCED DYNAMIC THRESHOLDING
        # =====================================================================
        max_score = max(nms_scores)
        
        # Sàn cứng: Nếu Pattern khớp nhất mà điểm vẫn dưới 0.45 -> Nhận diện thất bại hoàn toàn
        if max_score < 0.45:
            return [], []
            
        scores_array = np.array(nms_scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        # Cách 1 - Ngưỡng Thống kê: Dựa vào phân phối điểm số (Trung bình + Độ lệch chuẩn nhỏ)
        # Mục đích: Giữ lại cả các linh kiện bị mờ, nét đứt
        stat_thresh = mean_score + 0.2 * std_score
        
        # Cách 2 - Ngưỡng Tỉ lệ: Dựa vào "Kẻ dẫn đầu" (Leader)
        # Mục đích: Chỉ giữ lại các linh kiện đạt ít nhất 80% độ hoàn hảo so với linh kiện tốt nhất
        ratio_thresh = max_score * 0.80
        
        # Quyết định Ngưỡng cuối (Dynamic Thresh):
        # Lấy giá trị thấp hơn (min) giữa Stat và Ratio để mở rộng khả năng bắt linh kiện.
        # Tuy nhiên, dựng "Sàn" ở mức 0.68 để chặn đứng các ký tự text (A, B, C...) bị nhận nhầm.
        dynamic_thresh = max(0.68, min(stat_thresh, ratio_thresh))
        
        print(f"[DEBUG] NMS Candidates: {len(nms_scores)} | Max Score: {max_score:.3f} | Thresh Áp dụng: {dynamic_thresh:.3f}")
        
        final_boxes = []
        final_scores = []
        
        # Lược trích lần cuối: Loại bỏ toàn bộ kết quả dưới ngưỡng Dynamic
        for box, score in zip(nms_boxes, nms_scores):
            if score >= dynamic_thresh:
                final_boxes.append(box)
                final_scores.append(score)
                
        return final_boxes, final_scores