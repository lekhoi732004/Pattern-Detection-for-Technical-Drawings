# 🔍 Zero-Shot Technical Pattern Detection System

Hệ thống phát hiện linh kiện điện tử và ký hiệu kỹ thuật trên bản vẽ CAD/B.O.M dựa trên phương pháp **Advanced Zero-Shot Template Matching**. Hệ thống không yêu cầu huấn luyện trước (No Training Required) và có thể hoạt động trực tiếp thông qua một hình ảnh ký hiệu mẫu (Pattern) duy nhất.

---

## ✨ Các tính năng kỹ thuật nổi bật
* **Tốc độ xử lý đa luồng (Multi-threading):** Tối ưu hóa không gian tìm kiếm Affine (quét 8 góc xoay và 80 mức scale) bằng `ThreadPoolExecutor`, giúp nén thời gian Inference xuống tiệm cận Real-time (1 - 2 giây/bản vẽ).
* **Binarization & Otsu's Thresholding:** Loại bỏ hoàn toàn nhiễu sáng và nền giấy, bảo toàn các khe hở siêu nhỏ của linh kiện (như cổng XOR) mà không làm biến dạng nét vẽ.
* **Contextual IoU Verification:** Bộ lọc ngữ cảnh hình học thông minh giúp chống nhận nhầm các ký hiệu có cấu trúc chồng chéo (ví dụ: phân biệt hoàn hảo cổng XOR và XNOR).
* **Population-Aware Dynamic Thresholding:** AI tự động phân tích độ "đặc trưng" của ký hiệu để thiết lập ngưỡng Threshold linh hoạt (Strict Peak Isolation), loại bỏ triệt để rác từ viền tọa độ hoặc text.

---

## 📂 Cấu trúc thư mục (Repository Structure)

```text
PATTERN_DETECTION_SOTATEK/
├── data/                  # Chứa dữ liệu ảnh test
│   ├── drawings/          # Các bản vẽ kỹ thuật (Schematics/CAD)
│   └── patterns/          # Các ảnh ký hiệu mẫu (Templates)
├── src/                   # Source code thuật toán lõi
│   ├── matcher.py         # Module thuật toán DeepFeatureMatcher
│   └── utils.py           # Tiện ích vẽ Bounding Box và xuất JSON
├── app.py                 # File khởi động Web UI (Gradio)
├── benchmark.py           # Script đánh giá hiệu năng (Speed & Accuracy)
└── requirements.txt       # Danh sách thư viện phụ thuộc
```
## ⚙️ Hướng dẫn cài đặt (Installation)

Dự án sử dụng môi trường Python tiêu chuẩn. Khuyến nghị sử dụng Python 3.11.

**Bước 1: Clone Repository**

```bash
git clone <đường-dẫn-repo-github-của-bạn>
cd pattern_detection_sotatek

```

**Bước 2: Tạo môi trường ảo và Cài đặt thư viện**

* Đối với hệ điều hành **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

* Đối với hệ điều hành **Linux/macOS**:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

*(Hệ thống sử dụng `opencv-python-headless` để tối ưu dung lượng và chống lỗi GUI khi deploy lên Server/Docker).*

---

## 🚀 Cách chạy ứng dụng (How to Run)

### 1. Khởi chạy Giao diện Web (Gradio UI)

Đây là cách chính để tương tác với hệ thống và xem kết quả trực quan.

```bash
python app.py

```

* Sau khi chạy lệnh, truy cập địa chỉ: `http://127.0.0.1:7860` trên trình duyệt Web.
* Giao diện cung cấp sẵn các ví dụ mẫu (Examples) ở cuối trang để Reviewer có thể test nhanh với 1 click.

### 2. Chạy Script Đánh giá Hiệu năng (Benchmark CLI)

Để kiểm tra tốc độ (Latency) và độ chính xác (Counting Accuracy) của mô hình trên tập Test Cases nội bộ.

```bash
python benchmark.py

```

---

## 🎯 Hướng dẫn chạy Inference (Inference Example)

Nếu bạn muốn test hệ thống trên giao diện Web, hãy làm theo các bước sau:
1. **Crop pattern:** Crop pattern từ Drawing bạn muốn test
2. **Upload Pattern (Mục 1):** Tải lên hình ảnh ký hiệu cần tìm. *Lưu ý: Nên crop pattern gọn gàng, hạn chế dính quá nhiều đoạn dây nối dài thừa thãi.*
3. **Upload Drawing (Mục 2):** Tải lên bản vẽ kỹ thuật/Schematic tổng thể.
4. **Tùy chỉnh Hyperparameters:**
* Mặc định, hệ thống sẽ bật **✨ Tự động tính Threshold (Dynamic)** để AI tự phân tích và đưa ra ngưỡng cắt tốt nhất.
* Nếu muốn tinh chỉnh thủ công, hãy bỏ tick ô tự động và kéo thanh trượt **Confidence Threshold** (ví dụ: nâng lên 0.75 nếu bản vẽ quá nhiễu, hoặc hạ xuống 0.50 nếu bản vẽ bị mờ).


5. **Click "🚀 Chạy Inference":** Chờ khoảng 1-2 giây để hệ thống xử lý.
6. **Xem Kết quả:**
* **Bản vẽ Output:** Trả về hình ảnh gốc đã được vẽ các Bounding Box đỏ kèm điểm số Confidence Score. Thuật toán tự động chống tràn viền text nếu linh kiện nằm sát mép ảnh.
* **Tọa độ JSON:** Trả về định dạng JSON chuẩn chứa `id`, `bbox` (x_min, y_min, x_max, y_max) và `confidence_score`, sẵn sàng để tích hợp vào các hệ thống Microservices khác.


