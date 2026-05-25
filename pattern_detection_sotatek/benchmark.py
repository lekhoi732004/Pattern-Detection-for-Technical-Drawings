import time
from PIL import Image
from src.matcher import DeepFeatureMatcher

def run_benchmark():
    print("[INFO] Đang tải mô hình Benchmark...")
    matcher = DeepFeatureMatcher(device='cpu')
    # =========================================================================
    test_cases = [
        {
            "name": "Biến trở / Chiết áp", 
            "draw": "data/drawings/1.png", 
            "pat": "data/patterns/9.png", 
            "expected": 5
        },
        # ===============================================================
        {
            "name": "Điện trở (Zig-zag)", 
            "draw": "data/drawings/4.jpg", 
            "pat": "data/patterns/7.png", 
            "expected": 22
        },
        # ===============================================================
        {
            "name": "Tụ điện", 
            "draw": "data/drawings/4.jpg", 
            "pat": "data/patterns/6.png", 
            "expected": 5 
        }
    ]
    
    print("\n" + "="*75)
    print(f"{'Tên Test Case':<35} | {'Thực tế':<10} | {'Dự đoán':<10} | {'Thời gian (s)':<12}")
    print("="*75)
    
    total_time = 0
    total_expected = 0
    total_detected = 0
    
    for tc in test_cases:
        try:
            # Nạp ảnh
            draw_img = Image.open(tc["draw"]).convert("RGB")
            pat_img = Image.open(tc["pat"]).convert("RGB")
        except FileNotFoundError as e:
            print(f"[LỖI] Không tìm thấy file: {e.filename}. Vui lòng kiểm tra lại thư mục data!")
            continue
            
        # Bắt đầu bấm giờ
        start_time = time.time()
        
        # Chạy thuật toán (Bật chế độ Auto Threshold)
        boxes, scores = matcher.detect(draw_img, pat_img, auto_thresh=True)
        
        # Kết thúc bấm giờ
        end_time = time.time()
        
        latency = end_time - start_time
        detected = len(boxes)
        expected = tc["expected"]
        
        total_time += latency
        total_expected += expected
        total_detected += detected
        
        # In kết quả từng dòng
        print(f"{tc['name']:<35} | {expected:<10} | {detected:<10} | {latency:.3f}s")
        
    print("-" * 75)
    
    # Tính toán các chỉ số tổng quan
    if total_expected > 0:
        accuracy = (1 - abs(total_expected - total_detected) / total_expected) * 100
        avg_latency = total_time / len(test_cases)
        
        print(f"Tổng thời gian chạy : {total_time:.3f} giây")
        print(f"Tốc độ trung bình   : {avg_latency:.3f} giây / ảnh")
        print(f"Độ chính xác đếm    : {accuracy:.2f}%")
    else:
        print("Chưa có dữ liệu để tính toán.")
    print("="*75)

if __name__ == "__main__":
    run_benchmark()