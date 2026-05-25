import gradio as gr
from PIL import Image
from src.matcher import DeepFeatureMatcher
from src.utils import draw_bounding_boxes, format_results_to_json

print("Đang tải mô hình...")
matcher = DeepFeatureMatcher(device='cpu')
print("Khởi tạo hoàn tất! Sẵn sàng phục vụ UI.")

def process_images(pattern_img, drawing_img, auto_thresh, manual_threshold):
    if pattern_img is None or drawing_img is None:
        return None, {"error": "Vui lòng upload đầy đủ cả Pattern và Drawing."}
    
    # Giao việc tính toán cho module detect
    boxes, scores = matcher.detect(
        drawing_pil=drawing_img, 
        pattern_pil=pattern_img, 
        auto_thresh=auto_thresh,
        manual_threshold=manual_threshold, 
        iou_thresh=0.1 
    )
    
    if not boxes:
        return drawing_img, {"message": "Không tìm thấy pattern nào phù hợp trên bản vẽ."}
        
    output_image = draw_bounding_boxes(drawing_img, boxes, scores)
    json_results = format_results_to_json(boxes, scores)
    
    return output_image, json_results

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Hệ thống Phát hiện Pattern Kỹ thuật (Zero-Shot)
        **SotaTek - AI Team Home Assessment** | Upload một ký hiệu bất kỳ và bản vẽ BOM kỹ thuật. 
        Hệ thống tích hợp công nghệ **Dynamic Thresholding** tự động điều chỉnh độ nhạy để bắt chính xác pattern.
        """
    )
     
    with gr.Row():
        with gr.Column(scale=1):
            pattern_input = gr.Image(type="pil", label="1. Upload Pattern (Ký hiệu)")
            gr.Markdown("### Tùy chỉnh (Hyperparameters)")
            
            # Checkbox kích hoạt tính năng tự động (Mặc định: Bật)
            auto_thresh_cb = gr.Checkbox(label="✨Tự động tính toán Threshold (Dynamic Threshold)", value=True)
            
            # Thanh trượt này chỉ có tác dụng nếu user tắt Checkbox ở trên
            threshold_slider = gr.Slider(minimum=0.4, maximum=0.99, value=0.65, step=0.01, label="Confidence Threshold (Thủ công)")
            
        with gr.Column(scale=2):
            drawing_input = gr.Image(type="pil", label="2. Upload Drawing (Bản vẽ kỹ thuật)")
            
    submit_btn = gr.Button("🚀 Chạy Inference", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("### Kết quả dự đoán")
    
    with gr.Row():
        with gr.Column(scale=2):
            image_output = gr.Image(type="pil", label="Bản vẽ Output")
        with gr.Column(scale=1):
            json_output = gr.JSON(label="Tọa độ Bounding Box & Scores")
            
    submit_btn.click(
        fn=process_images,
        inputs=[pattern_input, drawing_input, auto_thresh_cb, threshold_slider],
        outputs=[image_output, json_output]
    )

    # =====================================================================
    #VÍ DỤ MẪU (EXAMPLES) DÀNH CHO REVIEWER
    # =====================================================================
    gr.Markdown("---")
    gr.Markdown("### 📌 Ví dụ Test Nhanh (Click và ấn 🚀 Chạy Inference)")
    
    gr.Examples(
        examples=[
            # [pattern, drawing, auto_thresh, manual_threshold]
            ["data/patterns/9.png", "data/drawings/1.png", True, 0.65],
            ["data/patterns/2.png", "data/drawings/2.jpg", True, 0.65],
            ["data/patterns/5.png", "data/drawings/4.jpg", True, 0.65]
        ],
        inputs=[pattern_input, drawing_input, auto_thresh_cb, threshold_slider],
        outputs=[image_output, json_output],
        fn=process_images,
        cache_examples=False, # Tắt cache để tránh sinh file rác trên ổ cứng
        label="Chọn một ví dụ dưới đây:"
    )

if __name__ == "__main__":
    demo.launch()