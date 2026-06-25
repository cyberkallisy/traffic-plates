import os
import cv2
import numpy as np
import json
from pathlib import Path
from detector import TrafficPlateDetector
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import pytesseract

# Paths
INPUT_DIR = r"C:\\Users\\gsash\\Downloads\\test\\New folder"
OUTPUT_DIR = INPUT_DIR

# Initialize detectors
paddle_detector = TrafficPlateDetector(
    yolo_path=r"C:/Users/gsash/Downloads/vnpr/models/models--Koushim--yolov8-license-plate-detection/snapshots/9aaa5cd490abe0c165882ba87f4f62658ab54d01/best.pt",
    coco_yolo_path=r"C:/Users/gsash/yolov8n.pt",
    crops_dir=os.path.join(OUTPUT_DIR, "crops"),
    results_dir=os.path.join(OUTPUT_DIR, "results")
)

# Load Qwen model
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
qwen_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True,
)

# Process each image
for img_path in Path(INPUT_DIR).glob("*.*"):
    if not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        continue

    print(f"Processing: {img_path.name}")
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # 1. PaddleOCR pipeline
    paddle_result = paddle_detector.detect(str(img_path))
    paddle_plates = paddle_result["plates"]

    # 2. Qwen 2.5 VL
    image = Image.open(str(img_path)).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Detect number plates and read text"},
        ],
    }]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[image], padding=True, return_tensors="pt").to(qwen_model.device)
    output_ids = qwen_model.generate(**inputs, max_new_tokens=2048)
    response = qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    qwen_plates = []
    # Extract Qwen results (simplified)
    if "bbox" in response and "text" in response:
        qwen_plates = [{"bbox": [100,100,200,200], "text": "QWEN", "model": "qwen"}]

    # 3. Tesseract (minimax)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tesseract_text = pytesseract.image_to_string(gray)
    tesseract_plates = [{"bbox": [300,300,400,400], "text": tesseract_text.strip(), "model": "minimax"}]

    # Combine results
    combined_img = img.copy()
    colors = {
        "paddle": (0, 255, 0),
        "qwen": (255, 0, 0),
        "minimax": (0, 0, 255)
    }

    # Draw all results
    for plate in paddle_plates:
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), colors["paddle"], 2)
        cv2.putText(combined_img, f"Paddle: {plate['text']}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["paddle"], 2)

    for plate in qwen_plates:
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), colors["qwen"], 2)
        cv2.putText(combined_img, f"Qwen: {plate['text']}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["qwen"], 2)

    for plate in tesseract_plates:
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), colors["minimax"], 2)
        cv2.putText(combined_img, f"Tesseract: {plate['text']}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["minimax"], 2)

    # Save combined result
    output_path = os.path.join(OUTPUT_DIR, f"combined_{img_path.name}")
    cv2.imwrite(output_path, combined_img)
    print(f"Saved combined result to: {output_path}")