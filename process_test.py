import os
import glob
from detector import TrafficPlateDetector

input_dir = r"C:\\Users\\gsash\\Downloads\\test\\New folder"
output_dir = r"C:\\Users\\gsash\\Downloads\\test"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize detector with output directory
detector = TrafficPlateDetector(
    yolo_path=r"C:/Users/gsash/Downloads/vnpr/models/models--Koushim--yolov8-license-plate-detection/snapshots/9aaa5cd490abe0c165882ba87f4f62658ab54d01/best.pt",
    coco_yolo_path=r"C:/Users/gsash/yolov8n.pt",
    crops_dir=os.path.join(output_dir, "crops"),
    results_dir=output_dir
)

detector.warmup()

# Process all images
for img_path in glob.glob(os.path.join(input_dir, "*.*")):
    if img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        print(f"Processing {os.path.basename(img_path)}...")
        result = detector.detect(img_path)
        print(f"  Found {result['num_plates']} plates")

print(f"\nResults saved to: {output_dir}")