"""
YOLO12 Number-Plate Detection Only (No OCR, no vehicle detection)
================================================================
Runs the YOLO12 plate-detection model on every image in a source folder
and writes the output to TWO folders:

  1. results_yolo12_annotated/  - full image with plate bbox drawn
  2. results_yolo12_crops/      - just the cropped plate region(s)

Also writes a JSON summary next to each image so you can inspect the
detections programmatically.

Usage:
    python detect_yolo12_only.py
    python detect_yolo12_only.py --src <input_dir> --conf 0.25
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths (defaults match this project layout)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = (
    "C:/Users/gsash/Downloads/traffic-plates/models_hf/best.pt"  # YOLO12 plate
)
DEFAULT_SRC = "C:/Users/gsash/Downloads/test/New folder"
DEFAULT_OUT_ANNOTATED = "C:/Users/gsash/Downloads/traffic-plates/results_yolo12_annotated"
DEFAULT_OUT_CROPS = "C:/Users/gsash/Downloads/traffic-plates/results_yolo12_crops"
DEFAULT_OUT_JSON = "C:/Users/gsash/Downloads/traffic-plates/results_yolo12_summary.json"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(src_dir: Path):
    """Yield (path, filename) for every supported image in src_dir (non-recursive)."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")
    found = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            found.append(p)
    return found


def draw_annotation(img: np.ndarray, detections, label_prefix: str = "plate") -> np.ndarray:
    """Draw bboxes + labels on a copy of the image."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = det["confidence"]
        # Green box, white outline + black text for readability on any bg
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
        label = f"{label_prefix} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out


def detect_folder(
    src_dir: Path,
    out_annotated: Path,
    out_crops: Path,
    out_json: Path,
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    img_size: int = 640,
):
    out_annotated.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    print(f"[YOLO12] Loading model: {model_path}")
    model = YOLO(model_path)
    # Print the class names so we know what we're predicting against
    print(f"[YOLO12] Model class names: {model.names}")

    images = list_images(src_dir)
    if not images:
        print(f"[YOLO12] No images found in {src_dir} (extensions: {sorted(VALID_EXTS)})")
        return

    print(f"[YOLO12] Found {len(images)} image(s) in {src_dir}")

    summary = {
        "model": str(model_path),
        "source_folder": str(src_dir),
        "annotated_folder": str(out_annotated),
        "crops_folder": str(out_crops),
        "conf_threshold": conf,
        "iou_threshold": iou,
        "img_size": img_size,
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "images": [],
        "totals": {"images": 0, "plates_detected": 0, "images_with_plates": 0},
    }

    total_plates = 0
    images_with_plates = 0
    t_run = time.time()

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx}/{len(images)}] SKIP (unreadable): {img_path.name}")
            summary["images"].append({
                "file": img_path.name,
                "error": "unreadable",
            })
            continue

        h, w = img.shape[:2]
        t0 = time.time()
        # Run YOLO12 on this image. verbose=False keeps the console clean.
        result = model.predict(img, conf=conf, iou=iou, imgsz=img_size, verbose=False)[0]
        dt = time.time() - t0

        # Build per-image detection list
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf_val = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names.get(cls_id, str(cls_id))
                detections.append({
                    "bbox_xyxy": xyxy,
                    "confidence": round(conf_val, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        # Sort by confidence desc so the "best" crop is named first
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # ----- Folder 1: annotated full image -----
        annotated = draw_annotation(img, detections)
        annotated_path = out_annotated / img_path.name
        cv2.imwrite(str(annotated_path), annotated)

        # ----- Folder 2: cropped plate regions -----
        crop_paths = []
        for n, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["bbox_xyxy"]
            # Clamp to image bounds (YOLO can occasionally overshoot)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            crop = img[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            stem = img_path.stem
            crop_name = f"{stem}_plate{n}_{det['confidence']:.2f}.jpg"
            crop_path = out_crops / crop_name
            cv2.imwrite(str(crop_path), crop)
            det["crop_file"] = crop_name
            crop_paths.append(crop_name)

        plates = len(detections)
        total_plates += plates
        if plates > 0:
            images_with_plates += 1
        print(f"  [{idx}/{len(images)}] {img_path.name}  size={w}x{h}  plates={plates}  ({dt*1000:.0f} ms)")

        summary["images"].append({
            "file": img_path.name,
            "size": [w, h],
            "num_plates": plates,
            "inference_ms": round(dt * 1000, 1),
            "detections": detections,
            "annotated_file": img_path.name,
            "crop_files": crop_paths,
        })

    summary["totals"]["images"] = len(images)
    summary["totals"]["plates_detected"] = total_plates
    summary["totals"]["images_with_plates"] = images_with_plates
    summary["run_finished"] = datetime.now().isoformat(timespec="seconds")
    summary["total_seconds"] = round(time.time() - t_run, 2)

    out_json.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[YOLO12] Done in {summary['total_seconds']}s")
    print(f"[YOLO12] {images_with_plates}/{len(images)} images had plates, {total_plates} total plates")
    print(f"[YOLO12] Annotated -> {out_annotated}")
    print(f"[YOLO12] Crops     -> {out_crops}")
    print(f"[YOLO12] Summary   -> {out_json}")


def main():
    p = argparse.ArgumentParser(description="YOLO12-only plate detection (no OCR).")
    p.add_argument("--src", default=DEFAULT_SRC, help="Source folder with images")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO12 .pt file")
    p.add_argument("--out-annotated", default=DEFAULT_OUT_ANNOTATED)
    p.add_argument("--out-crops", default=DEFAULT_OUT_CROPS)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = p.parse_args()

    detect_folder(
        src_dir=Path(args.src),
        out_annotated=Path(args.out_annotated),
        out_crops=Path(args.out_crops),
        out_json=Path(args.out_json),
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
    )


if __name__ == "__main__":
    main()