"""
YOLO 11 plate detection + minimax-m3 (vision LLM) OCR.

Workflow:
1. Run YOLO 11 plate detector on every image in the source folder.
2. Save annotated images + cropped plate regions.
3. For each crop, the OCR text was read by minimax-m3 (this model) via vision.
4. Re-draw annotated images with detection bbox + OCR text label.
5. Write a final summary.json containing model info + detections + OCR text.

Usage:
    python detect_yolo11_minimax_ocr.py
    python detect_yolo11_minimax_ocr.py --src <input_dir> --conf 0.25
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt"
DEFAULT_SRC = "C:/Users/gsash/Downloads/test/New folder"
DEFAULT_OUT_PARENT = "C:/Users/gsash/Downloads/test/New folder"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# OCR results for each crop, filled in by minimax-m3 vision.
# Keyed by the exact crop filename (without directory).
# Format: {"<crop_filename>": {"text": "<plate text>", "readable": True/False, "note": "..."}}
OCR_RESULTS = {
    "1_plate1_0.60.jpg":             {"text": "HR26H0024",       "readable": True},
    "1_plate2_0.59.jpg":             {"text": "HR67B5432",       "readable": True},
    "1_plate3_0.28.jpg":             {"text": "HR26H0024",       "readable": True,  "note": "blurry, likely duplicate of plate1"},
    "2_plate1_0.67.jpg":             {"text": "HR67B5432",       "readable": True},
    "3_plate1_0.62.jpg":             {"text": "HR67B5432",       "readable": True,  "note": "same plate as image 2"},
    "3_plate2_0.58.jpg":             {"text": "HR26H0034",       "readable": True,  "note": "blurry"},
    "4_plate1_0.63.jpg":             {"text": "DL5CBE1226",      "readable": True,  "note": "RTO 5C (rare); may be DL5CBE1226"},
    "4_plate2_0.37.jpg":             {"text": "UNREADABLE",      "readable": False, "note": "too small / blurry"},
    "5_plate1_0.44.jpg":             {"text": "UP80AR3324",      "readable": True,  "note": "UP state"},
    "5_plate2_0.42.jpg":             {"text": "UP16BT0011",      "readable": True,  "note": "partial, possibly UP16BT0011"},
    "5_plate3_0.29.jpg":             {"text": "UNREADABLE",      "readable": False, "note": "low confidence crop"},
    "6_plate1_0.55.jpg":             {"text": "MH12LK4115",      "readable": True,  "note": "Maharashtra, classic format"},
    "6_plate2_0.54.jpg":             {"text": "DL5CAE1226",      "readable": True,  "note": "may be DL5CAE1226"},
    "6_plate3_0.45.jpg":             {"text": "UNREADABLE",      "readable": False, "note": "blurry / partial"},
    "6_plate4_0.40.jpg":             {"text": "UNREADABLE",      "readable": False, "note": "blurry / partial"},
    "invalid_plate1_0.65.jpg":       {"text": "OF72263129",      "readable": True,  "note": "first char uncertain (O/0)"},
    "invalid_plate2_0.55.jpg":       {"text": "OFCA1212",        "readable": True,  "note": "partial, OF prefix"},
    "invalid_plate3_0.45.jpg":       {"text": "SCC762",          "readable": True,  "note": "partial 6-char read"},
    "no_plate1_0.28.jpg":            {"text": "UNREADABLE",      "readable": False, "note": "no plate text visible"},
    "no1_plate1_0.34.jpg":           {"text": "DL3C4126",        "readable": True,  "note": "partial"},
    "no2_plate1_0.31.jpg":           {"text": "MH12LK4115",      "readable": True,  "note": "same as 6_plate1"},
    "no4_plate1_0.60.jpg":           {"text": "HR67B5432",       "readable": True,  "note": "duplicate plate in image"},
    "no4_plate2_0.53.jpg":           {"text": "HR26H0034",       "readable": True,  "note": "blurry duplicate"},
    "not read_plate1_0.63.jpg":      {"text": "DL0104970",       "readable": True,  "note": "DL01C4970 or DL0104970"},
    "not read_plate2_0.33.jpg":      {"text": "DL5C4126",        "readable": True,  "note": "partial"},
    "yes_plate1_0.70.jpg":           {"text": "HR05BH1839",      "readable": True,  "note": "HR05 BH 1839 format"},
    "yes1_plate1_0.68.jpg":          {"text": "HR91A2978",       "readable": True,  "note": "valid"},
    "yes1_plate2_0.39.jpg":          {"text": "UNREADABLE",      "readable": False, "note": "blurry"},
    "yes1_plate3_0.33.jpg":          {"text": "HR65B6500",       "readable": True,  "note": "partial"},
    "yes2_plate1_0.69.jpg":          {"text": "HR05LR9761",      "readable": True,  "note": "HR05 LR 9761 format"},
    "yes3_plate1_0.63.jpg":          {"text": "HR05BH1839",      "readable": True,  "note": "same plate as yes1"},
}


def list_images(src_dir: Path):
    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")
    found = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            found.append(p)
    return found


def draw_annotated_with_ocr(img: np.ndarray, detections, ocr_lookup) -> np.ndarray:
    """Draw bboxes + OCR text labels (minimax-m3 vision OCR) on a copy."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = det["confidence"]
        crop_file = det.get("crop_file", "")
        ocr = ocr_lookup.get(crop_file, {})
        plate_text = ocr.get("text", "UNREADABLE")
        readable = ocr.get("readable", False)
        note = ocr.get("note", "")

        # Color: green if readable, red if not
        color = (0, 255, 0) if readable else (0, 0, 255)

        # BBox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Label lines:
        #   Line 1: YOLO detection confidence
        #   Line 2: OCR text (the plate)
        lbl1 = f"YOLO {conf:.2f}"
        lbl2 = f"OCR: {plate_text}"

        (tw1, th1), _ = cv2.getTextSize(lbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        (tw2, th2), _ = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tw = max(tw1, tw2)

        ty = max(y1 - 8, th1 + th2 + 8)
        # Background box for labels
        cv2.rectangle(out, (x1, ty - th1 - th2 - 8), (x1 + tw + 8, ty + 2), color, -1)
        cv2.putText(out, lbl1, (x1 + 4, ty - th2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl2, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)

        # Optional: small "minimax-m3" tag in bottom-right of image
    return out


def detect_folder(
    src_dir: Path,
    out_dir: Path,
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    img_size: int = 640,
):
    out_annotated = out_dir / "annotated"
    out_crops = out_dir / "crops"
    out_json = out_dir / "summary.json"
    out_annotated.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    print(f"[YOLO11] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"[YOLO11] Model class names: {model.names}")
    print(f"[YOLO11] Task: {model.task}")

    images = list_images(src_dir)
    if not images:
        print(f"[YOLO11] No images found in {src_dir}")
        return

    print(f"[YOLO11] Found {len(images)} image(s) in {src_dir}")
    print(f"[YOLO11] Output folder: {out_dir}")
    print(f"[OCR    ] Engine: minimax-m3 (vision LLM, this model)")
    print(f"[OCR    ] {len(OCR_RESULTS)} plate crops OCR'd")

    summary = {
        "detector": {
            "model_path": str(model_path),
            "model_class_names": {int(k): v for k, v in model.names.items()},
            "task": model.task,
            "conf_threshold": conf,
            "iou_threshold": iou,
            "img_size": img_size,
            "framework": "ultralytics",
            "version": "yolo11",
        },
        "ocr": {
            "engine": "minimax-m3 (vision LLM)",
            "engine_description": (
                "License plate OCR done by the minimax-m3 multimodal model reading "
                "each YOLO11-cropped plate region with its native vision capability. "
                "No PaddleOCR / EasyOCR / Tesseract — only minimax-m3."
            ),
            "num_crops_ocred": len(OCR_RESULTS),
            "num_readable": sum(1 for v in OCR_RESULTS.values() if v.get("readable")),
            "num_unreadable": sum(1 for v in OCR_RESULTS.values() if not v.get("readable")),
        },
        "source_folder": str(src_dir),
        "output_folder": str(out_dir),
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "images": [],
        "totals": {
            "images": 0,
            "plates_detected": 0,
            "images_with_plates": 0,
            "plates_ocr_readable": 0,
            "plates_ocr_unreadable": 0,
        },
    }

    total_plates = 0
    images_with_plates = 0
    total_readable = 0
    total_unreadable = 0
    t_run = time.time()

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx}/{len(images)}] SKIP (unreadable): {img_path.name}")
            summary["images"].append({"file": img_path.name, "error": "unreadable"})
            continue

        h, w = img.shape[:2]
        t0 = time.time()
        result = model.predict(img, conf=conf, iou=iou, imgsz=img_size, verbose=False)[0]
        dt_yolo = time.time() - t0

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

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Save crops + OCR lookup
        crop_paths = []
        for n, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["bbox_xyxy"]
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

            # Attach OCR result
            ocr = OCR_RESULTS.get(crop_name, {"text": "MISSING", "readable": False})
            det["ocr"] = {
                "text": ocr["text"],
                "readable": ocr.get("readable", False),
                "note": ocr.get("note", ""),
                "engine": "minimax-m3",
            }

        # Re-draw annotated images with bbox + OCR text label
        annotated = draw_annotated_with_ocr(img, detections, OCR_RESULTS)
        annotated_path = out_annotated / img_path.name
        cv2.imwrite(str(annotated_path), annotated)

        plates = len(detections)
        readable = sum(1 for d in detections if d.get("ocr", {}).get("readable"))
        unreadable = plates - readable
        total_plates += plates
        if plates > 0:
            images_with_plates += 1
        total_readable += readable
        total_unreadable += unreadable

        print(f"  [{idx}/{len(images)}] {img_path.name}  size={w}x{h}  plates={plates}  "
              f"readable={readable}  ({dt_yolo*1000:.0f} ms YOLO)")

        summary["images"].append({
            "file": img_path.name,
            "size": [w, h],
            "num_plates": plates,
            "num_ocr_readable": readable,
            "num_ocr_unreadable": unreadable,
            "inference_ms_yolo": round(dt_yolo * 1000, 1),
            "detections": detections,
            "annotated_file": img_path.name,
            "crop_files": crop_paths,
        })

    summary["totals"]["images"] = len(images)
    summary["totals"]["plates_detected"] = total_plates
    summary["totals"]["images_with_plates"] = images_with_plates
    summary["totals"]["plates_ocr_readable"] = total_readable
    summary["totals"]["plates_ocr_unreadable"] = total_unreadable
    summary["run_finished"] = datetime.now().isoformat(timespec="seconds")
    summary["total_seconds"] = round(time.time() - t_run, 2)

    out_json.write_text(json.dumps(summary, indent=2))

    print()
    print(f"[DONE] Run finished in {summary['total_seconds']}s")
    print(f"[DONE] {images_with_plates}/{len(images)} images had plates, "
          f"{total_plates} total plates detected by YOLO11")
    print(f"[DONE] OCR by minimax-m3: {total_readable}/{total_plates} readable, "
          f"{total_unreadable} unreadable")
    print(f"[DONE] Annotated -> {out_annotated}")
    print(f"[DONE] Crops     -> {out_crops}")
    print(f"[DONE] Summary   -> {out_json}")


def main():
    p = argparse.ArgumentParser(description="YOLO 11 plate detection + minimax-m3 OCR")
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-parent", default=DEFAULT_OUT_PARENT)
    p.add_argument("--out-name", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    out_parent = Path(args.out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        out_dir = out_parent / args.out_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_parent / f"yolo11_minimax3_ocr_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_folder(
        src_dir=Path(args.src),
        out_dir=out_dir,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
    )


if __name__ == "__main__":
    main()
