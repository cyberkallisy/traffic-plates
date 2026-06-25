"""
YOLO11 vs YOLO12 — License-Plate-Only Detection Comparison
============================================================

Runs two pretrained plate-detection models (single class) on every image in a
test folder, saves annotated images to two separate result folders, and writes
a side-by-side JSON summary.

Inputs
------
- YOLO11 plate: Pikurrot/yolo11n-licenseplates  (class 0 = "License Plate")
- YOLO12 plate: capinowo/yolo12n-plate-checkpoint/best.pt  (class 0 = "License_Plate")

Output
------
- results_yolo11/<name>__yolo11.jpg   (annotated + detections.json)
- results_yolo12/<name>__yolo12.jpg   (annotated + detections.json)
- comparison.json                     (side-by-side summary)

Usage
-----
    python compare_yolo11_yolo12.py
    python compare_yolo11_yolo12.py --source <other_dir>
    python compare_yolo11_yolo12.py --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("compare-yolo-plates")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_YOLO11 = BASE_DIR / "results_yolo11"
OUT_YOLO12 = BASE_DIR / "results_yolo12"

YOLO11_WEIGHTS = BASE_DIR / "models_hf" / "yolo11n_licenseplates.pt"
YOLO12_WEIGHTS = BASE_DIR / "models_hf" / "best.pt"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Annotation helper (works for any single-class plate detector)
# ---------------------------------------------------------------------------
def annotate(image: np.ndarray, boxes, confs, model_name: str) -> np.ndarray:
    """Draw green box + confidence for every detection."""
    out = image.copy()
    for (x1, y1, x2, y2), cf in zip(boxes, confs):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        label = f"PLATE {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), (0, 200, 0), -1)
        cv2.putText(out, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # header banner
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, f"{model_name} — {len(boxes)} plate(s) detected",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Detection on one image
# ---------------------------------------------------------------------------
def detect_one(model: YOLO, image_path: Path, out_dir: Path, suffix: str,
               model_name: str, conf: float) -> Dict[str, Any]:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"image": image_path.name, "error": "cv2.imread failed"}

    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, verbose=False, device="cpu")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    r = results[0]
    boxes_xyxy = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
    confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
    cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist() if r.boxes is not None else []
    h, w = img.shape[:2]

    annotated = annotate(img, boxes_xyxy, confs, model_name)
    out_path = out_dir / f"{image_path.stem}__{suffix}.jpg"
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "image": image_path.name,
        "image_size": [w, h],
        "inference_ms": round(elapsed_ms, 2),
        "num_plates": len(boxes_xyxy),
        "class_names": [r.names[c] for c in cls_ids],
        "detections": [
            {
                "class_id": int(c),
                "class_name": r.names[c],
                "confidence": round(float(cf), 4),
                "bbox": [round(float(v), 1) for v in box],
            }
            for box, cf, c in zip(boxes_xyxy, confs, cls_ids)
        ],
        "annotated_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(source_dir: Path, conf: float) -> int:
    if not YOLO11_WEIGHTS.exists():
        log.error(f"YOLO11 weights missing: {YOLO11_WEIGHTS}")
        return 1
    if not YOLO12_WEIGHTS.exists():
        log.error(f"YOLO12 weights missing: {YOLO12_WEIGHTS}")
        return 1
    if not source_dir.exists():
        log.error(f"Source dir missing: {source_dir}")
        return 1

    OUT_YOLO11.mkdir(parents=True, exist_ok=True)
    OUT_YOLO12.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in VALID_EXT])
    if not images:
        log.error(f"No images found in {source_dir}")
        return 1
    log.info(f"Found {len(images)} image(s) in {source_dir}")

    # Load both models
    log.info("Loading YOLO11 plate detector ...")
    m11 = YOLO(str(YOLO11_WEIGHTS))
    log.info(f"  classes: {m11.names}")

    log.info("Loading YOLO12 plate detector ...")
    m12 = YOLO(str(YOLO12_WEIGHTS))
    log.info(f"  classes: {m12.names}")

    results_11: List[Dict[str, Any]] = []
    results_12: List[Dict[str, Any]] = []

    for i, img_path in enumerate(images, 1):
        log.info(f"[{i}/{len(images)}] {img_path.name}")
        r11 = detect_one(m11, img_path, OUT_YOLO11, "yolo11", "YOLO11", conf)
        r12 = detect_one(m12, img_path, OUT_YOLO12, "yolo12", "YOLO12", conf)
        results_11.append(r11)
        results_12.append(r12)
        log.info(f"    YOLO11: {r11.get('num_plates')} plate(s) in {r11.get('inference_ms')} ms  |  "
                 f"YOLO12: {r12.get('num_plates')} plate(s) in {r12.get('inference_ms')} ms")

    # Per-model detections JSON
    with open(OUT_YOLO11 / "detections.json", "w") as f:
        json.dump({
            "model": "yolo11",
            "weights": str(YOLO11_WEIGHTS),
            "source_dir": str(source_dir),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "conf_threshold": conf,
            "num_images": len(images),
            "total_plates": sum(r.get("num_plates", 0) for r in results_11),
            "results": results_11,
        }, f, indent=2)
    with open(OUT_YOLO12 / "detections.json", "w") as f:
        json.dump({
            "model": "yolo12",
            "weights": str(YOLO12_WEIGHTS),
            "source_dir": str(source_dir),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "conf_threshold": conf,
            "num_images": len(images),
            "total_plates": sum(r.get("num_plates", 0) for r in results_12),
            "results": results_12,
        }, f, indent=2)

    # Side-by-side comparison
    total11 = sum(r.get("num_plates", 0) for r in results_11)
    total12 = sum(r.get("num_plates", 0) for r in results_12)
    time11 = sum(r.get("inference_ms", 0.0) for r in results_11)
    time12 = sum(r.get("inference_ms", 0.0) for r in results_12)
    images_with_plate_11 = sum(1 for r in results_11 if r.get("num_plates", 0) > 0)
    images_with_plate_12 = sum(1 for r in results_12 if r.get("num_plates", 0) > 0)

    comparison = {
        "source_dir": str(source_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conf_threshold": conf,
        "num_images": len(images),
        "yolo11": {
            "weights": str(YOLO11_WEIGHTS),
            "classes": m11.names,
            "total_plates": total11,
            "images_with_at_least_one_plate": images_with_plate_11,
            "avg_plates_per_image": round(total11 / max(1, len(images)), 3),
            "total_inference_ms": round(time11, 1),
            "avg_inference_ms_per_image": round(time11 / max(1, len(images)), 1),
            "results_folder": str(OUT_YOLO11),
        },
        "yolo12": {
            "weights": str(YOLO12_WEIGHTS),
            "classes": m12.names,
            "total_plates": total12,
            "images_with_at_least_one_plate": images_with_plate_12,
            "avg_plates_per_image": round(total12 / max(1, len(images)), 3),
            "total_inference_ms": round(time12, 1),
            "avg_inference_ms_per_image": round(time12 / max(1, len(images)), 1),
            "results_folder": str(OUT_YOLO12),
        },
        "per_image": [
            {
                "image": results_11[i]["image"],
                "yolo11": {"plates": results_11[i].get("num_plates", 0),
                           "ms": results_11[i].get("inference_ms", 0.0)},
                "yolo12": {"plates": results_12[i].get("num_plates", 0),
                           "ms": results_12[i].get("inference_ms", 0.0)},
            }
            for i in range(len(images))
        ],
    }
    with open(BASE_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"  Images:                       {len(images)}")
    log.info(f"  YOLO11 total plates:          {total11} (on {images_with_plate_11} images)")
    log.info(f"  YOLO12 total plates:          {total12} (on {images_with_plate_12} images)")
    log.info(f"  YOLO11 total inference time:  {time11:.0f} ms  (avg {time11/max(1,len(images)):.0f} ms/img)")
    log.info(f"  YOLO12 total inference time:  {time12:.0f} ms  (avg {time12/max(1,len(images)):.0f} ms/img)")
    log.info("=" * 60)
    log.info(f"  YOLO11 results: {OUT_YOLO11}")
    log.info(f"  YOLO12 results: {OUT_YOLO12}")
    log.info(f"  Comparison:     {BASE_DIR / 'comparison.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help=f"Directory with test images (default: {DEFAULT_SOURCE})")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    args = p.parse_args()
    return run(args.source, args.conf)


if __name__ == "__main__":
    sys.exit(main())