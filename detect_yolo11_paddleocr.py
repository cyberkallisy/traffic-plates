"""
Plate detection with YOLO11 + PaddleOCR ONLY (no EasyOCR, no Tesseract).

Pipeline:
  1. YOLO11 plate detector (morsetechlab/yolov11-license-plate-detection)
     finds each plate ROI in the image.
  2. Each plate crop is preprocessed (CLAHE + adaptive threshold + upscale)
     and fed to PaddleOCR (English, mobile, angle-cls).
  3. We pick the best PaddleOCR reading per plate (highest confidence
     among the candidates that look like a plate).
  4. Annotated image + crops + JSON summary are written to the output dir.

Usage:
    python detect_yolo11_paddleocr.py
"""

import os
# PaddleOCR (paddlepaddle 2.6.x) ships a protobuf that fails to register on
# this Windows venv unless the pure-python protobuf backend is forced. See
# memory note — must be set before paddle imports.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
INPUT_DIR  = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_ROOT   = Path(r"C:/Users/gsash/Downloads/test/New folder")
YOLO11_PT  = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt")
CONF_THR   = 0.25      # YOLO plate-detection confidence (yolo11n is well-tuned)
OCR_CONF_THR = 0.30    # drop PaddleOCR lines below this confidence
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Output layout:  <New folder>/yolo11_paddleocr_<ts>/{annotated, crops, summary.json}

# Indian-plate regexes — only used to PICK the best OCR candidate, not to
# reject outputs (lowercase/garbled plates still go to summary).
INDIAN_RE     = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
INDIAN_BH_RE  = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
INDIAN_VALID  = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HR","HP",
    "JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TS","TR","UP","UK","WB","BH",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("yolo11-paddleocr")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _clean_text(s: str) -> str:
    """Strip noise, normalise to A-Z0-9."""
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _is_indian_plate(text: str) -> bool:
    return bool(INDIAN_RE.match(text) or INDIAN_BH_RE.match(text))


def _preprocess_plate(crop_bgr: np.ndarray, target_h: int = 200) -> np.ndarray:
    """CLAHE + adaptive threshold + upscale, returning a 3-channel BGR
    image that PaddleOCR's mobile detector likes."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    if max(h, w) < 120:
        s = target_h / max(h, 1)
        crop_bgr = cv2.resize(crop_bgr, (int(w * s), int(h * s)),
                              interpolation=cv2.INTER_CUBIC)
    # White-pad (PaddleOCR's text detector likes clean borders)
    pad = 20
    crop_bgr = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _paddle_ocr_best(ocr: PaddleOCR, crop_bgr: np.ndarray) -> dict:
    """Run PaddleOCR on a plate crop, return the best reading.

    Strategy:
      - try the raw crop AND its preprocessed variant, keep all candidates
      - normalise each line, drop those below OCR_CONF_THR
      - prefer a candidate that matches an Indian plate format
      - tie-break by raw confidence
    """
    candidates = []  # list of dicts: text, conf, valid_format
    for label, img in (("raw", crop_bgr),
                       ("preproc", _preprocess_plate(crop_bgr))):
        if img is None or img.size == 0:
            continue
        try:
            res = ocr.ocr(img, cls=True)
        except Exception as e:
            log.warning("PaddleOCR failed on %s variant: %s", label, e)
            continue
        if not res or not res[0]:
            continue
        for line in res[0]:
            try:
                bbox, (text, conf) = line
            except Exception:
                continue
            text = _clean_text(text)
            if not text or conf < OCR_CONF_THR:
                continue
            candidates.append({
                "text": text,
                "conf": float(conf),
                "valid_format": _is_indian_plate(text),
                "variant": label,
            })

    if not candidates:
        return {"text": "", "conf": 0.0, "valid_format": False, "raw": []}

    valid = [c for c in candidates if c["valid_format"]]
    pool  = valid if valid else candidates
    pool.sort(key=lambda c: c["conf"], reverse=True)
    best = pool[0]
    return {
        "text": best["text"],
        "conf": round(best["conf"], 3),
        "valid_format": best["valid_format"],
        "raw": candidates,  # every OCR line Paddle returned
    }


def _annotate(img_bgr: np.ndarray, plates: list) -> np.ndarray:
    """Draw plate bboxes + OCR text on a copy of the image."""
    out = img_bgr.copy()
    for i, p in enumerate(plates, 1):
        x1, y1, x2, y2 = p["bbox"]
        text = p["ocr"]["text"] or "(no text)"
        conf = p["ocr"]["conf"]
        valid = p["ocr"]["valid_format"]
        color = (0, 200, 0) if valid else (0, 165, 255)  # green / orange
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"#{i} {text} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - th - 8, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = OUT_ROOT / f"yolo11_paddleocr_{ts}"
    annot_dir = out_dir / "annotated"
    crops_dir = out_dir / "crops"
    annot_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading YOLO11 from %s", YOLO11_PT)
    t0 = time.time()
    yolo = YOLO(str(YOLO11_PT))
    names = yolo.names  # dict {int: str}
    log.info("YOLO11 loaded in %.1fs. Classes: %s", time.time() - t0, names)

    log.info("Loading PaddleOCR (en, mobile, angle-cls) ...")
    t0 = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False,
                    use_gpu=False, det_db_thresh=0.3, det_db_box_thresh=0.5)
    log.info("PaddleOCR ready in %.1fs", time.time() - t0)

    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT)
    log.info("Found %d images in %s", len(images), INPUT_DIR)

    summary = {
        "run_id": f"yolo11_paddleocr_{ts}",
        "input_dir": str(INPUT_DIR),
        "output_dir": str(out_dir),
        "detector": "yolo11_plate.pt (morsetechlab/yolov11-license-plate-detection)",
        "ocr_engine": "PaddleOCR 2.7 (en, mobile, angle-cls, CPU) — ONLY",
        "yolo_conf_threshold": CONF_THR,
        "ocr_conf_threshold": OCR_CONF_THR,
        "images_processed": 0,
        "plates_total": 0,
        "plates_valid_format": 0,
        "results": [],
    }

    for idx, img_path in enumerate(images, 1):
        log.info("[%d/%d] %s", idx, len(images), img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("  could not read, skipping")
            continue
        h, w = img.shape[:2]

        # --- YOLO11 plate detection ---
        t0 = time.time()
        det = yolo.predict(img, conf=CONF_THR, verbose=False)[0]
        yolo_ms = (time.time() - t0) * 1000

        plates = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()
            clses = det.boxes.cls.cpu().numpy().astype(int)
            for box, c, cls in zip(xyxy, confs, clses):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img[y1:y2, x1:x2].copy()
                # Save crop
                stem = img_path.stem
                crop_name = f"{stem}_plate{len(plates)+1}_{c:.2f}.jpg"
                cv2.imwrite(str(crops_dir / crop_name), crop)

                # --- PaddleOCR ---
                t1 = time.time()
                ocr_res = _paddle_ocr_best(ocr, crop)
                ocr_ms = (time.time() - t1) * 1000

                plates.append({
                    "plate_idx": len(plates) + 1,
                    "bbox": [x1, y1, x2, y2],
                    "yolo_conf": round(float(c), 3),
                    "class_id": int(cls),
                    "class_name": names.get(int(cls), str(cls)),
                    "ocr": ocr_res,
                    "timing_ms": {
                        "yolo": round(yolo_ms, 1),
                        "paddleocr": round(ocr_ms, 1),
                    },
                    "crop_file": crop_name,
                })

        # --- Annotate ---
        annotated = _annotate(img, plates)
        annot_name = f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(annot_dir / annot_name), annotated)

        n_plates = len(plates)
        n_valid  = sum(1 for p in plates if p["ocr"]["valid_format"])
        summary["images_processed"] += 1
        summary["plates_total"]     += n_plates
        summary["plates_valid_format"] += n_valid
        summary["results"].append({
            "image": img_path.name,
            "image_size": [w, h],
            "num_plates": n_plates,
            "num_valid_format": n_valid,
            "plates": plates,
            "annotated_file": annot_name,
        })
        # Console one-liner
        txts = [p["ocr"]["text"] for p in plates if p["ocr"]["text"]]
        log.info("  -> %d plate(s) | OCR: %s | valid=%d",
                 n_plates, txts or "[]", n_valid)

    # --- Persist summary ---
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("=" * 60)
    log.info("Done. %d images, %d plates (%d valid format).",
             summary["images_processed"],
             summary["plates_total"],
             summary["plates_valid_format"])
    log.info("Summary: %s", summary_path)
    log.info("Annotated: %s", annot_dir)
    log.info("Crops:     %s", crops_dir)


if __name__ == "__main__":
    main()