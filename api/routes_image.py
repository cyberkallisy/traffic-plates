import os
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, render_template, request, redirect, send_from_directory

from core.engine import engine, _read_image, _decode_b64

image_bp = Blueprint("image", __name__)

HERE = Path(__file__).resolve().parent.parent
YOLO_MODEL = HERE / "yolo11_plate.pt"
REPORT_PATH = HERE / "test" / "New folder" / "images" / "yolo11_awiros_ocr_run" / "report.html"
UPLOAD_DIR = HERE / "uploads"
RESULTS_DIR = HERE / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_upload(file_storage) -> str:
    suffix = Path(file_storage.filename or "").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False)
    file_storage.save(tmp.name)
    tmp.close()
    return tmp.name

def _save_result_image(stem: str, img_bgr) -> str:
    import cv2
    fname = f"{stem}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(RESULTS_DIR / fname), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"/results/{fname}"

def _save_crop_image(stem: str, img_bgr) -> str:
    import cv2
    fname = f"{stem}_crop_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(RESULTS_DIR / fname), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"/results/{fname}"

@image_bp.get("/")
def index():
    return render_template(
        "index.html",
        yolo_model=YOLO_MODEL.name,
        report_url=str(REPORT_PATH) if REPORT_PATH.exists() else "",
    )

@image_bp.post("/api/predict")
def api_predict():
    try:
        if "file" not in request.files:
            return jsonify(error="No file uploaded. Send a file in the 'file' field."), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify(error="Empty filename."), 400
        path = _save_upload(f)
        try:
            img = _read_image(path)
            t0 = time.time()
            out = engine.predict(img, conf=float(request.form.get("conf", 0.25)))
            out["total_ms"] = round((time.time() - t0) * 1000, 1)
            out["filename"] = f.filename
            out["timestamp"] = datetime.now().isoformat(timespec="seconds")
            return jsonify(out)
        finally:
            try: os.unlink(path)
            except OSError: pass
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@image_bp.post("/api/detect")
def api_detect():
    try:
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="No image uploaded. Send a file in the 'image' field."), 400

        conf = float(request.form.get("conf", 0.25))
        stem = Path(f.filename).stem
        path = _save_upload(f)
        try:
            img = _read_image(path)
            t0 = time.time()
            out = engine.predict(img, conf=conf)
            elapsed = round(time.time() - t0, 2)

            plates = []
            for i, det in enumerate(out["detections"], 1):
                ocr = det["ocr"]
                text = ocr.get("text", "") or ""
                ocr_conf = ocr.get("confidence", 0.0)
                valid_format = bool(ocr.get("readable")) and len(text) >= 4
                x1, y1, x2, y2 = det["bbox_xyxy"]
                H, W = img.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(W, x2), min(H, y2)
                crop_bgr = img[y1c:y2c, x1c:x2c]
                crop_url = _save_crop_image(f"{stem}_{i}", crop_bgr) if crop_bgr.size else ""
                plates.append({
                    "text": text if text else None,
                    "valid_format": valid_format,
                    "detection_confidence": det["confidence"],
                    "ocr_confidence": ocr_conf,
                    "state_code": "",
                    "format_type": "standard",
                    "crop_url": crop_url,
                    "bbox_xyxy": det["bbox_xyxy"],
                    "raw_ocr": text,
                    "fixes_applied": [],
                    "engine": "Awiros ANPR-OCR",
                })

            annotated_url = _save_result_image(stem, _decode_b64(out["annotated_jpeg_b64"]))
            response = {
                "filename": f.filename,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "size": out["size"],
                "num_plates": len(plates),
                "num_valid": sum(1 for p in plates if p["valid_format"]),
                "num_vehicles": 0,
                "vehicle_counts": {},
                "vehicles": [],
                "elapsed_seconds": elapsed,
                "inference_ms_yolo": out["inference_ms_yolo"],
                "inference_ms_ocr_total": out["inference_ms_ocr_total"],
                "plates": plates,
                "annotated_url": annotated_url,
                "engine": {
                    "detector": "YOLO11 (yolo11_plate.pt)",
                    "ocr": "Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)",
                    "device": "cpu",
                },
            }
            return jsonify(response)
        finally:
            try: os.unlink(path)
            except OSError: pass
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@image_bp.get("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

@image_bp.get("/health")
def health():
    return jsonify(
        status="ok",
        yolo_loaded=engine.yolo is not None,
        awiros_loaded=engine.awiros is not None,
        report_exists=REPORT_PATH.exists(),
    )

@image_bp.get("/report")
def report():
    if REPORT_PATH.exists():
        return redirect(f"/static-report/report.html", code=302)
    return jsonify(error="Report not built yet. Run build_report.py first."), 404

@image_bp.get("/static-report/<path:filename>")
def static_report(filename):
    return send_from_directory(REPORT_PATH.parent, filename)
