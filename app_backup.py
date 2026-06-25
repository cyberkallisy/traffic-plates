"""
traffic-plates Flask web app — YOLO 11 plate detection + Awiros ANPR-OCR.

Routes:
  GET  /                  single-page UI (drag/drop upload, dark theme)
  POST /api/predict       upload one image, run detection + OCR, return JSON
                          with annotated image (base64) + per-plate OCR text
  GET  /report            link to the existing HTML comparison report
  GET  /health            liveness probe

Usage:
    python app.py            # serves on http://127.0.0.1:8766
    python app.py --port N   # custom port
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, redirect

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

YOLO_MODEL = HERE / "yolo11_plate.pt"
AWIROS_DIR = HERE / "awiros_anpr"
REPORT_PATH = HERE / "test" / "New folder" / "images" / "yolo11_awiros_ocr_run" / "report.html"
UPLOAD_DIR = HERE / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(HERE / "templates"),
    static_folder=str(HERE / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload cap (videos)


# ── Engine: load models ONCE at startup ──────────────────────────────────────
class ANPREngine:
    """Lazy-loads YOLO11 + Awiros ANPR-OCR the first time it's used."""

    def __init__(self):
        self.yolo = None
        self.awiros = None

    def _ensure_yolo(self):
        if self.yolo is None:
            print(f"[APP] Loading YOLO11 model: {YOLO_MODEL.name}")
            from ultralytics import YOLO
            self.yolo = YOLO(str(YOLO_MODEL))
            print(f"[APP] YOLO11 loaded | classes: {self.yolo.names}")

    def _ensure_awiros(self):
        if self.awiros is None:
            print(f"[APP] Loading Awiros ANPR-OCR...")
            # Import the wrapper class from the pipeline script.
            sys.path.insert(0, str(HERE))
            from detect_yolo11_awiros_ocr import AwirosANPR
            self.awiros = AwirosANPR(awiros_dir=AWIROS_DIR, device="cpu")
            self.awiros.load()
            print(f"[APP] Awiros loaded | dict: {self.awiros.dict_path.name}")

    def warmup(self):
        """Force both models to load so the first request is fast."""
        self._ensure_yolo()
        self._ensure_awiros()

    def predict(self, img_bgr: np.ndarray, conf: float = 0.25) -> dict:
        """Run YOLO11 + Awiros on one image. Returns dict with detections."""
        self._ensure_yolo()
        self._ensure_awiros()

        h, w = img_bgr.shape[:2]

        # 1. YOLO plate detection
        t0 = time.time()
        result = self.yolo.predict(
            img_bgr, conf=conf, iou=0.45, imgsz=640, verbose=False
        )[0]
        dt_yolo = time.time() - t0

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                c = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                detections.append({
                    "bbox_xyxy": xyxy,
                    "confidence": round(c, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # 2. Crop + Awiros OCR per detection
        crops_bgr = []
        t_ocr_total = 0.0
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            crop = img_bgr[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                det["ocr"] = {"text": "", "confidence": 0.0, "readable": False,
                              "engine": "Awiros ANPR-OCR", "inference_ms": 0}
                crops_bgr.append(None)
                continue
            crops_bgr.append(crop)
            t1 = time.time()
            ocr = self.awiros.predict_crop(crop)
            dt_ocr = time.time() - t1
            t_ocr_total += dt_ocr
            det["ocr"] = {
                "text": ocr["text"],
                "confidence": ocr["confidence"],
                "readable": ocr["readable"],
                "engine": "Awiros ANPR-OCR",
                "inference_ms": round(dt_ocr * 1000, 1),
            }

        # 3. Draw annotated image (reused helper from pipeline script)
        from detect_yolo11_awiros_ocr import draw_annotated
        annotated = draw_annotated(img_bgr, detections)

        return {
            "size": [w, h],
            "inference_ms_yolo": round(dt_yolo * 1000, 1),
            "inference_ms_ocr_total": round(t_ocr_total * 1000, 1),
            "num_plates": len(detections),
            "num_readable": sum(1 for d in detections if d["ocr"]["readable"]),
            "detections": detections,
            "annotated_jpeg_b64": _img_to_b64(annotated, ".jpg"),
            "crops_jpeg_b64": [_img_to_b64(c, ".jpg") if c is not None else ""
                               for c in crops_bgr],
        }


engine = ANPREngine()


# ── helpers ──────────────────────────────────────────────────────────────────
def _img_to_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode OpenCV BGR image to a base64 data-URI."""
    if img is None or img.size == 0:
        return ""
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def _read_image(path: str) -> np.ndarray:
    """Unicode-safe image read for Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def _decode_b64(data_uri: str) -> np.ndarray:
    """Decode a base64 image (data URI or raw) back to a BGR numpy array."""
    if not data_uri:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _save_upload(file_storage) -> str:
    suffix = Path(file_storage.filename or "").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False)
    file_storage.save(tmp.name)
    tmp.close()
    return tmp.name


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return render_template(
        "index.html",
        yolo_model=YOLO_MODEL.name,
        report_url=str(REPORT_PATH) if REPORT_PATH.exists() else "",
    )


RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_result_image(stem: str, img_bgr) -> str:
    """Save annotated JPG to results/ and return its URL path."""
    fname = f"{stem}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(RESULTS_DIR / fname), img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"/results/{fname}"


def _save_crop_image(stem: str, img_bgr) -> str:
    fname = f"{stem}_crop_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(RESULTS_DIR / fname), img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"/results/{fname}"


@app.post("/api/predict")
def api_predict():
    """Legacy route (kept). Frontend uses /api/detect below."""
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
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.post("/api/detect")
def api_detect():
    """Run YOLO11 plate detection + Awiros ANPR-OCR on one uploaded image.

    Form fields:
        image      — uploaded file (required)
        conf       — YOLO confidence threshold (default 0.25)

    Returns JSON shaped for templates/index.html (renderImageResults):
        num_plates, num_valid, num_vehicles, elapsed_seconds
        plates: [{text, valid_format, detection_confidence, ocr_confidence,
                  crop_url, bbox_xyxy, ...}]
        annotated_url
    """
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

            # Shape for the existing frontend (renderImageResults)
            plates = []
            for i, det in enumerate(out["detections"], 1):
                ocr = det["ocr"]
                text = ocr.get("text", "") or ""
                ocr_conf = ocr.get("confidence", 0.0)
                valid_format = bool(ocr.get("readable")) and len(text) >= 4
                # Build crop URL from the per-detection crop (re-extract & save)
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
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.post("/api/detect_video")
def api_detect_video():
    """Run YOLO11 plate detection + Awiros ANPR-OCR on each sampled frame,
    track plates by IoU across frames, then per-position voting across the
    track's OCR reads -> one final plate text per unique plate.

    Form fields:
        video        — uploaded video file (required)
        frame_stride — int, process every Nth frame (default 2)
        max_frames   — int, stop after this many raw frames (default unlimited)
        write_video  — "1" to write annotated.mp4, "0" to skip (default 1)
        iou          — tracker IoU threshold (default 0.20)

    Returns JSON shaped for templates/index.html (renderVideoResults):
        n_tracks, n_valid_plates, n_frames_processed, n_total_frames,
        elapsed_seconds, tracks: [...], annotated_video_url, report_url
    """
    try:
        f = request.files.get("video")
        if f is None or not f.filename:
            return jsonify(error="No video uploaded. Send a file in the 'video' field."), 400

        stride = max(1, int(request.form.get("frame_stride", 2)))
        max_frames_raw = request.form.get("max_frames", "")
        max_frames = int(max_frames_raw) if max_frames_raw.strip() else None
        write_video = request.form.get("write_video", "1") != "0"
        iou = float(request.form.get("iou", 0.20))

        # Save upload to a videos/ folder so the path is stable for the response.
        VIDEOS_DIR = HERE / "uploads_videos"
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(f.filename).suffix or ".mp4"
        ts_ms = int(time.time() * 1000)
        stem = Path(f.filename).stem
        video_name = f"{stem}_{ts_ms}{suffix}"
        video_path = VIDEOS_DIR / video_name
        f.save(str(video_path))

        # Output dir (one per upload — easy to find from the response)
        out_dir = HERE / "video_results" / f"{stem}_awiros_{ts_ms}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run the pipeline synchronously. Long videos block — that's expected.
        from anpr_video_awiros import process_video
        t0 = time.time()
        summary = process_video(
            video_path=video_path,
            out_dir=out_dir,
            stride=stride,
            max_frames=max_frames,
            write_video=write_video,
            awiros_dir=AWIROS_DIR,
            yolo_model=str(YOLO_MODEL),
            device="cpu",
            iou_thresh=iou,
        )
        elapsed = round(time.time() - t0, 2)

        # Build response in the shape renderVideoResults expects
        annotated_video_url = ""
        if write_video and (out_dir / "annotated.mp4").exists():
            annotated_video_url = f"/video-results/{out_dir.relative_to(HERE / 'video_results')}/annotated.mp4"
        report_url = f"/video-results/{out_dir.relative_to(HERE / 'video_results')}/report.html"

        # Compact track view for the frontend
        tracks_payload = []
        for t in summary["tracks"]:
            tracks_payload.append({
                "track_id": t["track_id"],
                "final_text": t["final_text"],
                "final_conf": t["final_conf"],
                "valid_indian": t["valid_indian"],
                "n_frames": t["n_frames"],
                "first_seen": t["first_seen"],
                "last_seen": t["last_seen"],
                "avg_yolo_conf": t["avg_yolo_conf"],
                "n_unique_reads": t["n_unique_reads"],
                "votes_per_pos": t["votes_per_pos"],
                # New: best-frame for clickable thumbnail
                "best_frame": t.get("best_frame"),
                "best_text": t.get("best_text", ""),
                "best_conf": t.get("best_conf", 0.0),
                "best_crop_url": (f"/video-results/{out_dir.relative_to(HERE / 'video_results')}/best_frames/{t['best_crop_file']}"
                                  if t.get("best_crop_file") else ""),
                # Per-frame reads (for the modal that opens on click)
                "all_frames": t.get("all_frames", []),
                "per_frame_reads": t.get("per_frame_reads", []),
                "crop_url": "",  # legacy
            })

        return jsonify({
            "filename": f.filename,
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_tracks": summary["n_tracks"],
            "n_valid_plates": summary["n_valid_plates"],
            "n_frames_processed": summary["n_frames_processed"],
            "n_total_frames": summary["n_total_frames"],
            "fps": summary["fps"],
            "stride": stride,
            "tracker": summary["tracker"],
            "elapsed_seconds": elapsed,
            "fps_processed": summary["fps_processed"],
            "tracks": tracks_payload,
            "annotated_video_url": annotated_video_url,
            "report_url": report_url,
            "engine": {
                "detector": "YOLO11 (yolo11_plate.pt)",
                "ocr": "Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)",
                "device": "cpu",
                "voting": "per-character position voting across the track's OCR reads",
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.get("/video-results/<path:relpath>")
def serve_video_result(relpath):
    """Serve annotated.mp4, frames/, crops/, report.html from video_results/<relpath>."""
    from flask import send_from_directory
    base = HERE / "video_results"
    # Security: resolve and ensure the final path is under base
    target = (base / relpath).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        return jsonify(error="Invalid path."), 400
    if target.is_dir():
        # Directory listing not supported — return 404 to avoid leaking paths.
        return jsonify(error="Directory listing disabled."), 404
    if not target.exists():
        return jsonify(error="Not found."), 404
    return send_from_directory(target.parent, target.name, as_attachment=False)


@app.get("/api/track_details/<path:relpath>")
def api_track_details(relpath):
    """Return per-frame OCR details for ONE track from a video result dir.

    Path: <video_dir>/track_<id>
    Reads tracks.json inside <video_results>/<video_dir>/ and returns only the
    matching track (full per_frame_reads, best_frame, all_frames, votes_per_pos).
    """
    base = HERE / "video_results"
    target_dir = (base / relpath).resolve()
    try:
        target_dir.relative_to(base.resolve())
    except ValueError:
        return jsonify(error="Invalid path."), 400
    if not target_dir.is_dir():
        return jsonify(error="Not a video result dir."), 404

    # relpath ends with /track_<id>
    track_id_raw = target_dir.name
    if not track_id_raw.startswith("track_"):
        return jsonify(error="Path must end with /track_<id>."), 400
    try:
        wanted_id = int(track_id_raw.split("_", 1)[1])
    except ValueError:
        return jsonify(error="Track id must be an integer."), 400

    # The actual video dir is the parent of target_dir
    video_dir = target_dir.parent
    summary_json = video_dir / "summary.json"
    tracks_json  = video_dir / "tracks.json"

    # Pick the most informative source — tracks.json has the rich per-frame data
    # written by anpr_video_awiros.process_video(); summary.json is a slim subset.
    track = None
    for jpath in (tracks_json, summary_json):
        if not jpath.exists():
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        tracks_list = data.get("tracks") or []
        for t in tracks_list:
            if int(t.get("track_id", -1)) == wanted_id:
                track = t
                break
        if track is not None:
            break

    if track is None:
        return jsonify(error=f"Track {wanted_id} not found."), 404

    # Build absolute URL prefix for crop files (frontend uses these in <img src>)
    rel_to_base = str(video_dir.relative_to(base))
    crop_url_prefix = f"/video-results/{rel_to_base}/crops/"
    best_crop_url = ""
    if track.get("best_crop_file"):
        best_crop_url = f"/video-results/{rel_to_base}/best_frames/{track['best_crop_file']}"

    # Augment per_frame_reads with full crop_url (one crop per frame)
    for r in track.get("per_frame_reads", []):
        if r.get("crop_file"):
            r["crop_url"] = crop_url_prefix + r["crop_file"]
    track["best_crop_url"] = best_crop_url
    return jsonify(track)


@app.get("/results/<path:filename>")
def serve_result(filename):
    """Serve annotated images / crops saved by /api/detect."""
    from flask import send_from_directory
    return send_from_directory(RESULTS_DIR, filename)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        yolo_loaded=engine.yolo is not None,
        awiros_loaded=engine.awiros is not None,
        report_exists=REPORT_PATH.exists(),
    )


@app.get("/report")
def report():
    """Redirect to the existing static HTML report (or 404 if absent)."""
    if REPORT_PATH.exists():
        return redirect(f"/static-report/report.html", code=302)
    return jsonify(error="Report not built yet. Run build_report.py first."), 404


# Serve the report folder at /static-report/<path:filename>
@app.get("/static-report/<path:filename>")
def static_report(filename):
    from flask import send_from_directory
    return send_from_directory(REPORT_PATH.parent, filename)


# ── Entry ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="traffic-plates ANPR web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip loading models at startup (loads on first request)")
    args = p.parse_args()

    print("=" * 60)
    print(f"traffic-plates ANPR web app")
    print(f"  YOLO11 model : {YOLO_MODEL}")
    print(f"  Awiros OCR   : {AWIROS_DIR / 'model.safetensors'}")
    print(f"  HTML report  : {REPORT_PATH}")
    print(f"  Listening on : http://{args.host}:{args.port}")
    print("=" * 60)

    if not args.no_warmup:
        engine.warmup()

    # use_reloader=False: required — models live in this process, no fork
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()