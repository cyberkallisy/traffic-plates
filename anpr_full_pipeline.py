"""
ANPR Full Pipeline:
    Input Image
        -> Real-ESRGAN (x2 super-resolution)
        -> YOLO11 License Plate Detection
        -> Crop Plate
        -> CLAHE + Denoising (bilateral + CLAHE + adaptive threshold)
        -> PaddleOCR (English, angle-cls)
        -> Indian Number Plate Regex Validation + State-code check

Usage:
    python anpr_full_pipeline.py

Outputs (one timestamped folder):
    <out>/annotated/      — every input image with plate bboxes + OCR labels
    <out>/crops/          — original plate crops (raw from YOLO)
    <out>/crops_esrgan/   — ESRGAN-upscaled plate crops
    <out>/crops_preproc/  — CLAHE + denoised crops fed to PaddleOCR
    <out>/results.json    — full structured output (per image, per plate)
    <out>/results.csv     — flat summary (image, plate_text, confidence, valid)
    <out>/report.html     — side-by-side visual report

Notes
-----
* Real-ESRGAN is loaded only if its weights file is present on disk. If
  missing (e.g. first run before download finishes) the pipeline falls back
  to the raw image — YOLO is happy with that anyway.
* Regex validation = (1) format matches Indian standard, (2) RTO state code
  is one of the 38 valid codes (AN..WB + BH for Bharat-series).
* Designed for the Facial-recognition venv (paddleocr + ultralytics +
  opencv all together). Avoids `python -c` invocations to dodge the
  Windows shm.dll DLL-load-order bug.
"""

import os
# PaddleOCR (paddlepaddle 2.6.x) requires the pure-python protobuf backend
# on this Windows venv — see memory. MUST be set before paddle imports.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import csv
import json
import re
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR   = Path(r"C:/Users/gsash/Downloads/test/New folder/images")
OUT_ROOT    = Path(r"C:/Users/gsash/Downloads/test/New folder")
YOLO11_PT   = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt")
ESRGAN_PTH  = Path(r"C:/Users/gsash/Downloads/Facial-recognition/models/RealESRGAN_x4plus_anime_6B.pth")

CONF_THR       = 0.25     # YOLO plate-detection confidence
OCR_CONF_THR   = 0.30     # PaddleOCR per-line confidence
ALLOWED_EXT    = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
ESRGAN_SCALE   = 2        # output scale (matches x2plus weights)
MIN_CROP_SIDE  = 80       # crops smaller than this get upscaled before ESRGAN
ESRGAN_TILE    = 0        # 0 = no tiling; 256/512 useful for very large crops

# Indian-plate regexes + RTO state-code whitelist.
#
# Indian plates come in two main shapes:
#   Standard (post-2005, "high-security"):
#       SS DD [L|LL] N|NN|NNN|NNNN   e.g. MH 01 AB 1234, KA 05 J 9999
#   Older / commercial / legacy:
#       SS D|DD N|NNN|NNNN           e.g. HR 6705431 (state + numeric only)
#   Bharat-series (BH):
#       21BH1234AA                    e.g. 22BH1234AB
#
# We keep a few patterns and pick the one that fits best. The RTO state-code
# check (text[:2] in INDIAN_VALID_STATES) is what makes us Indian-specific.
INDIAN_PATTERNS = [
    # standard
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"),
    # legacy all-numeric (state + district + 4-digit number)
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[0-9]{4,5}$"),
    # compact: state + district + 4-digit number (no letters in middle)
    re.compile(r"^[A-Z]{2}[0-9]{6,7}$"),
    # Bharat-series
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
]
INDIAN_VALID_STATES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HR","HP",
    "JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TS","TR","UP","UK","WB","BH",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("anpr-full")


# ---------------------------------------------------------------------------
# Real-ESRGAN loader (optional — skipped gracefully if weights missing)
# ---------------------------------------------------------------------------
_ESRGAN = None
def _load_esrgan():
    """Lazy-load Real-ESRGAN. Returns the upsampler or None."""
    global _ESRGAN
    if _ESRGAN is not None:
        return _ESRGAN
    if not ESRGAN_PTH.exists():
        log.warning("Real-ESRGAN weights not found at %s — running without ESRGAN", ESRGAN_PTH)
        return None
    # If download is partial, refuse to load (basicsr will read garbage keys)
    size = ESRGAN_PTH.stat().st_size
    # x4plus_anime_6B is ~17.5 MB; x2plus is ~53 MB. Both pass the 8 MB gate.
    if size < 8_000_000:
        log.warning("Real-ESRGAN weights only %.1f MB — skipping ESRGAN",
                    size / 1e6)
        return None
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        log.info("Loading Real-ESRGAN from %s ...", ESRGAN_PTH)
        t0 = time.time()
        # Anime-6B is x4 scale with 6 RRDB blocks — tuned for flat-color
        # anime-style art, but works very well on high-contrast text on
        # solid background (i.e. license plates).
        rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=6, num_grow_ch=32, scale=4)
        _ESRGAN = RealESRGANer(
            scale=4,
            model_path=str(ESRGAN_PTH),
            model=rrdb,
            tile=ESRGAN_TILE,
            tile_pad=10,
            pre_pad=0,
            half=False,            # CPU
            device="cpu",
        )
        log.info("Real-ESRGAN (anime-6B, x4) loaded in %.1fs", time.time() - t0)
        return _ESRGAN
    except Exception as e:
        log.error("Real-ESRGAN load failed: %s — running without ESRGAN", e)
        return None


def _esrgan_enhance(esrgan, crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale crop with Real-ESRGAN, returns BGR uint8."""
    if esrgan is None or crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    try:
        out, _ = esrgan.enhance(crop_bgr, outscale=ESRGAN_SCALE)
        return out
    except Exception as e:
        log.warning("Real-ESRGAN enhance failed: %s", e)
        return crop_bgr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_text(s: str) -> str:
    """Strip noise, normalise to A-Z0-9."""
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _is_indian_plate(text: str) -> bool:
    if not text or len(text) < 6:
        return False
    # BH-series is state-agnostic
    if INDIAN_PATTERNS[3].match(text):
        return True
    # All other patterns require a valid RTO state code at the front
    state = text[:2]
    if state not in INDIAN_VALID_STATES:
        return False
    return any(p.match(text) for p in INDIAN_PATTERNS[:3])


def _clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w,     int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE + Denoising pipeline.
    Steps:
      1. Resize up if tiny
      2. White-pad (PaddleOCR's text detector likes clean borders)
      3. Bilateral filter (denoise, edge-preserving)
      4. Convert to grayscale + CLAHE (contrast normalisation)
      5. Adaptive Gaussian threshold (text vs background)
      6. Re-cast to 3-channel BGR for PaddleOCR
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]

    # 1. Up-tiny crops so OCR sees enough pixels
    if max(h, w) < 120:
        s = 120 / max(h, 1)
        crop_bgr = cv2.resize(crop_bgr, (int(w * s), int(h * s)),
                              interpolation=cv2.INTER_CUBIC)
        h, w = crop_bgr.shape[:2]

    # 2. White-pad
    pad = 20
    crop_bgr = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # 3. Bilateral denoise (preserves edges unlike Gaussian blur)
    denoised = cv2.bilateralFilter(crop_bgr, 9, 75, 75)

    # 4. CLAHE on grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 5. Adaptive threshold
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)

    # 6. back to BGR
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _paddle_ocr_best(ocr: PaddleOCR, crop_bgr: np.ndarray,
                     variants: dict) -> dict:
    """
    Run PaddleOCR over each variant of the plate crop (raw, ESRGAN-upscaled,
    preprocessed). Pick the best candidate by (valid-format > confidence).
    """
    candidates = []
    for label, img in variants.items():
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
        "variant_winner": best["variant"],
        "raw": candidates,
    }


def _annotate(img_bgr: np.ndarray, plates: list) -> np.ndarray:
    """Draw plate bboxes + OCR text on a copy of the image."""
    out = img_bgr.copy()
    for i, p in enumerate(plates, 1):
        x1, y1, x2, y2 = p["bbox"]
        text = p["ocr"]["text"] or "(no text)"
        conf = p["ocr"]["conf"]
        valid = p["ocr"]["valid_format"]
        color = (0, 200, 0) if valid else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        tag = "OK" if valid else "?"
        label = f"#{i} [{tag}] {text} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - th - 8, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = OUT_ROOT / f"anpr_pipeline_{ts}"
    annot_dir  = out_dir / "annotated"
    crops_dir  = out_dir / "crops"
    crops_esr  = out_dir / "crops_esrgan"
    crops_pp   = out_dir / "crops_preproc"
    annot_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops_esr.mkdir(parents=True, exist_ok=True)
    crops_pp.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ANPR Full Pipeline starting")
    log.info("  input  : %s", INPUT_DIR)
    log.info("  output : %s", out_dir)
    log.info("=" * 60)

    # Load models
    log.info("Loading YOLO11 from %s", YOLO11_PT)
    t0 = time.time()
    yolo = YOLO(str(YOLO11_PT))
    names = yolo.names
    log.info("YOLO11 loaded in %.1fs. Classes: %s", time.time() - t0, names)

    log.info("Loading PaddleOCR (en, mobile, angle-cls) ...")
    t0 = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False,
                    use_gpu=False, det_db_thresh=0.3, det_db_box_thresh=0.5)
    log.info("PaddleOCR ready in %.1fs", time.time() - t0)

    esrgan = _load_esrgan()
    if esrgan is None:
        log.warning("Pipeline running WITHOUT Real-ESRGAN (using raw crops).")

    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT)
    log.info("Found %d images in %s", len(images), INPUT_DIR)

    summary = {
        "run_id": f"anpr_pipeline_{ts}",
        "input_dir": str(INPUT_DIR),
        "output_dir": str(out_dir),
        "pipeline": ["Real-ESRGAN (x2)", "YOLO11 plate detection",
                     "Crop", "CLAHE + bilateral denoise",
                     "PaddleOCR (en, mobile, angle-cls)",
                     "Indian number plate regex validation"],
        "models": {
            "yolo11": str(YOLO11_PT),
            "paddleocr": "2.7 (en, mobile, angle-cls, CPU)",
            "esrgan": str(ESRGAN_PTH) if esrgan else "(disabled — weights missing)",
        },
        "thresholds": {
            "yolo_conf": CONF_THR,
            "ocr_conf": OCR_CONF_THR,
        },
        "stats": {
            "images_processed": 0,
            "images_with_plates": 0,
            "plates_total": 0,
            "plates_valid_format": 0,
        },
        "results": [],
    }

    for idx, img_path in enumerate(images, 1):
        log.info("[%d/%d] %s", idx, len(images), img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("  could not read, skipping")
            continue
        h, w = img.shape[:2]

        # --- YOLO11 plate detection (on raw image — ESRGAN is for crops) ---
        t0 = time.time()
        det = yolo.predict(img, conf=CONF_THR, verbose=False)[0]
        yolo_ms = (time.time() - t0) * 1000

        plates = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()
            clses = det.boxes.cls.cpu().numpy().astype(int)
            for box, c, cls in zip(xyxy, confs, clses):
                clamped = _clamp_box(box, w, h)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                crop_raw = img[y1:y2, x1:x2].copy()

                stem = img_path.stem
                plate_idx = len(plates) + 1
                crop_name = f"{stem}_p{plate_idx}_{c:.2f}.jpg"

                # 1) save raw crop
                cv2.imwrite(str(crops_dir / crop_name), crop_raw)

                # 2) ESRGAN upscale (best-effort; skipped if weights missing)
                t1 = time.time()
                crop_esrgan = _esrgan_enhance(esrgan, crop_raw)
                esrgan_ms = (time.time() - t1) * 1000
                if crop_esrgan is not crop_raw:
                    cv2.imwrite(str(crops_esr / crop_name), crop_esrgan)

                # 3) preprocess for OCR (CLAHE + denoise) — on ESRGAN crop
                #    if we have one (more pixels = better adaptive threshold),
                #    otherwise on raw crop.
                ocr_input = crop_esrgan if crop_esrgan is not crop_raw else crop_raw
                t2 = time.time()
                crop_pp = _preprocess_for_ocr(ocr_input)
                preproc_ms = (time.time() - t2) * 1000
                cv2.imwrite(str(crops_pp / crop_name), crop_pp)

                # 4) PaddleOCR over all variants
                variants = {
                    "raw":     crop_raw,
                    "esrgan":  crop_esrgan if crop_esrgan is not crop_raw else None,
                    "preproc": crop_pp,
                }
                t3 = time.time()
                ocr_res = _paddle_ocr_best(ocr, crop_raw, variants)
                ocr_ms = (time.time() - t3) * 1000

                plates.append({
                    "plate_idx": plate_idx,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_xyxy_orig": [int(v) for v in box],
                    "yolo_conf": round(float(c), 3),
                    "class_id": int(cls),
                    "class_name": names.get(int(cls), str(cls)),
                    "ocr": ocr_res,
                    "crop_files": {
                        "raw":     crop_name,
                        "esrgan":  crop_name if crop_esrgan is not crop_raw else None,
                        "preproc": crop_name,
                    },
                    "timing_ms": {
                        "yolo":       round(yolo_ms, 1),
                        "esrgan":     round(esrgan_ms, 1) if esrgan else 0.0,
                        "preprocess": round(preproc_ms, 1),
                        "paddleocr":  round(ocr_ms, 1),
                    },
                })

        annotated = _annotate(img, plates)
        annot_name = f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(annot_dir / annot_name), annotated)

        n_plates = len(plates)
        n_valid  = sum(1 for p in plates if p["ocr"]["valid_format"])
        summary["stats"]["images_processed"] += 1
        if n_plates > 0:
            summary["stats"]["images_with_plates"] += 1
        summary["stats"]["plates_total"]       += n_plates
        summary["stats"]["plates_valid_format"] += n_valid

        summary["results"].append({
            "image": img_path.name,
            "image_size": [w, h],
            "num_plates": n_plates,
            "num_valid_format": n_valid,
            "plates": plates,
            "annotated_file": annot_name,
        })
        # one-liner on stdout
        txts = [p["ocr"]["text"] for p in plates if p["ocr"]["text"]]
        log.info("  -> %d plate(s) | OCR: %s | valid=%d",
                 n_plates, txts or "[]", n_valid)

    # --- Persist structured summary ---
    results_json = out_dir / "results.json"
    results_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # --- Flat CSV ---
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["image", "plate_idx", "yolo_conf",
                        "plate_text", "ocr_conf", "valid_format",
                        "winner_variant", "bbox"])
        for r in summary["results"]:
            for p in r["plates"]:
                w_csv.writerow([
                    r["image"], p["plate_idx"], p["yolo_conf"],
                    p["ocr"]["text"], p["ocr"]["conf"],
                    p["ocr"]["valid_format"],
                    p["ocr"].get("variant_winner", ""),
                    " ".join(str(v) for v in p["bbox"]),
                ])

    # --- Visual HTML report ---
    _write_html_report(out_dir, summary)

    log.info("=" * 60)
    s = summary["stats"]
    log.info("Done. %d images, %d had plates, %d plates total, %d valid Indian format.",
             s["images_processed"], s["images_with_plates"],
             s["plates_total"], s["plates_valid_format"])
    log.info("Summary:   %s", results_json)
    log.info("CSV:       %s", csv_path)
    log.info("Annotated: %s", annot_dir)
    log.info("Crops raw: %s", crops_dir)
    log.info("Crops ESRGAN: %s", crops_esr)
    log.info("Crops preproc: %s", crops_pp)
    log.info("HTML report: %s", out_dir / "report.html")


def _write_html_report(out_dir: Path, summary: dict):
    """Side-by-side report: image | annotated | crops (raw, ESRGAN, preproc)."""
    rows = []
    for r in summary["results"]:
        plates_html = []
        for p in r["plates"]:
            ok = "OK" if p["ocr"]["valid_format"] else "?"
            text = p["ocr"]["text"] or "(no text)"
            conf = p["ocr"]["conf"]
            variant = p["ocr"].get("variant_winner", "-")
            cf = p["crop_files"]
            esrgan_img = f"crops_esrgan/{cf['esrgan']}" if cf.get("esrgan") else ""
            plates_html.append(f"""
            <tr>
              <td>#{p['plate_idx']} [{ok}] {text} ({conf:.2f})</td>
              <td>yolo={p['yolo_conf']:.2f} winner={variant}</td>
              <td><img src="crops/{cf['raw']}" height="80"/></td>
              <td>{"<img src='" + esrgan_img + "' height='80'/>" if esrgan_img else "(no ESRGAN)"}</td>
              <td><img src="crops_preproc/{cf['preproc']}" height="80"/></td>
            </tr>""")
        rows.append(f"""
        <div class="image-block">
          <h3>{r['image']} — {r['num_plates']} plate(s), {r['num_valid_format']} valid</h3>
          <div class="annotated">
            <img src="annotated/{r['annotated_file']}" style="max-width:100%;max-height:400px;"/>
          </div>
          <table>
            <thead><tr><th>Plate</th><th>Stats</th><th>Raw crop</th><th>ESRGAN crop</th><th>Preproc crop</th></tr></thead>
            <tbody>{''.join(plates_html) if plates_html else '<tr><td colspan="5">(no plates detected)</td></tr>'}</tbody>
          </table>
        </div>""")

    s = summary["stats"]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>ANPR pipeline — {summary['run_id']}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .stats {{ background: #fff; padding: 12px; border-radius: 6px; margin: 10px 0; }}
  .image-block {{ background: #fff; padding: 15px; margin: 15px 0; border-radius: 6px; }}
  .annotated img {{ border: 1px solid #ccc; }}
  table {{ border-collapse: collapse; margin-top: 8px; }}
  th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; font-size: 14px; }}
  th {{ background: #efefef; }}
</style>
</head><body>
<h1>ANPR Full Pipeline — {summary['run_id']}</h1>
<div class="stats">
  <b>Stats:</b> {s['images_processed']} images processed,
  {s['images_with_plates']} with plates,
  {s['plates_total']} plates total,
  {s['plates_valid_format']} matching Indian format.
  <br><b>Models:</b>
  YOLO11={summary['models']['yolo11']},
  PaddleOCR={summary['models']['paddleocr']},
  ESRGAN={summary['models']['esrgan']}
</div>
{''.join(rows)}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()