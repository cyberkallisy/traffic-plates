"""
Traffic Photo Number Plate + Vehicle Detector
==============================================
Two-model pipeline running on every image:

1. **License-plate detector** — YOLOv10n (Rawzy license-plate-detection on
   HuggingFace). Detects plate ROIs, sends each crop to PaddleOCR (with
   Tesseract fallback), then applies position-aware correction tuned for
   Indian vehicle plates.

2. **Vehicle detector** — COCO-pretrained YOLOv8n (the standard 80-class
   model). We keep ONLY the 5 vehicle classes we care about for traffic
   photos: `bicycle` (1), `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
   Out-of-the-box, no fine-tune, ~6.5 MB weights.

Both models annotate the same returned image — plates in green/orange,
vehicles in class-specific colors.

OCR engine: PaddleOCR 2.7 (paddlepaddle 2.6.x). PaddleOCR tends to give
much higher raw accuracy than EasyOCR on number-plate text, so the
position-aware correction step is mostly a safety net now.

Detector (plates): YOLOv10n — Tsinghua's YOLOv10 (May 2024) nano variant,
trained on license plates by Rawzy on HuggingFace. Same ultralytics API
as v8; just a newer / faster / better architecture (NMS-free end-to-end).

Detector (vehicles): YOLOv8n COCO-pretrained (ultralytics default).
Filtered to the 5 vehicle classes via the `classes=` argument at
inference time — no post-filter needed.

Usage:
    from detector import TrafficPlateDetector
    det = TrafficPlateDetector()
    result = det.detect("path/to/image.jpg")
    # result = {
    #   "annotated_image": "results/abc123.jpg",
    #   "plates":   [ {text, confidence, bbox, valid_format, ...}, ... ],
    #   "vehicles": [ {class_name, class_id, confidence, bbox}, ... ],
    #   "vehicle_counts": {"car": 3, "truck": 1, "bus": 0, ...},
    #   "num_plates": 1, "num_vehicles": 4,
    #   "image_size": [1920, 1080]
    # }
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy imports for heavy ML libs (so the server can boot even if model load fails)
# ---------------------------------------------------------------------------
_yolo_model = None
_paddleocr_engine = None
_tesseract_available = None

logger = logging.getLogger("traffic-plates")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Indian number plate patterns
# ---------------------------------------------------------------------------
# Standard Indian format: SS DD[N] LL[N] NNN[N]   e.g. HR 26 DK 1234
# State (2 letters) + District (1-2 digits) + Series (1-2 letters) + Number (3-4 digits)
# Indian plate formats — based on actual RTO registrations
# Standard: SS DD[N] LL[N] NNN[N]  e.g. HR 26 DK 1234, DL 8 CAF 5032
# Series can be 1-3 letters (e.g. "CAF" in DL8CAF5032), district can be 1-2 digits
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
# BH series (Bharat): NN BH NNNN AA   e.g. 22 BH 1234 AB
INDIAN_BH_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
# Diplomatic / temporary plates
INDIAN_DIPLOMATIC_RE = re.compile(r"^[0-9]{1,3}[A-Z]{1,3}[0-9]{1,4}$")

VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UP", "UK",
    "WB", "BH"
}

# ---------------------------------------------------------------------------
# Position-aware OCR error correction
# ---------------------------------------------------------------------------
# When char in a LETTER slot, looks like digit -> swap to closest letter(s)
# Primary swap (most likely). Secondary swaps are tried as fallbacks by
# `correct_position_aware` when the primary doesn't validate as a real plate.
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P",
}
DIGIT_TO_LETTER_ALT = {
    # alternative readings used to generate candidates
    "0": ["D", "B", "Q"],   # 0 commonly misread as D / B / Q
    "1": ["L"],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
}
# When char in a DIGIT slot, looks like letter -> swap to closest digit
LETTER_TO_DIGIT = {
    "O": "0", "I": "1", "Z": "2", "E": "3", "A": "4",
    "S": "5", "G": "6", "T": "7", "B": "8", "P": "9",
    "D": "0",  # OCR commonly mis-reads 0 as D
    "Q": "0",  # Q looks like O/0
    "L": "1",  # L looks like 1
}


def _classify_slot(clean: str) -> List[str]:
    """Return the expected char-class for each position in a clean plate string.
    Returns one of: 'L' (letter), 'D' (digit), '?' (unknown).

    Standard Indian plate:
        Length 6  -> L L D L DDD        (e.g. HR 1 A 001)
        Length 7  -> L L D L DDDD       (e.g. HR 1 A 0001)
        Length 8  -> L L D LL DDD       (e.g. HR 26 DK 123)
        Length 9  -> L L DD LL DDD      (e.g. HR 26 DK 123)
        Length 10 -> L L DD LL DDDD     (e.g. HR 26 DK 1234)

    BH-series plate (Bharat):
        Length 8  -> D D L L DDDD       (e.g. 22 BH 1234)
        Length 9  -> D D L L DDDD L     (e.g. 22 BH 1234 A)
        Length 10 -> D D L L DDDD LL    (e.g. 22 BH 1234 AB)
    """
    slots = []
    n = len(clean)
    if n < 6 or n > 10:
        return ["?"] * n
    # Detect BH-series: positions 2-3 == "BH"
    is_bh = n >= 8 and clean[2:4] == "BH"
    if is_bh:
        # D D L L D... then L for last 0-2 chars
        slots += ["D", "D", "L", "L"]
        # Number: at least 4 digits
        num_slots = max(4, n - 6)  # leave up to 2 trailing letters
        num_slots = min(num_slots, n - 4)  # can't exceed remaining
        slots += ["D"] * num_slots
        # Series suffix: 0-2 letters
        while len(slots) < n:
            slots.append("L")
        return slots
    # State code: 2 letters
    slots += ["L", "L"]
    # District: 1 digit if total<=7, 2 digits if total>=8
    if n >= 8:
        slots += ["D", "D"]
    else:
        slots += ["D"]
    # Series: 1 letter if total<=7, 2-3 letters if total>=8/10
    # (e.g. "DL 8 CAF 5032" has 3-letter series "CAF")
    if n >= 10:
        slots += ["L", "L", "L"]
    elif n >= 8:
        slots += ["L", "L"]
    else:
        slots += ["L"]
    # Number: 3-4 digits (fill rest)
    for _ in range(n - len(slots)):
        slots.append("D")
    return slots


def correct_position_aware(raw: str) -> Tuple[str, List[str]]:
    """Apply position-aware correction to a noisy plate string.
    Returns (corrected, list_of_corrections_made).

    Two-stage:
      1. Primary correction using DIGIT_TO_LETTER / LETTER_TO_DIGIT.
      2. If primary doesn't validate as an Indian plate, generate up to
         `MAX_CANDIDATES` variants by swapping ambiguous chars to their
         alternative letter/digit readings, and return the first one that
         validates. This handles real OCR cases like "0" being read for "B".
    """
    MAX_CANDIDATES = 8
    if not raw:
        return "", []
    # First strip noise
    clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not clean:
        return "", []
    slots = _classify_slot(clean)
    if len(slots) != len(clean):
        return clean, []  # don't try to fix if length is non-standard

    def _apply(primary_map: dict, char: str) -> str:
        return primary_map.get(char, char)

    def _build(mapping: dict) -> Tuple[str, List[str]]:
        out, fixes = [], []
        for ch, slot in zip(clean, slots):
            if slot == "L" and ch.isdigit():
                new = _apply(mapping, ch)
                out.append(new)
                fixes.append(f"{ch}->{new}" if new != ch else f"{ch}->{ch}")
            elif slot == "D" and ch.isalpha():
                new = LETTER_TO_DIGIT.get(ch, ch)
                out.append(new)
                fixes.append(f"{ch}->{new}" if new != ch else f"{ch}->{ch}")
            else:
                out.append(ch)
        return "".join(out), fixes

    # Stage 1: primary correction
    primary_text, primary_fixes = _build(DIGIT_TO_LETTER)
    if validate_indian_plate(primary_text)["valid"]:
        return primary_text, primary_fixes

    # Stage 2: candidate search. For each ambiguous digit-in-letter-slot
    # position, try the alternatives from DIGIT_TO_LETTER_ALT. We swap one
    # position at a time, starting from the most-ambiguous (0) and going
    # outwards, until we find a candidate that validates.
    ambiguous_positions = [
        i for i, (ch, slot) in enumerate(zip(clean, slots))
        if slot == "L" and ch.isdigit() and DIGIT_TO_LETTER_ALT.get(ch)
    ]
    for pos in ambiguous_positions:
        for alt in DIGIT_TO_LETTER_ALT.get(clean[pos], []):
            mapping = dict(DIGIT_TO_LETTER)
            mapping[clean[pos]] = alt
            candidate, cand_fixes = _build(mapping)
            if validate_indian_plate(candidate)["valid"]:
                return candidate, cand_fixes

    return primary_text, primary_fixes


def validate_indian_plate(text: str) -> Dict[str, Any]:
    """Return dict with format info for an Indian plate candidate."""
    if not text:
        return {"valid": False, "format": "empty", "state": None}
    if INDIAN_BH_RE.match(text):
        return {"valid": True, "format": "BH-series", "state": "BH"}
    if INDIAN_PLATE_RE.match(text):
        state = text[:2]
        if state in VALID_INDIAN_STATES:
            return {"valid": True, "format": "standard", "state": state}
        return {"valid": False, "format": "standard-bad-state", "state": state}
    if INDIAN_DIPLOMATIC_RE.match(text):
        return {"valid": True, "format": "diplomatic", "state": text[-3:-2] if len(text) >= 4 else None}
    return {"valid": False, "format": "unknown", "state": None}


# ---------------------------------------------------------------------------
# COCO vehicle classes (COCO-pretrained YOLOv8n)
# ---------------------------------------------------------------------------
# Standard COCO has 80 classes; we only care about the 5 vehicle types that
# appear in Indian traffic photos. YOLO accepts the `classes=` argument at
# inference time to filter at the post-NMS step, so we don't have to do
# any manual class filtering on the result.
COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
# BGR colors (OpenCV) — one per vehicle class. Distinct from the plate
# annotation colors (green/orange) so they don't get confused.
VEHICLE_COLORS = {
    "bicycle":    (200,  50, 200),  # purple
    "car":        (255, 220,  60),  # cyan-yellow
    "motorcycle": (180, 105, 255),  # pink/magenta-ish
    "bus":        ( 60, 180, 255),  # orange
    "truck":      ( 80,  80, 220),  # red
}
# Friendly emoji per class (used in the UI chip)
VEHICLE_EMOJI = {
    "bicycle":    "🚲",
    "car":        "🚗",
    "motorcycle": "🏍️",
    "bus":        "🚌",
    "truck":      "🚚",
}

# ---------------------------------------------------------------------------
# Heavy model loaders (lazy)
# ---------------------------------------------------------------------------
def _get_yolo(model_path: str):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO (plate detector) from {model_path} ...")
        t0 = time.time()
        _yolo_model = YOLO(model_path)
        logger.info(f"YOLO plate detector loaded in {time.time() - t0:.1f}s")
    return _yolo_model


_yolo_vehicle_model = None


def _get_yolo_coco(model_path: str):
    """Lazy-load the COCO-pretrained YOLOv8n for vehicle detection.
    Separate singleton from `_get_yolo` so the two model files can have
    different paths and don't fight each other's cache slots."""
    global _yolo_vehicle_model
    if _yolo_vehicle_model is None:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO (vehicle detector, COCO) from {model_path} ...")
        t0 = time.time()
        _yolo_vehicle_model = YOLO(model_path)
        logger.info(f"YOLO vehicle detector loaded in {time.time() - t0:.1f}s")
    return _yolo_vehicle_model


def _get_paddleocr():
    """Lazy-load a singleton PaddleOCR engine (English, mobile, with text
    orientation classifier). The first call downloads model weights to
    ~/.paddleocr/whl (~16MB total) and takes ~30s; subsequent calls reuse
    the in-memory engine."""
    global _paddleocr_engine
    if _paddleocr_engine is None:
        from paddleocr import PaddleOCR
        logger.info("Initialising PaddleOCR (English, mobile, angle-cls) ...")
        t0 = time.time()
        _paddleocr_engine = PaddleOCR(
            use_angle_cls=True,   # text orientation classifier
            lang="en",            # English + latin + digits (best for plates)
            show_log=False,       # suppress per-line Paddle logs
            use_gpu=False,        # CPU inference (this box has no CUDA)
            det_db_thresh=0.3,    # detection binarization threshold
            det_db_box_thresh=0.5,
            # Use the mobile/light models (fast on CPU). Override to
            # server-mode for higher accuracy at the cost of speed.
            det_model_dir=None,
            rec_model_dir=None,
            cls_model_dir=None,
        )
        logger.info(f"PaddleOCR ready in {time.time() - t0:.1f}s")
    return _paddleocr_engine


def _tesseract_ok() -> bool:
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract  # noqa
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
    return _tesseract_available


# ---------------------------------------------------------------------------
# Image preprocessing for OCR
# ---------------------------------------------------------------------------
def preprocess_for_ocr(crop: np.ndarray, upscale: int = 3) -> np.ndarray:
    """Standard ANPR preprocessing: grayscale + CLAHE + bilateral + threshold + upscale."""
    # Upscale first (interpolation works better on larger image)
    h, w = crop.shape[:2]
    if max(h, w) < 120:
        crop = cv2.resize(crop, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    # Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    # Denoise (bilateral keeps edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    # Morphological close to fill character gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed


def _enhance_plate_crop(crop_bgr: np.ndarray,
                        target_height: int = 200,
                        pad_px: int = 20) -> np.ndarray:
    """Aggressively enhance a small plate crop before feeding it to OCR.

    Why this exists: YOLO plate detections are often tight bboxes that come
    out at ~60-100 px wide on a 1920x1080 traffic photo. PaddleOCR's text
    detector needs ~30+ px tall characters to reliably find anything, and
    mobile-mode PaddleOCR (which is what we're running for speed) wants at
    least ~150 px of total image height. So we:

      1. Pad with white border (Paddle's det_db loves clean borders)
      2. Upscale to `target_height` tall, preserving aspect ratio
      3. Mild bilateral denoise (keep edges, kill sensor noise)
      4. CLAHE on grayscale (boost character contrast)
      5. Unsharp mask for edge sharpness

    Returns a 3-channel BGR image — PaddleOCR prefers 3-channel input.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    # 1. Pad with white border (helps Paddle's det_db segment text from background)
    padded = cv2.copyMakeBorder(crop_bgr, pad_px, pad_px, pad_px, pad_px,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # 2. Upscale so the shorter side hits target_height, preserving aspect
    ph, pw = padded.shape[:2]
    scale = max(target_height / ph, 1.0)
    if scale > 1.0:
        padded = cv2.resize(padded, (int(pw * scale), int(ph * scale)),
                            interpolation=cv2.INTER_CUBIC)
    # 3. Bilateral denoise on a grayscale copy, keep colour for the OCR call
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    # 4. CLAHE for contrast (don't binarise — Paddle does that internally)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 5. Unsharp mask: sharpen = original + amount * (original - blurred)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    # Merge sharpened grayscale back into 3 channels so PaddleOCR stays happy
    enhanced_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------
class TrafficPlateDetector:
    def __init__(self,
                 yolo_path: str = None,
                 coco_yolo_path: str = None,
                 conf_threshold: float = 0.30,
                 iou_threshold: float = 0.45,
                 vehicle_conf_threshold: float = 0.35,
                 crops_dir: str = "crops",
                 results_dir: str = "results"):
        if yolo_path is None:
            # Default: use Koushim's YOLOv8 license-plate model. It's
            # specifically trained on Indian plates and gives more
            # accurate bboxes than the Rawzy YOLOv10n. (User preference
            # from prior session — YOLOv8 over YOLOv10 for this project.)
            candidate = Path(
                "C:/Users/gsash/Downloads/vnpr/models/"
                "models--Koushim--yolov8-license-plate-detection/"
                "snapshots/9aaa5cd490abe0c165882ba87f4f62658ab54d01/best.pt"
            )
            yolo_path = str(candidate) if candidate.exists() else "yolov8n.pt"
        if coco_yolo_path is None:
            # COCO yolov8n — auto-downloads if not present (~6.5MB).
            # Order of lookup: user-home cached copy first, then whatever
            # ultralytics has already auto-downloaded to its own cache.
            candidates = [
                Path("C:/Users/gsash/yolov8n.pt"),
                Path.home() / "yolov8n.pt",
                Path("C:/Users/gsash/Downloads/vnpr/yolov8n.pt"),
                Path("yolov8n.pt"),  # ultralytics will auto-download
            ]
            coco_yolo_path = next((str(p) for p in candidates if p.exists()), "yolov8n.pt")
        self.yolo_path = yolo_path
        self.coco_yolo_path = coco_yolo_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.vehicle_conf_threshold = vehicle_conf_threshold
        self.crops_dir = Path(crops_dir)
        self.results_dir = Path(results_dir)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Lazy-loaded
        self._yolo = None
        self._yolo_coco = None
        self._ocr = None
        self._ready = False

    def warmup(self):
        """Eagerly load all models so first request is fast."""
        if self._ready:
            return
        try:
            self._yolo = _get_yolo(self.yolo_path)
        except Exception as e:
            logger.error(f"YOLO plate detector load failed: {e}")
        try:
            self._yolo_coco = _get_yolo_coco(self.coco_yolo_path)
        except Exception as e:
            logger.error(f"YOLO vehicle detector load failed: {e}")
        try:
            self._ocr = _get_paddleocr()
        except Exception as e:
            logger.error(f"PaddleOCR init failed: {e}")
        self._ready = True

    # ---- YOLO plate detection ----
    def _detect_yolo(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Run YOLO, return list of (x1, y1, x2, y2, conf)."""
        if self._yolo is None:
            self._yolo = _get_yolo(self.yolo_path)
        results = self._yolo.predict(img_bgr, conf=self.conf_threshold,
                                     iou=self.iou_threshold, verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, conf))
        return boxes

    # ---- YOLO vehicle detection (COCO) ----
    def _detect_vehicles(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run COCO-pretrained YOLO, filter to the 5 vehicle classes, return
        list of {class_id, class_name, conf, bbox:[x1,y1,x2,y2]}.

        We use the `classes=` kwarg of ultralytics so the NMS step only
        keeps the classes we care about — no post-filtering needed.
        """
        if self._yolo_coco is None:
            self._yolo_coco = _get_yolo_coco(self.coco_yolo_path)
        vehicle_class_ids = list(COCO_VEHICLE_CLASSES.keys())
        results = self._yolo_coco.predict(
            img_bgr,
            conf=self.vehicle_conf_threshold,
            iou=self.iou_threshold,
            classes=vehicle_class_ids,
            verbose=False,
        )
        vehicles: List[Dict[str, Any]] = []
        for r in results:
            # r.names maps class_id -> class_name (COCO names)
            names = r.names
            for b in r.boxes:
                cls_id = int(b.cls[0].cpu().numpy())
                # Should never fail given the `classes=` filter, but be safe
                if cls_id not in COCO_VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf[0].cpu().numpy())
                vehicles.append({
                    "class_id": cls_id,
                    "class_name": COCO_VEHICLE_CLASSES[cls_id],
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
        return vehicles

    # ---- OCR on a cropped plate ----
    def _ocr_plate(self, crop: np.ndarray) -> Tuple[str, float, str, List[str]]:
        """Run OCR on a cropped plate image. Returns (final_text, confidence, raw_text, fixes).

        Strategy: PaddleOCR (mobile, angle-cls) + Tesseract, on MULTIPLE
        preprocessing variants of the same crop, then vote on the result.

        Why voting: a single OCR pass on a 60-100px-wide YOLO crop is noisy.
        PaddleOCR's mobile text detector hallucinates plausible-looking
        strings (e.g. "CHROSONTO3S" from a real "12 BH 8303" plate) when
        fed low-resolution input. Running the crop through several
        preprocessing pipelines (enhanced, binarised, sharp, Otsu) and
        multiple engines (PaddleOCR, Tesseract) gives a small set of
        candidates; the one that validates as a real Indian plate format
        wins. If none validate, we fall back to the highest-confidence
        result with position-aware correction.
        """
        if crop is None or crop.size == 0:
            return "", 0.0, "", []

        # --- Build preprocessing variants ---
        variants: List[Tuple[str, np.ndarray]] = []

        # 1) Enhanced (pad + upscale + sharpen, no binarise) — best for Paddle
        try:
            variants.append(("paddle_enh", _enhance_plate_crop(crop, target_height=240, pad_px=24)))
        except Exception as e:
            logger.warning(f"enhance variant failed: {e}")

        # 2) Binarised (CLAHE + adaptive threshold) — best for Tesseract
        try:
            binv = preprocess_for_ocr(crop, upscale=5)
            variants.append(("paddle_bin", cv2.cvtColor(binv, cv2.COLOR_GRAY2BGR)))
        except Exception as e:
            logger.warning(f"binarise variant failed: {e}")

        # 3) Sharp grayscale (heavy unsharp mask, no binarise) — second Paddle path
        try:
            padded = cv2.copyMakeBorder(crop, 24, 24, 24, 24,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
            ph, pw = padded.shape[:2]
            scale = max(240 / ph, 1.0)
            if scale > 1.0:
                padded = cv2.resize(padded, (int(pw * scale), int(ph * scale)),
                                    interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 5, 50, 50)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
            sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
            variants.append(("paddle_sharp", cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)))
        except Exception as e:
            logger.warning(f"sharp variant failed: {e}")

        # --- Run PaddleOCR on every variant ---
        candidates: List[Tuple[str, str, float]] = []  # (source, text, conf)
        if self._ocr is None:
            try:
                self._ocr = _get_paddleocr()
            except Exception as e:
                logger.warning(f"PaddleOCR init failed: {e}")

        for src, img in variants:
            if self._ocr is None:
                break
            try:
                results = self._ocr.ocr(img, cls=True)
                lines = results[0] if results else []
                if not lines:
                    continue
                lines = sorted(lines, key=lambda l: l[0][0][0])
                text = "".join(
                    l[1][0].upper().replace(" ", "").replace("\n", "")
                    for l in lines
                )
                conf = float(np.mean([l[1][1] for l in lines]))
                if text and len(text) >= 4:
                    candidates.append((src, text, conf))
            except Exception as e:
                logger.warning(f"PaddleOCR [{src}] failed: {e}")

        # --- Run Tesseract on the binarised variant (different strengths) ---
        if _tesseract_ok():
            try:
                # Try Tesseract with both psm 7 (single line) and psm 8 (single word)
                tess_input = preprocess_for_ocr(crop, upscale=5)
                import pytesseract
                for psm in (7, 8, 6):
                    try:
                        t_text = pytesseract.image_to_string(
                            tess_input,
                            config=(f"--psm {psm} -c "
                                    "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                        ).upper().replace(" ", "").replace("\n", "").replace("\r", "")
                        if t_text and len(t_text) >= 4:
                            candidates.append((f"tess_psm{psm}", t_text, 0.55))
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")

        # --- Run EasyOCR (different model, often better on small text) ---
        try:
            import easyocr  # lazy import — heavy model
            if not hasattr(self, "_easy_reader") or self._easy_reader is None:
                self._easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            easy = self._easy_reader
            # EasyOCR on the enhanced variant (it works well on grayscale-ish inputs)
            if variants:
                src_name, src_img = "easy_enh", variants[0][1]
                try:
                    easy_results = easy.readtext(
                        src_img, detail=1, paragraph=False,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        text_threshold=0.4,
                    )
                    if easy_results:
                        # EasyOCR returns [(bbox, text, conf), ...]
                        easy_text = "".join(
                            r[1].upper().replace(" ", "").replace("\n", "")
                            for r in sorted(easy_results, key=lambda x: x[0][0][0])
                        )
                        if easy_text and len(easy_text) >= 4:
                            avg_conf = float(np.mean([r[2] for r in easy_results]))
                            candidates.append((src_name, easy_text, avg_conf))
                except (ValueError, TypeError) as ve:
                    logger.warning(f"EasyOCR parse failed: {ve}")
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")

        if not candidates:
            return "", 0.0, "", []

        # --- Vote / score ---
        def _score(item: Tuple[str, str, float]) -> float:
            src, text, conf = item
            v = validate_indian_plate(text)
            # Highest priority: passes Indian-plate format validation
            if v["valid"]:
                base = 1000.0
            else:
                # Bonus if it at least looks plate-like (6-11 chars, mostly alnum)
                base = 0.0
                if 6 <= len(text) <= 11:
                    base += 50.0
                alnum = sum(1 for c in text if c.isalnum())
                base += 10.0 * (alnum / max(1, len(text)))
            # Tie-break: OCR confidence (small weight)
            return base + conf

        candidates.sort(key=_score, reverse=True)
        best_src, best_text, best_conf = candidates[0]

        # --- Apply position-aware correction on the chosen result ---
        final_text, fixes = correct_position_aware(best_text)
        # Re-validate: if correction made it valid, prefer the corrected
        if not validate_indian_plate(best_text)["valid"] and validate_indian_plate(final_text)["valid"]:
            pass  # already using final_text
        elif validate_indian_plate(best_text)["valid"]:
            # The raw OCR read is already valid; use it directly (no
            # risk of the corrector munging a good read).
            final_text = best_text
            fixes = []
        return final_text, float(best_conf), best_text, fixes

    # ---- Public API ----
    def detect(self, image_path: str, save_id: Optional[str] = None) -> Dict[str, Any]:
        """Run full pipeline on an image. Returns a result dict with both
        `plates` (with OCR'd text + Indian-format validation) and
        `vehicles` (COCO classes: bicycle/car/motorcycle/bus/truck)."""
        self.warmup()
        t_start = time.time()
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"error": f"Failed to read image: {image_path}",
                    "plates": [], "vehicles": [], "num_plates": 0, "num_vehicles": 0}
        h, w = img_bgr.shape[:2]
        annotated = img_bgr.copy()

        # Step 1: Vehicle detection (COCO-pretrained YOLOv8n)
        # Faster than plate detection and gives scene context for plates.
        t_veh = time.time()
        vehicles_raw = self._detect_vehicles(img_bgr)
        vehicle_t = round(time.time() - t_veh, 3)
        # Sort by area desc (largest first) for stable UI ordering
        vehicles_raw.sort(key=lambda v: (v["bbox"][2]-v["bbox"][0]) *
                                          (v["bbox"][3]-v["bbox"][1]),
                           reverse=True)
        # Build per-class counts (zero-initialised)
        vehicle_counts = {name: 0 for name in COCO_VEHICLE_CLASSES.values()}
        for v in vehicles_raw:
            vehicle_counts[v["class_name"]] += 1
            v["emoji"] = VEHICLE_EMOJI.get(v["class_name"], "🚙")
            v["color"] = list(VEHICLE_COLORS.get(v["class_name"], (200, 200, 200)))
        # Annotate vehicles on the image
        for v in vehicles_raw:
            x1, y1, x2, y2 = v["bbox"]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c - x1c < 8 or y2c - y1c < 8:
                continue
            color = tuple(v["color"])
            cv2.rectangle(annotated, (x1c, y1c), (x2c, y2c), color, 2)
            # OpenCV's Hershey fonts can't render emoji (they show as `????`).
            # Skip the emoji here — the browser UI shows it. Just use class + conf.
            label = f"{v['class_name']} {v['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Dark background for readability over varied photo colors
            cv2.rectangle(annotated, (x1c, y1c), (x1c + tw + 6, y1c + th + 6), color, -1)
            cv2.putText(annotated, label, (x1c + 3, y1c + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Step 2: Plate detection (YOLOv10n Rawzy)
        t_plate = time.time()
        boxes = self._detect_yolo(img_bgr)
        plate_t = round(time.time() - t_plate, 3)

        # Step 3-4: For each plate box, crop, preprocess, OCR
        plates: List[Dict[str, Any]] = []
        if save_id is None:
            save_id = Path(image_path).stem
        for i, (x1, y1, x2, y2, det_conf) in enumerate(boxes):
            # Expand the bbox with generous padding (50% on each side).
            # Why 50% and not just a few pixels: the YOLO plate detector
            # returns a tight bbox that may exclude the actual character
            # stroke edges. PaddleOCR's text detector works much better
            # when it has surrounding context (a few pixels of plate
            # background / frame). 50% padding is the sweet spot — enough
            # context for the OCR, not so much that we pull in unrelated
            # vehicle text. This was confirmed by tests on the synthetic
            # multi-plate image where 20% padding was too tight and 50-100%
            # made the difference between 0/4 and 4/4 plates read correctly.
            bw, bh = x2 - x1, y2 - y1
            pad_x = max(int(bw * 0.50), 10)
            pad_y = max(int(bh * 0.50), 6)
            x1c, y1c = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2c, y2c = min(w, x2 + pad_x), min(h, y2 + pad_y)
            if x2c - x1c < 12 or y2c - y1c < 8:
                continue
            crop = img_bgr[y1c:y2c, x1c:x2c]
            crop_filename = f"{save_id}_p{i}.jpg"
            crop_path = self.crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            # OCR
            final_text, ocr_conf, raw_text, fixes = self._ocr_plate(crop)
            validation = validate_indian_plate(final_text)
            plate_info = {
                "id": i,
                "text": final_text,
                "raw_ocr": raw_text,
                "fixes_applied": fixes,
                "detection_confidence": round(det_conf, 3),
                "ocr_confidence": round(ocr_conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_url": f"/crops/{crop_filename}",
                "valid_format": validation["valid"],
                "format_type": validation["format"],
                "state_code": validation.get("state"),
            }
            plates.append(plate_info)
            # Annotate the source image (drawn AFTER vehicles so plates sit on top)
            color = (0, 200, 80) if validation["valid"] else (0, 140, 255)
            cv2.rectangle(annotated, (x1c, y1c), (x2c, y2c), color, 3)
            label = final_text if final_text else f"plate #{i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1c, max(0, y1c - th - 8)), (x1c + tw + 8, y1c), color, -1)
            cv2.putText(annotated, label, (x1c + 4, max(15, y1c - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Step 5: Save annotated
        annotated_filename = f"{save_id}_annotated.jpg"
        annotated_path = self.results_dir / annotated_filename
        cv2.imwrite(str(annotated_path), annotated)
        # Sort plates by detection confidence desc
        plates.sort(key=lambda p: p["detection_confidence"], reverse=True)
        elapsed = round(time.time() - t_start, 2)
        return {
            "annotated_url": f"/results/{annotated_filename}",
            "original_url": f"/uploads/{Path(image_path).name}",
            "plates": plates,
            "vehicles": vehicles_raw,
            "vehicle_counts": vehicle_counts,
            "num_plates": len(plates),
            "num_valid": sum(1 for p in plates if p["valid_format"]),
            "num_vehicles": len(vehicles_raw),
            "image_size": [w, h],
            "elapsed_seconds": elapsed,
            "vehicle_elapsed_seconds": vehicle_t,
            "plate_elapsed_seconds": plate_t,
            "plate_model": "Koushim/yolov8-license-plate-detection (Indian-plate specialist)",
            "vehicle_model": "yolov8n (COCO-pretrained)",
            "ocr_backends": ["paddleocr-mobile", "easyocr"] + (["tesseract"] if _tesseract_ok() else []),
        }


# ---------------------------------------------------------------------------
# CLI for one-off testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path>")
        sys.exit(1)
    det = TrafficPlateDetector()
    result = det.detect(sys.argv[1])
    print(f"\nDetected {result.get('num_plates', 0)} plate(s) in {result.get('elapsed_seconds')}s")
    for p in result.get("plates", []):
        flag = "VALID" if p["valid_format"] else "INVALID"
        print(f"  [{flag}] {p['text']:<12} det={p['detection_confidence']:.2f} "
              f"ocr={p['ocr_confidence']:.2f} raw='{p['raw_ocr']}'")
