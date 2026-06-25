import time
import base64
import cv2
import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

YOLO_MODEL = HERE / "yolo11_plate.pt"
AWIROS_DIR = HERE / "awiros_anpr"

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
