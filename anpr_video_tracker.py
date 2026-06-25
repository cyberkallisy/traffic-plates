"""
ANPR with video-based tracker voting (Alternative Route #3).

Pipeline (per frame):
  1. YOLO plate detector -> list of plate bboxes + detector conf
  2. For each plate crop, run PaddleOCR (multi-variant for hard cases)
  3. Track plates across frames with one of: ByteTrack, BoT-SORT,
     DeepOC-SORT (ultralytics built-ins) or DeepSORT (deep_sort_realtime)
  4. For each track_id, accumulate all OCR readings
  5. After all frames, do CHARACTER-POSITION voting across that track's
     readings -> final plate text + final confidence

Why this matters: a single PaddleOCR pass on a 60-100px crop is noisy
(HR26DK833? / HR26DK8337 / HR26DK83?7 across frames). The same plate
seen 20 frames gives 20 OCR reads. Voting position-by-position
collapses that noise into the true plate with much higher confidence.

Usage:
    python anpr_video_tracker.py --video <path-to-video.mp4> --tracker bytetrack
    python anpr_video_tracker.py --video <path-to-video.mp4> --tracker deep_sort
    python anpr_video_tracker.py --video <path-to-video.mp4> --tracker botsort
    python anpr_video_tracker.py --video <path-to-video.mp4> --tracker deepocsort

Outputs (next to the input video, in <video_stem>_anpr_<tracker>_<ts>/):
    annotated.mp4    - video with bbox + track-id + per-frame OCR + voted text
    frames/          - per-frame annotated JPGs (sampled for HTML)
    crops/           - plate crops grouped by track_id
    tracks.json      - per-track detailed votes + final plate text
    summary.json     - top-level stats (num tracks, plates read, timing)
    report.html      - human-friendly HTML viewer with lightbox
"""

import os
# PaddleOCR + protobuf workaround (must be set BEFORE paddle imports).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json
import logging
import re
import sys
import time
import base64
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("anpr-video-tracker")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YOLO11_PT = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt")
DET_CONF  = 0.25          # YOLO plate detector threshold
OCR_CONF  = 0.30          # drop PaddleOCR lines below this
PAD_FRAC  = 0.50          # bbox padding (50% is the sweet spot per skill)
IMGSZ     = 640
DEVICE    = "cpu"

# PaddleOCR server det model (higher accuracy on tiny plates).
PADDLE_DET_DIR = Path(r"C:/Users/gsash/Downloads/traffic-plates/ch_PP-OCRv4_det_server_infer")
PADDLE_REC_DIR = Path(r"C:/Users/gsash/Downloads/traffic-plates/ch_PP-OCRv4_rec_server_infer")
PADDLE_CLS_DIR = Path(r"C:/Users/gsash/Downloads/traffic-plates/ch_PP-OCRv3_cls_server")

# Indian-plate format validators (used only as voting tie-breaker).
INDIAN_RE     = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
INDIAN_BH_RE  = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HR","HP",
    "JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TS","TR","UP","UK","WB","BH",
}

# Confusion pairs used in position-aware correction (if voting result fails validation).
CONFUSION_LETTER_TO_DIGIT = {
    "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
    "A": "4", "G": "6", "T": "7", "P": "9", "E": "3",
}
CONFUSION_DIGIT_TO_LETTER = {v: k for k, v in CONFUSION_LETTER_TO_DIGIT.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def is_valid_indian_plate(t: str) -> bool:
    if not t:
        return False
    if INDIAN_BH_RE.match(t):
        return True
    if INDIAN_RE.match(t):
        return t[:2] in INDIAN_STATES
    return False


def position_aware_correct(t: str) -> str:
    """If a track voted plate is slightly off (e.g. HRZ6DK1234), swap
    single-position letter<->digit confusions to match Indian structure."""
    if not t or is_valid_indian_plate(t):
        return t
    out = list(t)
    for i, ch in enumerate(out):
        if i in (0, 1, 4, 5):
            if ch in CONFUSION_LETTER_TO_DIGIT:
                out[i] = CONFUSION_LETTER_TO_DIGIT[ch]
        elif i in (2, 3) or i >= 7:
            if ch in CONFUSION_DIGIT_TO_LETTER:
                out[i] = CONFUSION_DIGIT_TO_LETTER[ch]
    return "".join(out)


def pad_bbox(x1, y1, x2, y2, w, h, pad_frac=PAD_FRAC):
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(bw * pad_frac), 10)
    pad_y = max(int(bh * pad_frac), 6)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y))


def preprocess_plate(crop_bgr: np.ndarray, target_h: int = 240) -> np.ndarray:
    """CLAHE + bilateral + upscale + white-pad. PaddleOCR mobile likes this."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    if max(h, w) < target_h:
        s = target_h / max(h, 1)
        crop_bgr = cv2.resize(crop_bgr, (int(w * s), int(h * s)),
                              interpolation=cv2.INTER_CUBIC)
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


# ---------------------------------------------------------------------------
# OCR (multi-backend: PaddleOCR for fast, vision_analyze for tiny plates)
# ---------------------------------------------------------------------------
class PlateOCR:
    """Lazy PaddleOCR wrapper. Tries server det first, falls back to mobile.
    For tiny plates (<60px tall), PaddleOCR will return empty — switch to
    vision_analyze() (handled in the main loop, not here)."""

    def __init__(self):
        self._engine = None
        self._variant = None

    def _get(self):
        if self._engine is not None:
            return self._engine
        from paddleocr import PaddleOCR
        if PADDLE_DET_DIR.exists() and PADDLE_REC_DIR.exists():
            log.info("PaddleOCR: using SERVER det+rec models (better on small plates)")
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                det_model_dir=str(PADDLE_DET_DIR),
                rec_model_dir=str(PADDLE_REC_DIR),
                cls_model_dir=str(PADDLE_CLS_DIR) if PADDLE_CLS_DIR.exists() else None,
                use_gpu=False,
                show_log=False,
            )
            self._variant = "server"
        else:
            log.info("PaddleOCR: using MOBILE det model (server files not found)")
            self._engine = PaddleOCR(
                use_angle_cls=True, lang="en", use_gpu=False, show_log=False,
            )
            self._variant = "mobile"
        return self._engine

    def read(self, crop_bgr: np.ndarray, min_height: int = 60):
        """Returns list of (text, conf) candidates.
        If crop height is below `min_height`, PaddleOCR will likely fail.
        Caller should use vision_analyze() instead for tiny plates."""
        engine = self._get()
        proc = preprocess_plate(crop_bgr, target_h=240)
        if proc is None or proc.size == 0:
            return [], self._variant
        try:
            results = engine.ocr(proc, cls=True)
        except Exception as e:
            log.debug("PaddleOCR failed: %s", e)
            return [], self._variant
        if not results or not results[0]:
            return [], self._variant
        cands = []
        for line in results[0]:
            try:
                box, (text, conf) = line
            except Exception:
                continue
            t = clean_text(text)
            if len(t) >= 4 and conf >= OCR_CONF:
                cands.append((t, float(conf)))
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands, self._variant

    @property
    def variant(self) -> str:
        return self._variant or "unknown"


# ---------------------------------------------------------------------------
# Character-position voting across a track's OCR reads
# ---------------------------------------------------------------------------
def vote_track_text(reads: list) -> dict:
    """
    reads: list of (text, conf) for one track_id.
    Returns: {"text": ..., "conf": ..., "valid": ..., "votes": ...}

    Voting rule per position:
      - Bucket each char by its (position, char) pair
      - Score = sum of OCR confidences that voted for it
      - Pick the highest-scoring char per position
      - Length = the most-common read length (with min-count tiebreak)
    """
    if not reads:
        return {"text": "", "conf": 0.0, "valid": False, "votes": {}}

    len_counter = Counter(len(t) for t, _ in reads)
    best_len, _ = len_counter.most_common(1)[0]

    pos_buckets: dict = defaultdict(lambda: defaultdict(float))
    for text, conf in reads:
        text = text[:best_len] if len(text) >= best_len else text
        for i, ch in enumerate(text):
            pos_buckets[i][ch] += conf

    chars = []
    confs = []
    votes_per_pos = {}
    for i in range(best_len):
        if i not in pos_buckets:
            chars.append("?")
            confs.append(0.0)
            votes_per_pos[i] = {}
            continue
        bucket = pos_buckets[i]
        winner, win_score = max(bucket.items(), key=lambda kv: kv[1])
        total = sum(bucket.values())
        win_conf = (win_score / total) if total > 0 else 0.0
        chars.append(winner)
        confs.append(win_conf)
        votes_per_pos[i] = {k: round(v, 3) for k, v in sorted(
            bucket.items(), key=lambda kv: -kv[1])}

    text = "".join(chars)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    valid = is_valid_indian_plate(text)

    if not valid:
        corrected = position_aware_correct(text)
        if is_valid_indian_plate(corrected) and corrected != text:
            text = corrected
            valid = True
            avg_conf = min(1.0, avg_conf + 0.10)

    return {
        "text": text,
        "conf": round(avg_conf, 3),
        "valid": valid,
        "votes": votes_per_pos,
    }


# ---------------------------------------------------------------------------
# Tracker backends
# ---------------------------------------------------------------------------
class BaseTracker:
    def update(self, detections_bgr_xyxy_conf: list, frame) -> list:
        raise NotImplementedError


class UltralyticsTracker(BaseTracker):
    """Wraps ultralytics YOLO.track() with one of: bytetrack, botsort, deepocsort.
    ONE call per frame does both detection AND tracking."""

    def __init__(self, model, tracker_type: str = "bytetrack"):
        self.model = model
        self.tracker_type = tracker_type
        log.info("Tracker: ultralytics %s", tracker_type)

    def update(self, detections, frame):
        results = self.model.track(
            frame, persist=True, tracker=f"{self.tracker_type}.yaml",
            verbose=False, imgsz=IMGSZ, conf=DET_CONF, device=DEVICE,
        )
        out = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            for tid, (x1, y1, x2, y2), cf in zip(ids, xyxy, confs):
                bw, bh = x2 - x1, y2 - y1
                # Aspect-ratio filter: real plates are 2:1 to 5:1 (wider than tall).
                # Car body panels / bus-side text are usually close to square or
                # very tall, so this kills the bulk of the false positives.
                if bh <= 0 or bw <= 0:
                    continue
                ar = bw / bh
                if ar < 1.8 or ar > 6.0:
                    continue
                # Size filter: plates 30..280 px tall in this 720p footage.
                if bh < 18 or bh > 280:
                    continue
                out.append((int(tid), (int(x1), int(y1), int(x2), int(y2)), float(cf)))
        return out


class DeepSORTTracker(BaseTracker):
    """Real DeepSORT via deep_sort_realtime. We feed the plate crops
    as detections to the tracker - this is the right granularity for
    plate tracking across frames."""

    def __init__(self, embedder: str = "mobilenet", max_age: int = 30):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.ds = DeepSort(
            max_age=max_age,
            n_init=2,
            nms_max_overlap=1.0,
            max_iou_distance=0.7,
            max_cosine_distance=0.2,
            embedder=embedder,
            half=False,
            bgr=True,
        )
        self._next_id = 0
        self._id_map = {}
        log.info("Tracker: DeepSORT (deep_sort_realtime, embedder=%s)", embedder)

    def update(self, detections, frame):
        if not detections:
            self.ds.update_tracks([], frame=frame)
            return []
        # deep_sort_realtime API expects raw_detections as an iterable of
        # ([x1, y1, x2, y2], confidence) tuples — NOT Detection objects.
        raw_dets = []
        for crop, (x1, y1, x2, y2), yconf in detections:
            raw_dets.append(([x1, y1, x2, y2], float(yconf)))
        tracks = self.ds.update_tracks(raw_dets, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            ds_id = t.track_id
            if ds_id not in self._id_map:
                self._id_map[ds_id] = self._next_id
                self._next_id += 1
            tid = self._id_map[ds_id]
            l, top, r, b = t.to_ltrb()
            out.append((tid, (int(l), int(top), int(r), int(b)), float(t.det_conf or 0.0)))
        return out


def make_tracker(name: str, model):
    n = name.lower()
    if n in ("bytetrack", "botsort", "deepocsort"):
        return UltralyticsTracker(model, tracker_type=n)
    if n in ("deep_sort", "deepsort"):
        return DeepSORTTracker(embedder="mobilenet", max_age=30)
    raise ValueError(f"Unknown tracker: {name}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_video(video_path: Path, tracker_name: str, out_dir: Path,
                  max_frames: int = None, write_video: bool = True,
                  frame_stride: int = 1) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    crops_dir = out_dir / "crops"
    frames_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Video: %s  %dx%d  fps=%.2f  frames=%d  stride=%d",
             video_path, W, H, fps, total, frame_stride)

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO11_PT))
    tracker = make_tracker(tracker_name, yolo)
    ocr = PlateOCR()
    log.info("PaddleOCR variant: %s", ocr.variant)

    out_video = None
    if write_video:
        out_video_path = out_dir / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))

    history = defaultdict(lambda: {
        "reads": [],
        "frames": [],
        "bboxes": [],
        "yconfs": [],
        "first_seen": None,
        "last_seen": None,
    })

    frame_log = []

    n_proc = 0
    t0 = time.time()
    last_log_t = t0

    is_ultra = isinstance(tracker, UltralyticsTracker)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Frame stride: when stride>1, only process every Nth frame.
        # We still read the frame so cv2's decoder stays in sync (trackers
        # like ByteTrack need the full sequence to be coherent), but we
        # increment n_proc and run the pipeline only on the keep-frames.
        n_proc += 1
        if max_frames and n_proc > max_frames:
            break
        if frame_stride > 1 and (n_proc - 1) % frame_stride != 0:
            continue

        if is_ultra:
            # ONE call: detection + tracking
            tracked = tracker.update([], frame)
            # Build crop list in same order as tracked ids
            plate_inputs = []
            tracked_with_crops = []
            for tid, (tx1, ty1, tx2, ty2), yconf in tracked:
                x1c, y1c, x2c, y2c = pad_bbox(tx1, ty1, tx2, ty2, W, H)
                crop = frame[y1c:y2c, x1c:x2c].copy()
                plate_inputs.append((crop, (tx1, ty1, tx2, ty2), yconf))
                tracked_with_crops.append((tid, (tx1, ty1, tx2, ty2), crop, yconf))
        else:
            # DeepSORT: detect first, hand off to tracker
            dets = yolo.predict(frame, verbose=False, imgsz=IMGSZ,
                                conf=DET_CONF, device=DEVICE)[0]
            plate_inputs = []
            if dets.boxes is not None and len(dets.boxes) > 0:
                for box in dets.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    yconf = float(box.conf[0].cpu().item())
                    x1c, y1c, x2c, y2c = pad_bbox(int(x1), int(y1), int(x2), int(y2), W, H)
                    crop = frame[y1c:y2c, x1c:x2c].copy()
                    plate_inputs.append((crop, (int(x1), int(y1), int(x2), int(y2)), yconf))
            tracked = tracker.update(plate_inputs, frame)
            tracked_with_crops = None  # use IoU association below

        # OCR per tracked plate
        frame_ocr = []
        if is_ultra:
            for tid, bbox, crop, yconf in tracked_with_crops:
                if crop.size == 0:
                    continue
                cands, _ = ocr.read(crop)
                if cands:
                    text, conf = cands[0]
                else:
                    text, conf = "", 0.0
                history[tid]["reads"].append((text, conf))
                history[tid]["frames"].append(n_proc)
                history[tid]["bboxes"].append(bbox)
                history[tid]["yconfs"].append(yconf)
                if history[tid]["first_seen"] is None:
                    history[tid]["first_seen"] = n_proc
                history[tid]["last_seen"] = n_proc
                frame_ocr.append((tid, bbox, text, conf, yconf))
        else:
            used = set()
            for tid, (tx1, ty1, tx2, ty2), _ in tracked:
                best_iou, best_idx = 0.0, -1
                for j, (_, (bx1, by1, bx2, by2), _) in enumerate(plate_inputs):
                    if j in used:
                        continue
                    ix1 = max(tx1, bx1); iy1 = max(ty1, by1)
                    ix2 = min(tx2, bx2); iy2 = min(ty2, by2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    union = ((tx2 - tx1) * (ty2 - ty1) + (bx2 - bx1) * (by2 - by1) - inter)
                    iou = inter / max(union, 1)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_idx < 0 or best_iou < 0.1:
                    continue
                used.add(best_idx)
                crop, _, yconf = plate_inputs[best_idx]
                if crop.size == 0:
                    continue
                cands, _ = ocr.read(crop)
                if cands:
                    text, conf = cands[0]
                else:
                    text, conf = "", 0.0
                history[tid]["reads"].append((text, conf))
                history[tid]["frames"].append(n_proc)
                history[tid]["bboxes"].append((tx1, ty1, tx2, ty2))
                history[tid]["yconfs"].append(yconf)
                if history[tid]["first_seen"] is None:
                    history[tid]["first_seen"] = n_proc
                history[tid]["last_seen"] = n_proc
                frame_ocr.append((tid, (tx1, ty1, tx2, ty2), text, conf, yconf))

        # Save crop (per track) for HTML viewer
        for tid, bbox, text, conf, yconf in frame_ocr:
            tx1, ty1, tx2, ty2 = bbox
            x1c, y1c, x2c, y2c = pad_bbox(tx1, ty1, tx2, ty2, W, H)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                crop_path = crops_dir / f"track{tid:03d}_f{n_proc:06d}.jpg"
                cv2.imwrite(str(crop_path), crop)

        # Annotate frame
        ann = frame.copy()
        for tid, bbox, text, conf, yconf in frame_ocr:
            tx1, ty1, tx2, ty2 = bbox
            cv2.rectangle(ann, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            label = f"ID{tid} {text or '?'} c={conf:.2f}"
            cv2.putText(ann, label, (tx1, max(0, ty1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(ann, f"frame {n_proc}/{total}  tracker={tracker_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(ann, f"frame {n_proc}/{total}  tracker={tracker_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if out_video is not None:
            out_video.write(ann)
        if n_proc % max(1, int(fps)) == 0 or n_proc == 1:
            cv2.imwrite(str(frames_dir / f"frame_{n_proc:06d}.jpg"), ann)
        frame_log.append({
            "frame": n_proc,
            "tracks": [
                {"id": tid, "bbox": list(bbox), "text": text,
                 "ocr_conf": round(conf, 3), "yolo_conf": round(yconf, 3)}
                for tid, bbox, text, conf, yconf in frame_ocr
            ]
        })

        if time.time() - last_log_t > 3.0:
            log.info("  frame %d / %d (%.1fs elapsed)", n_proc, total,
                     time.time() - t0)
            last_log_t = time.time()

    cap.release()
    if out_video is not None:
        out_video.release()

    elapsed = time.time() - t0
    log.info("Processed %d frames in %.1fs (%.2f fps)",
             n_proc, elapsed, n_proc / max(elapsed, 0.01))

    # Vote per track
    tracks_out = []
    for tid in sorted(history.keys()):
        h = history[tid]
        n_reads = len(h["reads"])
        if n_reads == 0:
            final_text, final_conf, valid = "", 0.0, False
            votes_per_pos = {}
        else:
            v = vote_track_text(h["reads"])
            final_text, final_conf, valid = v["text"], v["conf"], v["valid"]
            votes_per_pos = v["votes"]
        avg_yconf = float(np.mean(h["yconfs"])) if h["yconfs"] else 0.0
        most_common_reads = Counter(h["reads"]).most_common(5)
        tracks_out.append({
            "track_id": tid,
            "n_frames": n_reads,
            "first_frame": h["first_seen"],
            "last_frame": h["last_seen"],
            "avg_yolo_conf": round(avg_yconf, 3),
            "final_text": final_text,
            "final_conf": final_conf,
            "valid_indian": valid,
            "votes": votes_per_pos,
            "all_reads": [{"text": t, "conf": round(c, 3)} for t, c in h["reads"]],
            "reads_per_frame": list(zip(h["frames"], h["reads"])),
            "top_reads": [{"text": t, "conf": round(c, 3), "count": n}
                          for (t, c), n in most_common_reads],
        })

    tracks_out.sort(key=lambda r: (not r["valid_indian"], -r["n_frames"]))

    summary = {
        "video": str(video_path),
        "tracker": tracker_name,
        "yolo_model": str(YOLO11_PT),
        "paddle_variant": ocr.variant,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(n_proc / max(elapsed, 0.01), 2),
        "n_frames_processed": n_proc,
        "n_total_frames": total,
        "video_fps": round(fps, 2),
        "video_resolution": [W, H],
        "n_tracks": len(tracks_out),
        "n_valid_plates": sum(1 for t in tracks_out if t["valid_indian"]),
        "tracks": tracks_out,
    }
    with open(out_dir / "tracks.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "frame_log.json", "w") as f:
        json.dump({"frames": frame_log[:5000]}, f)

    slim = {k: summary[k] for k in
            ["video", "tracker", "elapsed_sec", "fps", "n_frames_processed",
             "n_tracks", "n_valid_plates"]}
    slim["top_tracks"] = [
        {"id": t["track_id"], "final": t["final_text"],
         "conf": t["final_conf"], "valid": t["valid_indian"],
         "frames": t["n_frames"]} for t in tracks_out[:10]
    ]
    with open(out_dir / "summary.json", "w") as f:
        json.dump(slim, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# HTML report (lightbox viewer)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>ANPR Video Tracker Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
background:#0f1115;color:#e6e6e6;margin:0;padding:24px;}}
h1{{margin:0 0 8px 0;font-size:22px;}}
.sub{{color:#9aa0a6;font-size:13px;margin-bottom:24px;}}
.card{{background:#161a22;border:1px solid #232936;border-radius:10px;
padding:16px;margin:14px 0;}}
.card h2{{margin:0 0 8px 0;font-size:17px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
gap:10px;}}
.thumb{{position:relative;cursor:zoom-in;border-radius:6px;overflow:hidden;
background:#000;}}
.thumb img{{width:100%;display:block;}}
.tbadge{{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.75);
padding:2px 6px;border-radius:4px;font-size:11px;color:#0f0;}}
.plate{{font-family:'Courier New',monospace;font-size:18px;
color:#0f0;letter-spacing:1px;}}
.plate.bad{{color:#f55;}}
.vote{{display:inline-block;background:#1c2230;border:1px solid #2a3142;
border-radius:4px;padding:1px 5px;margin:2px;font-size:12px;font-family:monospace;}}
.vote .ch{{color:#0f0;}}
.vote .c2{{color:#ff0;}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px;}}
th,td{{padding:4px 8px;text-align:left;border-bottom:1px solid #232936;}}
th{{color:#9aa0a6;font-weight:500;}}
tr:hover td{{background:#1a1f2a;}}
.modal{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
background:rgba(0,0,0,.92);z-index:999;align-items:center;justify-content:center;}}
.modal img{{max-width:95vw;max-height:95vh;box-shadow:0 0 20px #000;}}
.modal.show{{display:flex;cursor:zoom-out;}}
.meta{{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;}}
.mchip{{background:#1c2230;border:1px solid #2a3142;border-radius:4px;
padding:2px 8px;font-size:12px;}}
.mchip b{{color:#0f0;}}
</style></head><body>
<h1>🚗 ANPR Video Tracker Report</h1>
<div class="sub">
  Video: <code>{video}</code> &middot; Tracker: <b>{tracker}</b> &middot;
  {n_frames} frames &middot; {fps} fps &middot; {elapsed}s
  &middot; {n_tracks} tracks &middot; <b style="color:#0f0">{n_valid} valid plates</b>
</div>
<div class="card">
  <h2>📊 Per-track results (best first)</h2>
  <table>
    <tr><th>ID</th><th>Frames</th><th>First→Last</th><th>Final plate</th>
        <th>Conf</th><th>Valid</th><th>YOLO</th></tr>
    {rows}
  </table>
</div>
<div class="card">
  <h2>🖼 Sampled annotated frames (click to zoom)</h2>
  <div class="grid">{thumbs}</div>
</div>
<div class="card">
  <h2>🗳 Character-position voting detail (top track)</h2>
  <div>{vote_detail}</div>
</div>
<div class="modal" id="m" onclick="this.classList.remove('show')">
  <img id="mi" src=""></img>
</div>
<script>
function zm(src){{document.getElementById('mi').src=src;
document.getElementById('m').classList.add('show');}}
</script>
</body></html>"""


def _thumb_datauri(path: Path) -> str:
    if not path.exists():
        return ""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return ""
    h, w = img_bgr.shape[:2]
    new_w = 360
    new_h = int(h * new_w / w)
    img_bgr = cv2.resize(img_bgr, (new_w, new_h))
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_html_report(out_dir: Path, summary: dict):
    rows = []
    for t in summary["tracks"][:50]:
        cls = "" if t["valid_indian"] else "bad"
        rows.append(
            f"<tr><td>{t['track_id']}</td><td>{t['n_frames']}</td>"
            f"<td>{t['first_frame']}→{t['last_frame']}</td>"
            f"<td><span class='plate {cls}'>{t['final_text'] or '?'}</span></td>"
            f"<td>{t['final_conf']}</td>"
            f"<td>{'✓' if t['valid_indian'] else '✗'}</td>"
            f"<td>{t['avg_yolo_conf']}</td></tr>"
        )

    thumbs_html = []
    for f in sorted((out_dir / "frames").glob("*.jpg")):
        duri = _thumb_datauri(f)
        if duri:
            thumbs_html.append(
                f'<div class="thumb" onclick="zm(\'{duri}\')">'
                f'<img src="{duri}"/>'
                f'<div class="tbadge">{f.stem}</div></div>'
            )

    vote_html = ""
    if summary["tracks"]:
        top = summary["tracks"][0]
        vote_html = (f"<h3>Track {top['track_id']}: "
                     f"<span class='plate'>{top['final_text'] or '?'}</span> "
                     f"(conf {top['final_conf']})</h3>")
        vote_html += "<p>Per-position voting:</p><div>"
        for pos, bucket in top.get("votes", {}).items():
            parts = []
            keys = list(bucket.keys())
            for ch, score in bucket.items():
                cls = "ch" if ch == keys[0] else "c2"
                parts.append(f"<span class='{cls}'>{ch}:{score}</span>")
            vote_html += f'<div class="vote">pos {pos}: {", ".join(parts)}</div>'
        vote_html += "</div><p style='margin-top:8px'>All OCR reads for this track:</p>"
        vote_html += "<div>"
        for r in top.get("all_reads", []):
            vote_html += f'<div class="vote">{r["text"]} <span class="c2">{r["conf"]}</span></div>'
        vote_html += "</div>"

    html = HTML_TEMPLATE.format(
        video=summary["video"],
        tracker=summary["tracker"],
        n_frames=summary["n_frames_processed"],
        fps=summary["fps"],
        elapsed=summary["elapsed_sec"],
        n_tracks=summary["n_tracks"],
        n_valid=summary["n_valid_plates"],
        rows="".join(rows),
        thumbs="".join(thumbs_html),
        vote_detail=vote_html or "<i>no tracks</i>",
    )
    with open(out_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--tracker", default="bytetrack",
                   choices=["bytetrack", "botsort", "deepocsort", "deep_sort"])
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: <video_dir>/<video_stem>_anpr_<tracker>_<ts>)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame (default=1=all). Use 5 for 5x speedup.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip writing annotated.mp4 (faster)")
    args = p.parse_args()

    if not args.video.exists():
        log.error("Video not found: %s", args.video)
        return 2

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = args.video.parent / f"{args.video.stem}_anpr_{args.tracker}_{ts}"

    log.info("Output dir: %s", args.out)
    summary = process_video(
        args.video, args.tracker, args.out,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        frame_stride=args.stride,
    )

    log.info("Building HTML report...")
    build_html_report(args.out, summary)

    log.info("=" * 60)
    log.info("DONE")
    log.info("  video:        %s", args.video)
    log.info("  tracker:      %s", summary["tracker"])
    log.info("  frames:       %d / %d", summary["n_frames_processed"], summary["n_total_frames"])
    log.info("  tracks:       %d (valid Indian: %d)",
             summary["n_tracks"], summary["n_valid_plates"])
    log.info("  elapsed:      %.1fs (%.2f fps)", summary["elapsed_sec"], summary["fps"])
    log.info("  outputs:      %s", args.out)
    log.info("    -> annotated.mp4")
    log.info("    -> tracks.json   (per-track voting + all OCR reads)")
    log.info("    -> summary.json  (top-level stats)")
    log.info("    -> report.html   (HTML viewer with lightbox)")
    log.info("")
    log.info("Top tracks:")
    for t in summary["tracks"][:10]:
        marker = "✓" if t["valid_indian"] else "✗"
        log.info("  %s ID%-3d frames=%-3d plate=%-12s conf=%.2f yolo=%.2f",
                 marker, t["track_id"], t["n_frames"],
                 t["final_text"] or "?", t["final_conf"], t["avg_yolo_conf"])
    return 0


if __name__ == "__main__":
    sys.exit(main())