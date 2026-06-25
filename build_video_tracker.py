"""
Video-based ANPR with multi-tracker voting.

Why this exists
---------------
Single-frame ANPR reads a plate from one image. If that frame is blurry,
small, or partially occluded, the OCR has to guess — and the guess is
usually one plausible-looking string that may be wrong by 1-2 characters.

A police ANPR system rarely gets only one frame. A vehicle typically
appears for 5-30+ frames in the camera view. Aggregating those readings
across frames — *vote per character position* — produces a much more
confident final plate than any single read can give you.

This pipeline:
    video frames  ->  YOLO11 plate detect  ->  PaddleOCR per plate
                  ->  tracker (ByteTrack / BoT-SORT / DeepOC-SORT / DeepSORT)
                  ->  per-track char-position voting  ->  final plate text

Trackers supported
------------------
- bytetrack  : motion-only, fastest, weakest across occlusions
- botsort    : motion + appearance (ReID), default for moving cameras
- deepocsort : motion + appearance + observation-centric recovery
- deepsort   : classic DeepSORT (deep_sort_realtime package, own impl)

Usage
-----
    set VIDEO_PATH below, then:
        python build_video_tracker.py
"""

import os
# PaddleOCR (paddlepaddle 2.6.x) on this Windows venv needs the pure-python
# protobuf backend or its .pb2.py files fail to register. Must be set
# BEFORE paddle imports.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import re
import time
import logging
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# =====================================================================
# Config — edit these or pass via CLI flags
# =====================================================================
YOLO11_PT         = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt")
DEFAULT_VIDEO     = Path(r"C:/Users/gsash/Downloads/test/New folder")  # will look for .mp4 here
OUT_ROOT          = Path(r"C:/Users/gsash/Downloads/test/New folder")

# Tracking / detection
YOLO_CONF         = 0.25        # YOLO plate detection confidence
TRACKER           = "botsort"   # bytetrack | botsort | deepocsort | deepsort
TRACK_BUFFER      = 30          # frames a track survives without detection
PROCESS_EVERY     = 1           # run on every Nth frame (1 = every frame)

# OCR
OCR_CONF_THR      = 0.30        # below this, PaddleOCR line is dropped
PAD_PCT           = 0.50        # bbox padding for plate crop (50% = sweet spot)
MIN_PLATE_CHARS   = 4           # ignore OCR reads shorter than this

# Output
WRITE_VIDEO       = True        # also write an .mp4 with bbox + track-id + final plate
MAX_FRAMES        = None        # set to int to cap (debug); None = no cap

# =====================================================================
# Indian-plate knowledge
# =====================================================================
INDIAN_STD_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
INDIAN_BH_RE  = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HR","HP",
    "JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TS","TR","UP","UK","WB","BH",
}
# Confusion pairs (position-aware correction is the safety net)
LETTER_TO_DIGIT = {"O":"0","I":"1","Z":"2","S":"5","B":"8","A":"4","G":"6","T":"7","Q":"0"}
DIGIT_TO_LETTER = {v:k for k,v in LETTER_TO_DIGIT.items()}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("video-tracker-anpr")


# =====================================================================
# Helpers
# =====================================================================
def clean(s: str) -> str:
    """Uppercase, alnum only."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def is_indian_plate(s: str) -> bool:
    return bool(INDIAN_STD_RE.match(s) or INDIAN_BH_RE.match(s))


# Standard Indian plate layouts (clean() removes spaces, so BH becomes 10 chars).
# Index by character position to expected slot class. Multiple combos can sum to
# the same length, so we pick the most common shape per length.
#   length 7 : 2L+1D+1L+3D         (e.g. HR2A123)
#   length 8 : 2L+2D+1L+3D         (e.g. HR26A123)        — most common 8-char
#   length 9 : 2L+2D+2L+3D         (e.g. HR26DK123)
#   length 10: 2L+2D+2L+4D         (e.g. HR26DK1234)      — most common 10-char
#   length 11: 2L+2D+3L+4D         (e.g. HR26DKAA1234)
LAYOUT_STD = {
    7:  "LLDLDDD",
    8:  "LLDDLDDD",
    9:  "LLDDLLDDD",
    10: "LLDDLLDDDD",
    11: "LLDDLLLDDDD",
}
# BH (Bharat) series: NN BH NNNN AA → after clean() = "NNBHNNNNAA" = 10 chars
LAYOUT_BH_10 = "DDLLDDDDLL"   # 22 BH 1234 AB


def _slot_class(pos: int, total_len: int, has_bh: bool) -> str:
    """Return 'L' (letter) or 'D' (digit) for the character at `pos`
    in a `total_len`-char plate. Used for position-aware correction."""
    if has_bh:
        layout = LAYOUT_BH_10 if total_len == 10 else (LAYOUT_BH_9 if total_len == 9 else LAYOUT_BH_10)
        if pos < len(layout): return layout[pos]
        return "?"
    layout = LAYOUT_STD.get(total_len, LAYOUT_STD[9])  # default to 9-char
    if pos < len(layout): return layout[pos]
    return "?"


def correct_position_aware(text: str) -> tuple[str, list[str]]:
    """Apply slot-class correction. Returns (corrected, swaps_applied)."""
    if not text: return text, []
    if is_indian_plate(text): return text, []
    has_bh = "BH" in text[2:5] if len(text) >= 5 else False
    out = list(text)
    swaps = []
    for i, c in enumerate(out):
        slot = _slot_class(i, len(text), has_bh)
        if slot == "D" and c in LETTER_TO_DIGIT:
            out[i] = LETTER_TO_DIGIT[c]; swaps.append(f"{i}:{c}->{out[i]}")
        elif slot == "L" and c in DIGIT_TO_LETTER:
            out[i] = DIGIT_TO_LETTER[c]; swaps.append(f"{i}:{c}->{out[i]}")
    return "".join(out), swaps


def enhance_plate_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + adaptive threshold + upscale. Good general PaddleOCR input."""
    if crop_bgr is None or crop_bgr.size == 0: return crop_bgr
    h, w = crop_bgr.shape[:2]
    if max(h, w) < 160:
        s = 200.0 / max(h, 1)
        crop_bgr = cv2.resize(crop_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
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


# =====================================================================
# Track management + voting
# =====================================================================
class TrackHistory:
    """Per-track rolling history of (frame_idx, ocr_text, ocr_conf, yolo_conf, bbox)."""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.reads: list[dict] = []   # {frame, text, conf, yolo_conf, bbox}
        self.first_frame = None
        self.last_frame = None

    def add(self, frame_idx: int, text: str, conf: float, yolo_conf: float, bbox: list):
        if not text: return
        if self.first_frame is None: self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.reads.append({
            "frame": frame_idx, "text": text, "conf": float(conf),
            "yolo_conf": float(yolo_conf), "bbox": [int(v) for v in bbox],
        })

    def vote(self) -> dict:
        """Character-position weighted voting across all reads.

        For each character position, weight = OCR confidence.
        Pick the char with the highest total weight at that position.
        """
        reads = [r for r in self.reads if len(r["text"]) >= MIN_PLATE_CHARS]
        if not reads:
            return {
                "plate": "", "conf": 0.0, "valid": False,
                "method": "vote", "num_reads": 0,
                "votes": [], "position_consensus": [],
            }

        # Pad all reads to the same length (= mode of length, biased to longest valid)
        lens = [len(r["text"]) for r in reads]
        # Pick target length: max of (mode, median) — favor the common shape
        target_len = Counter(lens).most_common(1)[0][0]
        if target_len < 8:
            target_len = max(lens)  # plates are 8-10 chars; don't truncate to short misreads

        # Build per-position char->weight map
        char_weights = [defaultdict(float) for _ in range(target_len)]
        skipped = 0
        for r in reads:
            t = r["text"]
            w = max(r["conf"], 0.01) * max(r["yolo_conf"], 0.01)
            for i in range(target_len):
                if i < len(t):
                    ch = t[i]
                    if ch.isalnum():
                        char_weights[i][ch] += w
                else:
                    skipped += 1
        # If a position was never filled by any read (target_len > all reads), drop it
        while char_weights and not char_weights[-1]:
            char_weights.pop()

        # Pick winner per position
        plate_chars = []
        position_consensus = []
        for i, cw in enumerate(char_weights):
            if not cw:
                plate_chars.append("?")
                position_consensus.append({"pos": i, "winner": "?", "score": 0.0, "options": {}})
                continue
            total = sum(cw.values())
            winner, win_score = max(cw.items(), key=lambda x: x[1])
            plate_chars.append(winner)
            position_consensus.append({
                "pos": i, "winner": winner, "score": float(win_score),
                "options": {k: float(v) for k, v in sorted(cw.items(), key=lambda x: -x[1])},
                "consensus": win_score / total if total > 0 else 0.0,
            })
        voted = "".join(plate_chars)
        # Apply position-aware correction as safety net (only if vote didn't validate)
        if is_indian_plate(voted):
            final = voted
            swaps = []
        else:
            final, swaps = correct_position_aware(voted)
            if not is_indian_plate(final):
                # Try again on the corrected version - sometimes one pass flips a slot
                final, swaps2 = correct_position_aware(final)
                swaps += swaps2

        # Final confidence: average per-position consensus
        if position_consensus:
            avg_consensus = sum(p["consensus"] for p in position_consensus) / len(position_consensus)
        else:
            avg_consensus = 0.0
        # Blend with mean OCR conf so high-conf single reads still score well
        mean_ocr = sum(r["conf"] for r in reads) / len(reads)
        final_conf = 0.6 * avg_consensus + 0.4 * mean_ocr

        return {
            "plate": final,
            "conf": float(final_conf),
            "raw_vote": voted,
            "valid": is_indian_plate(final),
            "method": "char_position_vote" + ("+correction" if swaps else ""),
            "num_reads": len(reads),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "swaps_applied": swaps,
            "position_consensus": position_consensus,
            "all_reads": reads,
        }


# =====================================================================
# Tracker wrappers (ultralytics built-ins + deep_sort_realtime for DeepSORT)
# =====================================================================
class DeepSortWrapper:
    """Wrap deep_sort_realtime.DeepSort to look like the ultralytics
    trackers: takes (frame, [(bbox_xyxy, conf, cls)]) -> [(bbox_xyxy, track_id)]."""

    def __init__(self, max_age=30, n_init=2, max_iou_distance=0.7,
                 embedder="mobilenet", half=True):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(
            max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance,
            embedder=embedder, half=half, bgr=True,
        )
        self.max_age = max_age

    def update_tracks(self, detections, frame=None):
        """detections: list of (xyxy_tuple, conf, cls) ; returns list of
        (xyxy, track_id_or_None, is_confirmed)."""
        # deep_sort_realtime wants [(xywh_left_top, conf, cls), ...]
        ds_dets = []
        for (x1, y1, x2, y2), conf, cls in detections:
            w, h = x2 - x1, y2 - y1
            ds_dets.append(([x1, y1, w, h], conf, cls))
        tracks = self.tracker.update_tracks(ds_dets, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed(): continue
            if t.time_since_update > 0: continue
            l, t_, r, b = t.to_ltrb()
            out.append(([l, t_, r, b], int(t.track_id), True))
        return out


def make_tracker(name: str, frame_rate: int = 30):
    """Returns an object with .update_tracks(detections, frame) method.
    detections: list of (xyxy_list, conf, cls)
    returns: list of (xyxy_list, track_id, is_confirmed)
    """
    name = name.lower()
    if name == "deepsort":
        log.info("Tracker: DeepSORT (deep_sort_realtime, mobilenet embedder)")
        return DeepSortWrapper(max_age=TRACK_BUFFER, n_init=2)
    # ultralytics built-ins
    yaml_map = {
        "bytetrack":  "bytetrack.yaml",
        "botsort":    "botsort.yaml",
        "deepocsort": "deepocsort.yaml",
    }
    if name not in yaml_map:
        raise ValueError(f"Unknown tracker: {name}. Choose from: bytetrack, botsort, deepocsort, deepsort")
    # Edit the yaml to set track_buffer
    import yaml as _yaml, tempfile
    cfg_root = Path(os.path.dirname(__import__("ultralytics").__file__)) / "cfg" / "trackers"
    base = _yaml.safe_load((cfg_root / yaml_map[name]).read_text())
    base["track_buffer"] = TRACK_BUFFER
    tmp = Path(tempfile.gettempdir()) / f"tracker_{name}_{int(time.time())}.yaml"
    tmp.write_text(_yaml.dump(base))
    log.info(f"Tracker: {name}  (cfg={tmp.name}, track_buffer={TRACK_BUFFER})")
    return _YamlTracker(name, str(tmp), frame_rate)


class _YamlTracker:
    """Wraps ultralytics' built-in trackers via a yaml config file."""
    def __init__(self, name: str, yaml_path: str, frame_rate: int):
        self.name = name
        self.yaml_path = yaml_path
        self.frame_rate = frame_rate

    def update_tracks(self, detections, frame=None):
        """For ultralytics trackers we have to use the model.track() call
        with persist=True. So the real update happens in the model, not here.
        This wrapper is just a marker for code clarity."""
        raise NotImplementedError("ultralytics trackers are updated via model.track() in the main loop")


# =====================================================================
# OCR wrapper
# =====================================================================
class PaddleOCREngine:
    _instance = None
    @classmethod
    def get(cls):
        if cls._instance is None:
            log.info("Initialising PaddleOCR (mobile, English, angle-cls)...")
            t0 = time.time()
            cls._instance = PaddleOCR(
                use_angle_cls=True, lang="en", show_log=False,
                use_gpu=False, enable_mkldnn=False, cpu_threads=4,
            )
            log.info(f"PaddleOCR ready in {time.time()-t0:.1f}s")
        return cls._instance

    def read(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0: return "", 0.0
        try:
            result = self.get().ocr(crop_bgr, cls=True, det=True, rec=True)
        except Exception as e:
            log.debug(f"PaddleOCR exception: {e}")
            return "", 0.0
        if not result or not result[0]: return "", 0.0
        # Pick the best line by confidence
        best_text, best_conf = "", 0.0
        for line in result[0]:
            if not line or len(line) < 2: continue
            box, (text, conf) = line
            if conf < OCR_CONF_THR: continue
            t = clean(text)
            if len(t) < MIN_PLATE_CHARS: continue
            if conf > best_conf:
                best_text, best_conf = t, conf
        return best_text, best_conf


# =====================================================================
# Main pipeline
# =====================================================================
def crop_with_pad(frame: np.ndarray, xyxy: list, pad_pct: float) -> tuple[np.ndarray, list]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(bw * pad_pct), 8)
    pad_y = max(int(bh * pad_pct), 6)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    return crop, [cx1, cy1, cx2, cy2]


def discover_video(path: Path) -> Path:
    """If path is a directory, look for the first video file inside."""
    if path.is_file(): return path
    if path.is_dir():
        for ext in ("*.mp4", "*.MP4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.m4v"):
            found = sorted(path.glob(ext))
            if found: return found[0]
    raise FileNotFoundError(f"No video file at {path}")


def run(video_path: Path, out_dir: Path, tracker_name: str, write_video: bool,
        max_frames=None, process_every=1):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated_frames"
    annotated_dir.mkdir(exist_ok=True)
    tracks_dir = out_dir / "tracks"
    tracks_dir.mkdir(exist_ok=True)

    # Load detector
    log.info(f"Loading YOLO11 plate detector: {YOLO11_PT}")
    model = YOLO(str(YOLO11_PT))

    # Build tracker cfg
    import yaml as _yaml, tempfile
    yaml_map = {
        "bytetrack":  "bytetrack.yaml",
        "botsort":    "botsort.yaml",
        "deepocsort": "deepocsort.yaml",
    }
    if tracker_name == "deepsort":
        # Use ultralytics to DETECT and TRACK with bytetrack for bbox association,
        # then run a parallel deep-sort for appearance-based re-association.
        # Simpler: use bytetrack for bbox track-id, then refine with deep_sort
        # for ID consistency at end of video. For the demo we use bytetrack for
        # primary association and deep_sort as a post-process re-id pass.
        tracker_yaml = None
    else:
        cfg_root = Path(os.path.dirname(__import__("ultralytics").__file__)) / "cfg" / "trackers"
        base = _yaml.safe_load((cfg_root / yaml_map[tracker_name]).read_text())
        base["track_buffer"] = TRACK_BUFFER
        ty = Path(tempfile.gettempdir()) / f"tracker_{tracker_name}_{int(time.time())}.yaml"
        ty.write_text(_yaml.dump(base))
        tracker_yaml = str(ty)
    log.info(f"Tracker: {tracker_name}  track_buffer={TRACK_BUFFER}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Video: {video_path.name}  {w}x{h}  {fps:.1f}fps  {total_frames} frames")

    # Video writer
    vw = None
    if write_video:
        out_mp4 = out_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
        log.info(f"Annotated video -> {out_mp4}")

    ocr = PaddleOCREngine()
    # Per-track history keyed by track_id
    tracks: dict[int, TrackHistory] = {}
    # last seen bbox for each track (so we can show "stale" track too)
    last_bbox: dict[int, list] = {}
    # appearance crops (best crop per track) for the HTML gallery
    best_crops: dict[int, tuple[float, np.ndarray, int, str]] = {}  # conf, crop, frame, text

    # DeepSORT secondary: if requested, keep a parallel tracker for appearance-based re-id
    deepsort = None
    if tracker_name == "deepsort":
        deepsort = DeepSortWrapper(max_age=TRACK_BUFFER, n_init=2)

    frame_idx = 0
    t_start = time.time()
    log.info("Processing frames...")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if max_frames and frame_idx >= max_frames: break
        if frame_idx % process_every != 0:
            frame_idx += 1
            continue

        # 1) Detect plates (YOLO)
        if tracker_name != "deepsort":
            # Use ultralytics tracker (built-in)
            res = model.track(frame, persist=True, tracker=tracker_yaml,
                              conf=YOLO_CONF, verbose=False)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                ids = res.boxes.id
                for i, b in enumerate(res.boxes):
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0])
                    tid = int(ids[i]) if ids is not None else None
                    dets.append(([x1, y1, x2, y2], conf, tid))
        else:
            # DeepSORT path: detect with YOLO, then associate via deep-sort
            res = model.predict(frame, conf=YOLO_CONF, verbose=False)[0]
            dets_for_ds = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0])
                    dets_for_ds.append(([x1, y1, x2, y2], conf, 0))
            ds_out = deepsort.update_tracks(dets_for_ds, frame=frame)
            dets = [(bb, 0.99, tid) for (bb, tid, _ok) in ds_out]

        # 2) OCR each detection, accumulate into tracks
        for (x1, y1, x2, y2), yconf, tid in dets:
            if tid is None: continue
            xyxy = [x1, y1, x2, y2]
            crop, padded_xyxy = crop_with_pad(frame, xyxy, PAD_PCT)
            text, ocrconf = ocr.read(enhance_plate_crop(crop))
            if not text: continue
            if tid not in tracks:
                tracks[tid] = TrackHistory(tid)
            tracks[tid].add(frame_idx, text, ocrconf, yconf, padded_xyxy)
            last_bbox[tid] = padded_xyxy
            # Track best crop (highest ocr conf * yolo conf)
            score = ocrconf * yconf
            if tid not in best_crops or score > best_crops[tid][0]:
                best_crops[tid] = (score, crop.copy(), frame_idx, text)

        # 3) Draw on frame (we draw AFTER processing this frame's reads)
        drawn = frame.copy()
        for tid, t in tracks.items():
            bbox = last_bbox.get(tid)
            if bbox is None: continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Run live vote so the label updates as frames come in
            live = t.vote()
            label = f"#{tid} {live['plate'] or '...'}  c={live['conf']:.2f}"
            color = (0, 255, 0) if live["valid"] else (0, 200, 255)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(drawn, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(drawn, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Stamp frame info top-left
        cv2.putText(drawn, f"frame {frame_idx}/{total_frames}  tracker={tracker_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(drawn, f"frame {frame_idx}/{total_frames}  tracker={tracker_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        if vw: vw.write(drawn)
        # Save every 10th frame as a still (for the HTML gallery)
        if frame_idx % 10 == 0:
            cv2.imwrite(str(annotated_dir / f"frame_{frame_idx:06d}.jpg"), drawn,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_idx / max(elapsed, 0.001)
            eta = (total_frames - frame_idx) / max(fps_actual, 0.001)
            log.info(f"  frame {frame_idx}/{total_frames}  {fps_actual:.1f}fps  "
                     f"ETA {eta:.0f}s  active_tracks={len(tracks)}")

    cap.release()
    if vw: vw.release()

    log.info(f"Processed {frame_idx} frames in {time.time()-t_start:.1f}s")
    log.info(f"Found {len(tracks)} tracks")

    # 4) Per-track voting + JSON outputs
    summary_tracks = []
    for tid, t in sorted(tracks.items()):
        v = t.vote()
        if v["num_reads"] == 0: continue
        track_dir = tracks_dir / f"track_{tid:03d}"
        track_dir.mkdir(exist_ok=True)
        # Save best crop
        if tid in best_crops:
            _, crop, bframe, btext = best_crops[tid]
            # Upscale 3x for visibility
            h, w = crop.shape[:2]
            big = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(track_dir / f"best_crop_frame{bframe}.png"), big)
        # Save all frame-reads (small JSON)
        with open(track_dir / "reads.json", "w") as f:
            json.dump({
                "track_id": tid,
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "num_reads": v["num_reads"],
                "best_plate": v["plate"],
                "best_conf": v["conf"],
                "valid": v["valid"],
                "method": v["method"],
                "swaps_applied": v.get("swaps_applied", []),
                "position_consensus": v["position_consensus"],
                "reads": v["all_reads"],
            }, f, indent=2)
        summary_tracks.append({
            "track_id": tid,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "num_reads": v["num_reads"],
            "best_plate": v["plate"],
            "raw_vote": v["raw_vote"],
            "conf": v["conf"],
            "valid": v["valid"],
            "method": v["method"],
            "swaps_applied": v.get("swaps_applied", []),
            "best_frame": best_crops[tid][2] if tid in best_crops else None,
        })

    # 5) Write summary.json
    summary = {
        "video": str(video_path),
        "tracker": tracker_name,
        "track_buffer": TRACK_BUFFER,
        "yolo_conf": YOLO_CONF,
        "pad_pct": PAD_PCT,
        "total_frames": frame_idx,
        "elapsed_sec": round(time.time() - t_start, 2),
        "fps_actual": round(frame_idx / max(time.time() - t_start, 0.001), 2),
        "num_tracks": len(summary_tracks),
        "num_valid_plates": sum(1 for t in summary_tracks if t["valid"]),
        "tracks": summary_tracks,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {out_dir / 'summary.json'}")

    # 6) HTML viewer
    write_html_viewer(out_dir, summary, tracks, best_crops)
    log.info(f"Wrote {out_dir / 'viewer.html'}")

    return summary


# =====================================================================
# HTML viewer
# =====================================================================
def write_html_viewer(out_dir: Path, summary: dict, tracks: dict,
                      best_crops: dict):
    """Build a self-contained HTML report with:
        - video + per-track table
        - per-track plate detail with frame-by-frame vote breakdown
        - best-plate gallery (upscale 3x, clickable)
    """
    out_dir = Path(out_dir)
    annotated_dir = out_dir / "annotated_frames"
    tracks_dir = out_dir / "tracks"

    # Per-track sections with all reads
    track_sections = []
    for st in summary["tracks"]:
        tid = st["track_id"]
        reads_path = tracks_dir / f"track_{tid:03d}" / "reads.json"
        if not reads_path.exists(): continue
        rd = json.loads(reads_path.read_text())
        # Per-position vote bar (small inline SVG)
        pos_html = []
        for p in rd["position_consensus"]:
            ch = p["winner"]; cons = p["consensus"]
            opts = " ".join(f"{k}({v:.2f})" for k, v in list(p["options"].items())[:3])
            color = "#2ecc71" if cons > 0.8 else "#f39c12" if cons > 0.5 else "#e74c3c"
            pos_html.append(
                f'<span class="pos" style="border-color:{color}" title="{opts}">'
                f'<b>{ch}</b><sub>{cons:.2f}</sub></span>')
        positions = "".join(pos_html)

        # Per-frame reads table
        rows = []
        for r in rd["reads"]:
            t = r["text"]; c = r["conf"]; yc = r["yolo_conf"]
            # Mark which char-position agrees with the final vote
            agree = sum(1 for i, ch in enumerate(t) if i < len(rd["best_plate"])
                        and rd["best_plate"][i] == ch)
            total = min(len(t), len(rd["best_plate"]))
            agree_pct = (agree / total * 100) if total else 0
            color = "#2ecc71" if agree_pct > 80 else "#f39c12" if agree_pct > 50 else "#e74c3c"
            rows.append(
                f'<tr>'
                f'<td>{r["frame"]}</td>'
                f'<td><code>{t}</code></td>'
                f'<td>{c:.2f}</td>'
                f'<td>{yc:.2f}</td>'
                f'<td><span class="agree" style="background:{color}">{agree_pct:.0f}%</span></td>'
                f'</tr>')
        reads_table = "\n".join(rows)

        # Best crop
        best_crop_html = ""
        for p in (tracks_dir / f"track_{tid:03d}").glob("best_crop_*.png"):
            rel = os.path.relpath(p, out_dir).replace("\\", "/")
            best_crop_html = (
                f'<a href="{rel}" target="_blank">'
                f'<img class="best" src="{rel}" alt="best crop"></a>')
            break

        valid_badge = ("<span class='ok'>VALID</span>" if st["valid"]
                       else "<span class='warn'>not valid</span>")
        method = st["method"]
        swaps = st.get("swaps_applied") or []
        swaps_html = ("<br>swaps: " + ", ".join(swaps)) if swaps else ""

        track_sections.append(f"""
<section class="track">
  <h2>Track #{tid}  {valid_badge}
    <span class="plate">{st['best_plate'] or '(no plate)'}</span>
    <span class="conf">conf {st['conf']:.2f}</span>
  </h2>
  <div class="meta">
    frames {st['first_frame']}–{st['last_frame']}  ·
    {st['num_reads']} reads  ·  method: <code>{method}</code>{swaps_html}
  </div>
  <div class="positions"><b>Position votes:</b> {positions}</div>
  <div class="crop">{best_crop_html}</div>
  <details>
    <summary>frame-by-frame reads ({st['num_reads']})</summary>
    <table>
      <thead><tr><th>frame</th><th>OCR</th><th>OCR-conf</th><th>YOLO-conf</th><th>agree with final</th></tr></thead>
      <tbody>{reads_table}</tbody>
    </table>
  </details>
</section>""")

    # Per-track summary table
    table_rows = []
    for st in summary["tracks"]:
        valid = "✓" if st["valid"] else "✗"
        color = "#2ecc71" if st["valid"] else "#e74c3c"
        table_rows.append(
            f'<tr><td>#{st["track_id"]}</td>'
            f'<td><code>{st["best_plate"] or "-"}</code></td>'
            f'<td>{st["conf"]:.2f}</td>'
            f'<td>{st["num_reads"]}</td>'
            f'<td>{st["first_frame"]}–{st["last_frame"]}</td>'
            f'<td style="color:{color}">{valid}</td></tr>')
    table = "\n".join(table_rows)

    # Gallery of best crops
    gallery = []
    for st in summary["tracks"]:
        tid = st["track_id"]
        for p in (tracks_dir / f"track_{tid:03d}").glob("best_crop_*.png"):
            rel = os.path.relpath(p, out_dir).replace("\\", "/")
            gallery.append(
                f'<a class="g" href="#track_{tid}">'
                f'<img src="{rel}" alt="t{tid}">'
                f'<span>#{tid} <code>{st["best_plate"] or "-"}</code></span></a>')
            break

    rel_video = "annotated_video.mp4"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>ANPR video tracker report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
  h1, h2 {{ color: #fff; }}
  code {{ background: #333; padding: 2px 6px; border-radius: 3px; color: #6cf; }}
  .summary {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .summary .box {{ background: #2a2a2a; padding: 12px 20px; border-radius: 6px; min-width: 140px; }}
  .summary .box b {{ display: block; font-size: 28px; color: #6cf; }}
  video {{ max-width: 100%; background: #000; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; background: #2a2a2a; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #444; font-size: 14px; }}
  th {{ background: #333; color: #fff; }}
  .track {{ background: #2a2a2a; border-radius: 8px; padding: 16px; margin: 20px 0; }}
  .track h2 .plate {{ color: #6cf; font-family: monospace; font-size: 26px; margin: 0 12px; }}
  .track h2 .conf {{ color: #aaa; font-size: 16px; }}
  .track .meta {{ color: #aaa; font-size: 13px; margin: 4px 0; }}
  .positions {{ margin: 10px 0; }}
  .pos {{ display: inline-block; border: 2px solid; padding: 4px 8px; margin: 2px;
          font-family: monospace; font-size: 18px; min-width: 22px; text-align: center; }}
  .pos sub {{ font-size: 10px; color: #aaa; display: block; text-align: center; }}
  .crop img.best {{ max-width: 360px; border: 1px solid #555; margin: 6px 0; }}
  .agree {{ padding: 2px 6px; border-radius: 3px; color: #000; font-weight: bold; }}
  .ok {{ background: #2ecc71; color: #000; padding: 2px 8px; border-radius: 3px; }}
  .warn {{ background: #e74c3c; color: #000; padding: 2px 8px; border-radius: 3px; }}
  details {{ margin-top: 8px; }}
  details summary {{ cursor: pointer; color: #6cf; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }}
  .gallery .g {{ display: block; text-align: center; background: #2a2a2a; padding: 6px;
                  border-radius: 6px; text-decoration: none; color: #eee; }}
  .gallery img {{ max-width: 240px; max-height: 80px; display: block; margin: 0 auto 4px; }}
</style></head><body>
<h1>ANPR — Video Tracker Voting</h1>
<div class="summary">
  <div class="box">Tracker<b>{summary['tracker']}</b></div>
  <div class="box">Total frames<b>{summary['total_frames']}</b></div>
  <div class="box">Elapsed<b>{summary['elapsed_sec']}s</b></div>
  <div class="box">FPS<b>{summary['fps_actual']}</b></div>
  <div class="box">Tracks<b>{summary['num_tracks']}</b></div>
  <div class="box">Valid plates<b style="color:#2ecc71">{summary['num_valid_plates']}</b></div>
</div>

<h2>Annotated video</h2>
<video controls src="{rel_video}"></video>

<h2>Per-track summary</h2>
<table>
  <thead><tr><th>track</th><th>best plate</th><th>conf</th><th>reads</th><th>frames</th><th>valid?</th></tr></thead>
  <tbody>{table}</tbody>
</table>

<h2>Best-plate gallery</h2>
<div class="gallery">{''.join(gallery)}</div>

<h2>Per-track detail</h2>
{''.join(track_sections)}
</body></html>"""

    (out_dir / "viewer.html").write_text(html, encoding="utf-8")


# =====================================================================
# CLI
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default=None,
                   help="Path to video file. If omitted, looks for first .mp4 in DEFAULT_VIDEO dir.")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory. Default: <video dir>/anpr_video_<ts>")
    p.add_argument("--tracker", type=str, default=TRACKER,
                   choices=["bytetrack", "botsort", "deepocsort", "deepsort"])
    p.add_argument("--no-video", action="store_true", help="Skip writing annotated .mp4")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    p.add_argument("--every", type=int, default=PROCESS_EVERY,
                   help="Process every Nth frame (default 1 = every frame)")
    args = p.parse_args()

    video_arg = Path(args.video) if args.video else DEFAULT_VIDEO
    video = discover_video(video_arg)
    log.info(f"Using video: {video}")

    if args.out:
        out = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = video.parent / f"anpr_video_{args.tracker}_{ts}"

    summary = run(video, out, args.tracker, not args.no_video,
                  args.max_frames, args.every)

    print("\n" + "=" * 60)
    print(f"TRACKER   : {summary['tracker']}")
    print(f"FRAMES    : {summary['total_frames']}  ({summary['elapsed_sec']}s, "
          f"{summary['fps_actual']} fps)")
    print(f"TRACKS    : {summary['num_tracks']}  "
          f"(valid plates: {summary['num_valid_plates']})")
    print("-" * 60)
    for t in summary["tracks"]:
        flag = "✓" if t["valid"] else "✗"
        print(f"  #{t['track_id']:>2}  {t['best_plate'] or '(no plate)':<14}  "
              f"conf={t['conf']:.2f}  reads={t['num_reads']:>2}  "
              f"frames {t['first_frame']:>4}–{t['last_frame']:<4}  {flag}")
    print("=" * 60)
    print(f"Output   : {out}")
    print(f"Viewer   : {out/'viewer.html'}")


if __name__ == "__main__":
    main()
