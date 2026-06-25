"""
Video ANPR using YOLO11 plate detector + Awiros ANPR-OCR (per-frame).

Pipeline per sampled frame:
  1. YOLO11 plate detector -> list of plate bboxes + detector conf
  2. For each plate crop, run Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)
  3. Track plates across frames with simple IoU-based association
  4. After the video, vote per character-position across each track's reads
     -> final plate text + final confidence per unique plate.

Awiros is a SINGLE-IMAGE recognition model (no temporal context). The
tracker + voting is what gives us the multi-frame boost: same plate seen
20 frames gives 20 OCR reads, and per-position voting collapses the noise.

Usage:
    python anpr_video_awiros.py --video <video.mp4> --stride 2 --max-frames 300
    python anpr_video_awiros.py --video <video.mp4> --stride 5 --no-video

Outputs (next to input video, in <stem>_awiros_<ts>/):
    annotated.mp4     - video with bbox + track-id + per-frame OCR (optional)
    frames/           - one annotated JPG per processed frame
    crops/            - per-track plate crops grouped by track_id
    tracks.json       - per-track detailed votes + final plate text
    summary.json      - top-level stats
    report.html       - human-friendly HTML viewer (lightbox)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# PaddleOCR + protobuf workaround (must be set BEFORE paddle imports).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the AwirosANPR + YOLO11 loader from the image pipeline.
from detect_yolo11_awiros_ocr import (
    AwirosANPR,
    DEFAULT_MODEL as YOLO11_PT,
)

log = logging.getLogger("anpr_video_awiros")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
IMGSZ = 640
DET_CONF = 0.25          # YOLO plate detection confidence threshold
DEVICE = "cpu"
OCR_MIN_HEIGHT = 18       # Awiros on crops below this height usually returns nothing


# ---------------------------------------------------------------------------
# ByteTracker wrapper (ultralytics BYTETracker, SimpleTracker-compatible API)
# ---------------------------------------------------------------------------
class ByteTracker:
    """ByteTrack via ultralytics, exposing the SimpleTracker interface.

    Input:  update(detections) where detections = [(bbox_xyxy, yconf), ...]
    Output: [(tid, bbox_xyxy, yconf), ...] with persistent track IDs across frames.

    Uses ultralytics' built-in BYTETracker (no extra deps beyond ultralytics).
    ByteTrack is motion-only (Kalman + IoU), no ReID model needed — matches our
    Awiros-only, lightweight pipeline.
    """

    class _ResultsView:
        """Minimal Results-like object BYTETracker.update() expects.

        BYTETracker accesses: .conf, .cls, .xyxy (parse_bboxes prefers .xywhr,
        falls back to .xywh). We provide xywh + xyxy and hide xywhr via a
        __getattr__ override that raises AttributeError (so the `hasattr` check
        in parse_bboxes is False and it uses xywh instead).
        """
        def __init__(self, xyxy, conf, cls_):
            xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
            self.xyxy = xyxy
            self.xywh = self._xyxy_to_xywh(xyxy)
            self.conf = np.asarray(conf, dtype=float)
            self.cls  = np.asarray(cls_, dtype=int)
        def __getattr__(self, name):
            raise AttributeError(
                f"_ResultsView has no attribute '{name}'. "
                "BYTETracker expects .xyxy, .xywh, .conf, .cls."
            )
        @staticmethod
        def _xyxy_to_xywh(xyxy):
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            return np.stack([cx, cy, w, h], axis=1)
        def __len__(self): return len(self.conf)
        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return ByteTracker._ResultsView(self.xyxy[key], self.conf[key], self.cls[key])
            raise TypeError("BYTETracker._ResultsView only supports boolean-mask indexing")

    def __init__(self, frame_rate: int = 30, max_age: int = 30,
                 high_thresh: float = 0.5, low_thresh: float = 0.10,
                 match_thresh: float = 0.8):
        """Wrap ultralytics BYTETracker (motion-only, no ReID).

        `frame_rate` is accepted for API symmetry with SimpleTracker and is used
        by callers to size max_age; BYTETracker itself doesn't take a frame_rate.
        """
        import argparse
        from ultralytics.trackers.byte_tracker import BYTETracker as _BYTETracker
        args = argparse.Namespace()
        args.tracker_yaml        = ''
        args.track_high_thresh   = high_thresh
        args.track_low_thresh    = low_thresh
        args.new_track_thresh    = high_thresh + 0.1
        args.track_buffer        = max_age
        args.match_thresh        = match_thresh
        args.fuse_score          = False
        args.min_box_area        = 10
        args.min_consecutive_frames = 1
        self._bt = _BYTETracker(args=args)
        self.max_age = max_age

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / max(ua, 1)

    def update(self, detections: list) -> list:
        """detections: list of (bbox, yconf). Returns list of (tid, bbox, yconf)."""
        if not detections:
            return []
        xyxy = np.array([[*d[0]] for d in detections], dtype=float)
        conf = np.array([d[1] for d in detections], dtype=float)
        cls  = np.zeros(len(detections), dtype=int)  # single class: plate
        results = self._ResultsView(xyxy, conf, cls)
        tracks = self._bt.update(results)  # np.ndarray, rows = [x1,y1,x2,y2,tid,conf,cls,idx]
        out = []
        for t in tracks:
            tx1, ty1, tx2, ty2, tid = float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4])
            # Match track bbox back to the best input detection to keep the
            # original YOLO confidence (BYTETracker's conf column is fused).
            best_iou, best_idx = 0.0, 0
            for i, (bbox, _) in enumerate(detections):
                iou = self._iou_xyxy((tx1, ty1, tx2, ty2), bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            out.append((tid, tuple(detections[best_idx][0]), detections[best_idx][1]))
        return out


# ---------------------------------------------------------------------------
# Character-position voting (same logic as anpr_video_tracker.vote_track_text)
# ---------------------------------------------------------------------------
def vote_track_text(reads: list) -> dict:
    """reads: list of (text, conf) tuples for ONE track across frames.
    Returns: {text, conf, valid, votes_per_pos}
    """
    if not reads:
        return {"text": "", "conf": 0.0, "valid": False, "votes": {}}

    # Length = most-common read length
    len_counter = Counter(len(t) for t, _ in reads)
    best_len, _ = len_counter.most_common(1)[0]

    # Per position: bucket each char by (position, char) -> sum of confs
    pos_buckets = defaultdict(lambda: defaultdict(float))
    for text, conf in reads:
        text = text[:best_len] if len(text) >= best_len else text
        for i, ch in enumerate(text):
            pos_buckets[i][ch] += conf

    chars = []
    confs = []
    votes_per_pos = {}
    for i in range(best_len):
        bucket = pos_buckets.get(i, {})
        if not bucket:
            chars.append("?")
            confs.append(0.0)
            votes_per_pos[i] = {}
            continue
        best_ch, best_score = max(bucket.items(), key=lambda kv: kv[1])
        chars.append(best_ch)
        # Confidence = best_score / total_score across this position
        total = sum(bucket.values())
        confs.append(round(best_score / total, 4) if total > 0 else 0.0)
        votes_per_pos[i] = {ch: round(s, 3) for ch, s in bucket.items()}

    text = "".join(chars)
    # Filter ? chars when computing final conf
    real = [c for c in confs if c > 0]
    final_conf = round(sum(real) / len(real), 4) if real else 0.0
    valid = final_conf >= 0.40 and "?" not in text and len(text) >= 6

    return {"text": text, "conf": final_conf, "valid": valid, "votes": votes_per_pos}


# ---------------------------------------------------------------------------
# Main per-video pipeline
# ---------------------------------------------------------------------------
def _pad_bbox(x1, y1, x2, y2, W, H, pad: float = 0.06):
    """Pad bbox by `pad` fraction on each side, clamped to frame."""
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(W, x2 + px),
        min(H, y2 + py),
    )


def process_video(
    video_path: Path,
    out_dir: Path,
    stride: int = 2,
    max_frames: int = None,
    write_video: bool = True,
    awiros_dir: Path = None,
    yolo_model: str = None,
    device: str = "cpu",
    iou_thresh: float = 0.20,
) -> dict:
    """Run YOLO11 + Awiros on a video, track plates by IoU, vote per position.

    Returns the summary dict (also written to summary.json by caller).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    crops_dir = out_dir / "crops"
    best_frames_dir = out_dir / "best_frames"   # NEW: best-frame annotated crops per track
    frames_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    best_frames_dir.mkdir(exist_ok=True)

    # ── Load models ONCE ──
    from ultralytics import YOLO
    log.info("YOLO11: %s", yolo_model or YOLO11_PT)
    yolo = YOLO(str(yolo_model or YOLO11_PT))
    log.info("Awiros dir: %s", awiros_dir)
    awiros = AwirosANPR(awiros_dir=Path(awiros_dir or HERE / "awiros_anpr"), device=device)
    awiros.load()

    # ── Open video ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Video: %s  %dx%d  fps=%.2f  frames=%d  stride=%d",
             video_path, W, H, fps, total, stride)

    out_video = None
    if write_video:
        out_video_path = out_dir / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))

    tracker = ByteTracker(frame_rate=int(fps), max_age=max(30, fps * 2))

    history = defaultdict(lambda: {
        "reads": [],       # list[(text, conf)]
        "frames": [],      # list[int]
        "bboxes": [],      # list[(x1,y1,x2,y2)]
        "yconfs": [],      # list[float]
        "first_seen": None,
        "last_seen": None,
        # For best-frame picking: the (frame_num, bbox_area, ocr_text, ocr_conf, yconf)
        # of the BEST frame for this track (largest bbox = closest to camera).
        # "best_frame":   int  (frame number)
        # "best_bbox":    (x1,y1,x2,y2)  — bbox of plate in best frame
        # "best_text":    str  — OCR text in best frame
        # "best_conf":    float — OCR conf in best frame
        "best_frame": None,
        "best_bbox": None,
        "best_text": "",
        "best_conf": 0.0,
        # Per-frame annotated crops already saved (used by modal)
        # "crop_files":   list[str]   crop filenames for each frame
        "crop_files": [],
    })

    n_proc = 0
    t0 = time.time()
    last_log_t = t0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_proc += 1
        if max_frames and n_proc > max_frames:
            break
        if stride > 1 and (n_proc - 1) % stride != 0:
            if out_video is not None:
                out_video.write(frame)  # passthrough unprocessed frame
            continue

        # ── Detect plates ──
        det_result = yolo.predict(
            frame, verbose=False, imgsz=IMGSZ, conf=DET_CONF, device=device,
        )[0]
        raw_dets = []
        if det_result.boxes is not None and len(det_result.boxes) > 0:
            for box in det_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                yconf = float(box.conf[0].cpu().item())
                bw, bh = x2 - x1, y2 - y1
                if bh <= 0 or bw <= 0:
                    continue
                ar = bw / bh
                # Aspect-ratio filter: real plates are ~2:1 to ~5:1
                if ar < 1.5 or ar > 6.5:
                    continue
                if bh < 16 or bh > 320:
                    continue
                raw_dets.append(((int(x1), int(y1), int(x2), int(y2)), yconf))

        # ── Track ──
        tracked = tracker.update(raw_dets)

        # ── OCR per tracked plate via Awiros ──
        frame_ocr = []
        for tid, bbox, yconf in tracked:
            x1, y1, x2, y2 = bbox
            x1c, y1c, x2c, y2c = _pad_bbox(x1, y1, x2, y2, W, H, pad=0.06)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            ocr = awiros.predict_crop(crop)
            text = ocr["text"] or ""
            conf = ocr["confidence"]
            history[tid]["reads"].append((text, conf))
            history[tid]["frames"].append(n_proc)
            history[tid]["bboxes"].append(bbox)
            history[tid]["yconfs"].append(yconf)
            if history[tid]["first_seen"] is None:
                history[tid]["first_seen"] = n_proc
            history[tid]["last_seen"] = n_proc
            frame_ocr.append((tid, bbox, text, conf, yconf))

            # Save crop
            crop_fname = f"track{tid:03d}_f{n_proc:06d}.jpg"
            crop_path = crops_dir / crop_fname
            cv2.imwrite(str(crop_path), crop)
            history[tid]["crop_files"].append(crop_fname)

            # Pick best frame for this track: largest bbox area (closest
            # to camera = most readable) with a non-empty OCR read preferred.
            bx1, by1, bx2, by2 = bbox
            area = max(1, bx2 - bx1) * max(1, by2 - by1)
            current_best = history[tid]["best_bbox"]
            best_area = ((current_best[2] - current_best[0]) * (current_best[3] - current_best[1])) if current_best else 0
            prefer = False
            if current_best is None:
                prefer = True
            elif area > best_area * 1.05:
                # Significantly larger bbox (5%+ bigger) — always prefer
                prefer = True
            elif area >= best_area * 0.95 and text and not history[tid]["best_text"]:
                # Roughly same size but new frame has text and old didn't
                prefer = True
            if prefer:
                history[tid]["best_frame"] = n_proc
                history[tid]["best_bbox"] = bbox
                history[tid]["best_text"] = text
                history[tid]["best_conf"] = conf
                # NEW: save FULL annotated frame (entire video frame, not just crop)
                # with bbox + label drawn on it, so the user can compare the OCR
                # result against the actual frame the plate was seen in.
                best_ann = frame.copy()
                bx1, by1, bx2, by2 = bbox
                ann_color = (0, 255, 0) if (text and conf >= 0.20) else (0, 0, 255)
                cv2.rectangle(best_ann, (bx1, by1), (bx2, by2), ann_color, 3)
                ann_label = f"ID{tid} {text or '?'}  Awiros={conf:.2f}"
                (atlw, atlh), _ = cv2.getTextSize(ann_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                aty = max(0, by1 - 10)
                cv2.rectangle(best_ann, (bx1, aty - atlh - 6),
                              (bx1 + atlw + 6, aty + 4), ann_color, -1)
                cv2.putText(best_ann, ann_label, (bx1 + 4, aty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                best_ann_file = f"track{tid:03d}_best_annotated_f{n_proc:06d}.jpg"
                cv2.imwrite(str(best_frames_dir / best_ann_file), best_ann,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                history[tid]["best_annotated_file"] = best_ann_file

        # ── Annotate frame ──
        ann = frame.copy()
        for tid, bbox, text, conf, yconf in frame_ocr:
            tx1, ty1, tx2, ty2 = bbox
            color = (0, 255, 0) if (text and conf >= 0.20) else (0, 0, 255)
            cv2.rectangle(ann, (tx1, ty1), (tx2, ty2), color, 2)
            label = f"ID{tid} {text or '?'} Awiros={conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            ty = max(0, ty1 - 8)
            cv2.rectangle(ann, (tx1, ty - th - 4), (tx1 + tw + 4, ty + 2), color, -1)
            cv2.putText(ann, label, (tx1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(ann, f"frame {n_proc}/{total}  YOLO11 + Awiros ANPR-OCR",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(ann, f"frame {n_proc}/{total}  YOLO11 + Awiros ANPR-OCR",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if out_video is not None:
            out_video.write(ann)
        # Save one annotated frame per second of video (for HTML viewer)
        if n_proc % max(1, int(fps)) == 0 or n_proc == 1:
            cv2.imwrite(str(frames_dir / f"frame_{n_proc:06d}.jpg"), ann)

        # Progress log every ~3 seconds
        if time.time() - last_log_t > 3.0:
            elapsed = time.time() - t0
            log.info("  frame %d/%d  tracks=%d  (%.1fs elapsed)",
                     n_proc, total, len(frame_ocr), elapsed)
            last_log_t = time.time()

    cap.release()
    if out_video is not None:
        out_video.release()

    # ── Vote per track ──
    tracks_summary = []
    for tid in sorted(history.keys()):
        h = history[tid]
        if not h["reads"]:
            continue
        voted = vote_track_text(h["reads"])
        # Per-frame reads: list of {frame, text, conf, yconf, bbox, crop_file}
        per_frame_reads = []
        for i, fnum in enumerate(h["frames"]):
            text, conf = h["reads"][i]
            per_frame_reads.append({
                "frame": int(fnum),
                "text": text,
                "ocr_conf": round(float(conf), 4),
                "yolo_conf": round(float(h["yconfs"][i]), 4),
                "bbox": [int(v) for v in h["bboxes"][i]],
                "crop_file": h["crop_files"][i] if i < len(h["crop_files"]) else "",
            })
        # Best frame — copy that crop into best_frames/ for a clean URL the
        # frontend can fetch without scanning 50+ crops.
        best_frame_num = int(h["best_frame"]) if h["best_frame"] is not None else int(h["first_seen"])
        best_crop_file = ""
        if best_frame_num is not None:
            for i, fnum in enumerate(h["frames"]):
                if int(fnum) == best_frame_num:
                    src = crops_dir / h["crop_files"][i]
                    if src.exists():
                        best_crop_file = f"track{tid:03d}_best_f{best_frame_num:06d}.jpg"
                        dst = best_frames_dir / best_crop_file
                        try:
                            import shutil
                            shutil.copy2(str(src), str(dst))
                        except Exception:
                            best_crop_file = h["crop_files"][i]   # fall back to crops/ dir
                    break
        tracks_summary.append({
            "track_id": int(tid),
            "n_frames": len(h["frames"]),
            "first_seen": int(h["first_seen"]),
            "last_seen": int(h["last_seen"]),
            "frames": h["frames"][:50],  # cap for json size
            "all_frames": h["frames"],   # ALL frame numbers where this track appeared
            "per_frame_reads": per_frame_reads,
            "best_frame": best_frame_num,
            "best_text": h["best_text"] or "",
            "best_conf": round(float(h["best_conf"]), 4),
            "best_crop_file": best_crop_file,
            "best_annotated_file": h.get("best_annotated_file", ""),
            "final_text": voted["text"],
            "final_conf": voted["conf"],
            "valid_indian": voted["valid"],
            "avg_yolo_conf": round(sum(h["yconfs"]) / len(h["yconfs"]), 4),
            "n_unique_reads": len(set(t for t, _ in h["reads"] if t)),
            "votes_per_pos": voted["votes"],
        })

    # Sort by validity desc, then conf desc, then frame count desc
    tracks_summary.sort(key=lambda t: (
        -int(t["valid_indian"]),
        -t["final_conf"],
        -t["n_frames"],
    ))

    elapsed = round(time.time() - t0, 2)
    summary = {
        "video": str(video_path),
        "video_name": video_path.name,
        "n_total_frames": total,
        "n_frames_processed": n_proc,
        "fps": round(fps, 2),
        "stride": stride,
        "tracker": "ByteTracker (ultralytics, max_age={}f, frame_rate={}fps)".format(
            tracker.max_age, int(fps)),
        "elapsed_sec": elapsed,
        "fps_processed": round(n_proc / elapsed, 2) if elapsed > 0 else 0.0,
        "n_tracks": len(tracks_summary),
        "n_valid_plates": sum(1 for t in tracks_summary if t["valid_indian"]),
        "detector": "YOLO11 (yolo11_plate.pt)",
        "ocr": "Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)",
        "device": device,
        "iou_thresh": iou_thresh,
        "tracks": tracks_summary,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done in %.1fs — %d tracks, %d valid plates",
             elapsed, summary["n_tracks"], summary["n_valid_plates"])
    return summary


def main():
    p = argparse.ArgumentParser(description="YOLO11 + Awiros video ANPR (per-frame + IoU tracker + position voting)")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stride", type=int, default=2,
                   help="Process every Nth frame (default=2). Use 5 for ~2.5x speedup.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true", help="Skip writing annotated.mp4")
    p.add_argument("--awiros-dir", type=Path, default=HERE / "awiros_anpr")
    p.add_argument("--yolo-model", type=str, default=str(YOLO11_PT))
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--iou", type=float, default=0.20)
    args = p.parse_args()

    if not args.video.exists():
        log.error("Video not found: %s", args.video)
        return 2

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = args.video.parent / f"{args.video.stem}_awiros_{ts}"

    process_video(
        video_path=args.video,
        out_dir=args.out,
        stride=args.stride,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        awiros_dir=args.awiros_dir,
        yolo_model=args.yolo_model,
        device=args.device,
        iou_thresh=args.iou,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())