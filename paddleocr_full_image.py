"""PaddleOCR (full-image, no YOLO, no crop) on every test image.

Reads each image in C:/Users/gsash/Downloads/test/New folder and runs
PaddleOCR mobile-EN on the WHOLE image (no plate detector, no crop).
Filters OCR lines for Indian-plate-like text and reports the best candidate
plus everything else it saw.

Writes:
  paddleocr_full_<ts>.json   - per-image structured results
  paddleocr_full_<ts>.txt    - human-readable table
"""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUTPUT_DIR = INPUT_DIR  # save report in same folder per user instruction

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Indian-plate regex (broad) — used only to *rank* candidates, not to reject.
INDIAN_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
INDIAN_BH_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HR","HP",
    "JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TS","TR","UP","UK","WB","BH",
}


def clean(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def is_indian_plate(s: str) -> bool:
    s = clean(s)
    return bool(INDIAN_RE.match(s) or INDIAN_BH_RE.match(s))


def pick_best_plate(lines):
    """Among PaddleOCR lines, return the best plate-like candidate.

    Score = valid_format (Indian plate regex) is the strongest signal,
    followed by confidence, then by length (8-10 chars is plate-shaped).
    """
    if not lines:
        return None
    candidates = []
    for ln in lines:
        text = clean(ln["text"])
        if not text:
            continue
        valid = is_indian_plate(text)
        # plates are 7-10 chars; penalize very short/long
        len_score = 1.0 if 7 <= len(text) <= 10 else 0.0
        score = (2.0 if valid else 0.0) + ln["conf"] + 0.3 * len_score
        candidates.append({"text": text, "raw": ln["text"], "conf": ln["conf"],
                            "valid_format": valid, "score": round(score, 3)})
    if not candidates:
        return None
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[PaddleOCR-full] init mobile-EN (CPU, angle-cls) ...")
    t0 = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False,
                    use_gpu=False, det_db_thresh=0.3, det_db_box_thresh=0.5)
    print(f"[PaddleOCR-full] ready in {time.time()-t0:.1f}s")

    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT)
    print(f"[PaddleOCR-full] {len(images)} images in {INPUT_DIR}")

    out = {
        "engine": "PaddleOCR 2.7 (mobile, en, angle-cls, CPU)",
        "input_dir": str(INPUT_DIR),
        "started": datetime.now().isoformat(timespec="seconds"),
        "mode": "FULL IMAGE — no YOLO, no crop, no detector",
        "per_image": [],
    }

    total_seconds = 0.0
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx}/{len(images)}] SKIP unreadable: {img_path.name}")
            out["per_image"].append({"image": img_path.name, "error": "unreadable"})
            continue

        t0 = time.time()
        try:
            res = ocr.ocr(img, cls=True)
        except Exception as e:
            out["per_image"].append({"image": img_path.name, "error": str(e)})
            continue
        dt = time.time() - t0
        total_seconds += dt

        lines = []
        if res and res[0]:
            for line in res[0]:
                try:
                    bbox, (text, conf) = line
                except Exception:
                    continue
                lines.append({"text": text, "conf": round(float(conf), 4),
                              "bbox": [[int(p[0]), int(p[1])] for p in bbox]})

        best = pick_best_plate(lines)

        rec = {
            "image": img_path.name,
            "image_size": list(img.shape[:2][::-1]),
            "ocr_seconds": round(dt, 3),
            "lines_seen": len(lines),
            "lines": lines,
            "best_plate": best,
        }
        out["per_image"].append(rec)

        marker = "✓" if best and best["valid_format"] else ("·" if best else "✗")
        best_text = best["text"] if best else "(no plate-like text)"
        print(f"  [{idx}/{len(images)}] {img_path.name:<18} "
              f"lines={len(lines):>3}  best={best_text:<14} {marker}  ({dt:.2f}s)")

    out["finished"] = datetime.now().isoformat(timespec="seconds")
    out["total_seconds"] = round(total_seconds, 2)

    # Count wins
    n_total = sum(1 for r in out["per_image"] if "error" not in r)
    n_with_plate = sum(1 for r in out["per_image"]
                       if r.get("best_plate") and r["best_plate"]["text"])
    n_valid_format = sum(1 for r in out["per_image"]
                         if r.get("best_plate") and r["best_plate"]["valid_format"])
    out["summary"] = {
        "images": n_total,
        "with_plate_text": n_with_plate,
        "valid_indian_format": n_valid_format,
    }

    json_path = OUTPUT_DIR / f"paddleocr_full_{ts}.json"
    txt_path  = OUTPUT_DIR / f"paddleocr_full_{ts}.txt"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # human-readable table
    lines = []
    lines.append("=" * 90)
    lines.append("  PaddleOCR (mobile, EN, CPU) on FULL IMAGE — no YOLO, no crop, no detector")
    lines.append("=" * 90)
    lines.append(f"  Source          : {INPUT_DIR}")
    lines.append(f"  Images          : {n_total}")
    lines.append(f"  Paddle saw text : {n_with_plate}/{n_total}")
    lines.append(f"  Valid IN format : {n_valid_format}/{n_total}")
    lines.append(f"  Total time      : {total_seconds:.1f}s")
    lines.append("-" * 90)
    lines.append(f"  {'IMAGE':<18}  {'LINES':>5}  {'BEST':<16}  {'CONF':>5}  FORMAT")
    lines.append("-" * 90)
    for r in out["per_image"]:
        if "error" in r:
            lines.append(f"  {r['image']:<18}  ERROR: {r['error']}")
            continue
        bp = r.get("best_plate")
        if bp:
            txt = bp["text"]
            conf = bp["conf"]
            fmt = "YES" if bp["valid_format"] else "no"
        else:
            txt = "(no plate-like)"
            conf = 0.0
            fmt = "—"
        lines.append(f"  {r['image']:<18}  {r['lines_seen']:>5}  "
                     f"{txt:<16}  {conf:>5.2f}  {fmt}")
    lines.append("=" * 90)
    txt_path.write_text("\n".join(lines))

    print()
    print(f"[PaddleOCR-full] DONE. {n_with_plate}/{n_total} plates seen, "
          f"{n_valid_format} valid IN format, {total_seconds:.1f}s")
    print(f"[PaddleOCR-full] JSON : {json_path}")
    print(f"[PaddleOCR-full] TXT  : {txt_path}")


if __name__ == "__main__":
    main()