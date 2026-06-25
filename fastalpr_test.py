"""
fast-alpr batch test on the 16 traffic-plates test images.

fast-alpr = YOLOv9 detector (built-in) + CCT OCR (built-in, fast-plate-ocr).
This is a single, self-contained pipeline — no project YOLO weights are used.
The script writes:
  - per-image annotated JPGs  (out_root/annotated/<name>.jpg)
  - per-image JSON            (out_root/json/<name>.json)
  - per-plate crops           (out_root/crops/<name>__<idx>__<text>.jpg)
  - summary JSON              (out_root/summary.json)

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "C:/Users/gsash/Downloads/Facial-recognition/venv/Scripts/python.exe" \
        fastalpr_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2

# Force pure-Python protobuf before any paddle/torch import (paranoia; not strictly needed here)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from fast_alpr import ALPR


INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_ROOT = Path(r"C:/Users/gsash/Downloads/traffic-plates/fastalpr_results_20260622_070000")

# Use the lightest detector + lightest OCR (CPU-bound Windows box)
DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"   # tiny CCT, fast on CPU
CONF_THRESH = 0.25

EXT_OK = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(input_dir: Path) -> list[Path]:
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXT_OK
    )
    return files


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "annotated").mkdir(exist_ok=True)
    (OUT_ROOT / "json").mkdir(exist_ok=True)
    (OUT_ROOT / "crops").mkdir(exist_ok=True)

    images = collect_images(INPUT_DIR)
    print(f"[fast-alpr] {len(images)} images in {INPUT_DIR}")

    print(f"[fast-alpr] loading detector={DETECTOR_MODEL} ocr={OCR_MODEL} device=cpu ...")
    t0 = time.time()
    alpr = ALPR(
        detector_model=DETECTOR_MODEL,
        ocr_model=OCR_MODEL,
        detector_conf_thresh=CONF_THRESH,
        ocr_device="cpu",
    )
    print(f"[fast-alpr] ALPR ready in {time.time() - t0:.1f}s")

    summary: list[dict] = []
    grand_t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        t1 = time.time()
        try:
            draw = alpr.draw_predictions(str(img_path))
        except Exception as e:
            print(f"[fast-alpr] {img_path.name}: ERROR {type(e).__name__}: {e}")
            summary.append({
                "file": img_path.name,
                "error": f"{type(e).__name__}: {e}",
                "plates": [],
                "elapsed_s": round(time.time() - t1, 3),
            })
            (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
                json.dumps(summary[-1], indent=2)
            )
            continue

        results = draw.results
        # Save annotated image
        annotated_path = OUT_ROOT / "annotated" / (img_path.stem + ".jpg")
        cv2.imwrite(str(annotated_path), draw.image)

        plates: list[dict] = []
        for i, r in enumerate(results):
            bb = r.detection.bounding_box
            x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
            det_conf = float(r.detection.confidence)
            label = str(r.detection.label)

            if r.ocr is not None:
                text = str(r.ocr.text)
                # Confidence may be scalar or per-char list
                ocr_conf_raw = r.ocr.confidence
                if isinstance(ocr_conf_raw, (list, tuple)):
                    ocr_conf = float(sum(ocr_conf_raw) / max(1, len(ocr_conf_raw)))
                else:
                    ocr_conf = float(ocr_conf_raw)
                region = r.ocr.region
                region_conf = r.ocr.region_confidence
            else:
                text = ""
                ocr_conf = 0.0
                region = None
                region_conf = None

            plates.append({
                "idx": i,
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": det_conf,
                "text": text,
                "ocr_conf": ocr_conf,
                "region": region,
                "region_conf": region_conf,
            })

            # Save crop of plate region from the original frame
            crop = draw.image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                safe_text = "".join(
                    ch if ch.isalnum() else "_" for ch in text
                )[:24] or "notext"
                crop_path = OUT_ROOT / "crops" / f"{img_path.stem}__{i:02d}__{safe_text}.jpg"
                cv2.imwrite(str(crop_path), crop)

        elapsed = time.time() - t1
        summary.append({
            "file": img_path.name,
            "num_plates": len(plates),
            "plates": plates,
            "elapsed_s": round(elapsed, 3),
        })
        (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
            json.dumps(summary[-1], indent=2)
        )
        print(
            f"[fast-alpr] [{idx:02d}/{len(images)}] {img_path.name}: "
            f"{len(plates)} plate(s) "
            f"texts={[p['text'] for p in plates]} "
            f"({elapsed:.2f}s)"
        )

    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "engine": "fast-alpr",
        "detector": DETECTOR_MODEL,
        "ocr": OCR_MODEL,
        "conf_thresh": CONF_THRESH,
        "num_images": len(images),
        "total_elapsed_s": round(time.time() - grand_t0, 3),
        "results": summary,
    }, indent=2))

    total_plates = sum(len(r["plates"]) for r in summary)
    print(f"\n[fast-alpr] DONE — {len(images)} images, {total_plates} plates, "
          f"{time.time() - grand_t0:.1f}s total")
    print(f"[fast-alpr] results → {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())