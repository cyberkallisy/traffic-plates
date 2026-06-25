"""
TrOCR batch test on the 16 traffic-plates test images.

Pipeline:
  YOLO11 plate detect (project's local copy)  →  crop each plate
       ↓
  TrOCR (microsoft/trocr-base-printed or trocr-small-printed)
       ↓
  per-image JSON + annotated JPG + crops
  summary.json

Why two TrOCR variants:
  trocr-small-printed  (~62M params, ~250MB) — fast, decent on printed text
  trocr-base-printed   (~125M params, ~500MB) — slower, more accurate

We run BASE for the final output (per the user's "TrOCR" request — base is the
canonical choice). If memory pressure becomes an issue, swap to SMALL.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "C:/Users/gsash/Downloads/Facial-recognition/venv/Scripts/python.exe" \
        trocr_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Paranoia: force pure-python protobuf before any paddle/torch import
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
YOLO_PT = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt")  # 5.46 MB local copy
OUT_ROOT = Path(r"C:/Users/gsash/Downloads/traffic-plates/trocr_results_20260622_070000")

TROCR_MODEL = "microsoft/trocr-base-printed"
YOLO_CONF = 0.25
PAD_FRAC = 0.50   # 50% bbox padding — see ANPR skill multi-engine voting note

EXT_OK = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXT_OK
    )


def crop_with_pad(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  pad_frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(bw * pad_frac), 8)
    pad_y = max(int(bh * pad_frac), 6)
    x1c = max(0, x1 - pad_x)
    y1c = max(0, y1 - pad_y)
    x2c = min(w, x2 + pad_x)
    y2c = min(h, y2 + pad_y)
    return img[y1c:y2c, x1c:x2c].copy()


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "annotated").mkdir(exist_ok=True)
    (OUT_ROOT / "json").mkdir(exist_ok=True)
    (OUT_ROOT / "crops").mkdir(exist_ok=True)

    images = collect_images(INPUT_DIR)
    print(f"[TrOCR] {len(images)} images in {INPUT_DIR}")

    # --- Load models --------------------------------------------------------
    print(f"[TrOCR] loading YOLO11 from {YOLO_PT} ...")
    yolo = YOLO(str(YOLO_PT))

    print(f"[TrOCR] loading TrOCR ({TROCR_MODEL}) ...")
    t_load = time.time()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[TrOCR] TrOCR ready on {device} in {time.time() - t_load:.1f}s")

    def trocr_read(crop_bgr: np.ndarray) -> tuple[str, float]:
        """Run TrOCR on a BGR crop → (text, confidence)."""
        if crop_bgr.size == 0:
            return "", 0.0
        # BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # Upscale tiny crops so TrOCR's ViT (typically 224 or 384 input) sees
        # legible glyphs. Target ~384 on the long side.
        target_long = 384
        w, h = pil_img.size
        long_side = max(w, h)
        if long_side < target_long:
            scale = target_long / long_side
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        with torch.no_grad():
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
            outputs = model.generate(
                pixel_values,
                max_new_tokens=16,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )
            seq = outputs.sequences
            text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
            # Average log-prob over generated tokens (rough confidence)
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                conf = float(torch.sigmoid(outputs.sequences_scores).item())
            else:
                conf = 0.0
        return text, conf

    # --- Process each image -------------------------------------------------
    summary: list[dict] = []
    grand_t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        t1 = time.time()
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[TrOCR] {img_path.name}: cannot read")
            summary.append({"file": img_path.name, "error": "cannot_read",
                            "plates": [], "elapsed_s": 0.0})
            (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
                json.dumps(summary[-1], indent=2))
            continue

        # YOLO detect plates
        yres = yolo.predict(bgr, conf=YOLO_CONF, verbose=False)[0]
        plates: list[dict] = []
        annotated = bgr.copy()
        for i, box in enumerate(yres.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            det_conf = float(box.conf[0].cpu().numpy())

            crop = crop_with_pad(bgr, x1, y1, x2, y2, PAD_FRAC)
            text, ocr_conf = trocr_read(crop)

            plates.append({
                "idx": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": det_conf,
                "text": text,
                "ocr_conf": ocr_conf,
            })

            # Draw box + text
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{text or '(no text)'} ({ocr_conf:.2f})"
            ty = max(0, y1 - 8)
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Save crop
            safe_text = "".join(ch if ch.isalnum() else "_" for ch in text)[:24] or "notext"
            crop_path = OUT_ROOT / "crops" / f"{img_path.stem}__{i:02d}__{safe_text}.jpg"
            cv2.imwrite(str(crop_path), crop)

        # Save annotated image
        cv2.imwrite(str(OUT_ROOT / "annotated" / (img_path.stem + ".jpg"), ), annotated)

        elapsed = time.time() - t1
        summary.append({
            "file": img_path.name,
            "num_plates": len(plates),
            "plates": plates,
            "elapsed_s": round(elapsed, 3),
        })
        (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
            json.dumps(summary[-1], indent=2))
        print(
            f"[TrOCR] [{idx:02d}/{len(images)}] {img_path.name}: "
            f"{len(plates)} plate(s) "
            f"texts={[p['text'] for p in plates]} "
            f"({elapsed:.2f}s)"
        )

    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "engine": "TrOCR",
        "trocr_model": TROCR_MODEL,
        "yolo_model": str(YOLO_PT),
        "conf_thresh": YOLO_CONF,
        "pad_frac": PAD_FRAC,
        "device": device,
        "num_images": len(images),
        "total_elapsed_s": round(time.time() - grand_t0, 3),
        "results": summary,
    }, indent=2))

    total_plates = sum(len(r["plates"]) for r in summary)
    print(f"\n[TrOCR] DONE — {len(images)} images, {total_plates} plates, "
          f"{time.time() - grand_t0:.1f}s total")
    print(f"[TrOCR] results → {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())