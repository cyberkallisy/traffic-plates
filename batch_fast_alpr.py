"""
Fast-ALPR batch test on the 16 test images.
For each image:
  - run ALPR.predict
  - save annotated image (bbox drawn + OCR text label)
  - save per-image JSON with bbox + text + confidence
  - save cropped plate images to crops/fast_alpr/<image>/
Then write summary.json + a markdown table with images.
"""
import sys
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Force unbuffered stdout on Windows
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_ROOT  = Path(r"C:/Users/gsash/Downloads/test/New folder/fast_alpr_results")
CROPS_DIR = OUT_ROOT / "crops"
ANNOT_DIR = OUT_ROOT / "annotated"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

# 16 test images (skip the comparison_report.txt etc.)
IMG_EXTS = {".png", ".jpg", ".jpeg"}
imgs = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
print(f"Found {len(imgs)} images in {INPUT_DIR}", flush=True)

print("\n[1/2] Loading fast-alpr (YOLOv9-t-384 + CCT-xs-v2)...", flush=True)
t0 = time.time()
from fast_alpr import ALPR  # noqa: E402
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
    detector_conf_thresh=0.3,
    ocr_device="cpu",
)
print(f"   loaded in {time.time()-t0:.1f}s", flush=True)

summary = []
overall_start = time.time()

print(f"\n[2/2] Processing {len(imgs)} images...\n", flush=True)
for idx, img_path in enumerate(imgs, 1):
    img_start = time.time()
    results = alpr.predict(str(img_path))
    elapsed = time.time() - img_start

    # PIL image for annotation
    pil = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    plates = []
    img_stem = img_path.stem
    img_crops_dir = CROPS_DIR / img_stem
    img_crops_dir.mkdir(parents=True, exist_ok=True)

    for j, r in enumerate(results):
        det = r.detection
        ocr = r.ocr
        bb = det.bounding_box
        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)

        # Annotate
        draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)
        label = f"{ocr.text} ({det.confidence:.2f})"
        # text background
        tw = draw.textlength(label, font=font)
        draw.rectangle([(x1, max(0, y1 - 22)), (x1 + int(tw) + 6, y1)], fill="lime")
        draw.text((x1 + 3, max(0, y1 - 20)), label, fill="black", font=font)

        # Crop with 50% padding (sweet spot)
        iw, ih = pil.size
        bw, bh = x2 - x1, y2 - y1
        pad_x = max(int(bw * 0.50), 10)
        pad_y = max(int(bh * 0.50), 6)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(iw, x2 + pad_x), min(ih, y2 + pad_y)
        crop = pil.crop((cx1, cy1, cx2, cy2))
        crop_path = img_crops_dir / f"plate_{j:02d}.png"
        crop.save(crop_path)

        plates.append({
            "index": j,
            "bbox": [x1, y1, x2, y2],
            "detector_conf": float(det.confidence),
            "ocr_text": ocr.text,
            "ocr_char_confs": list(ocr.confidence) if ocr.confidence else [],
            "region": ocr.region,
            "region_conf": float(ocr.region_confidence) if ocr.region_confidence else None,
            "crop_path": str(crop_path),
        })

    # Save annotated image
    ann_path = ANNOT_DIR / f"{img_stem}.png"
    pil.save(ann_path)

    # Per-image JSON
    rec = {
        "image": str(img_path),
        "image_name": img_path.name,
        "num_plates": len(plates),
        "elapsed_sec": round(elapsed, 3),
        "plates": plates,
    }
    (OUT_ROOT / f"{img_stem}.json").write_text(json.dumps(rec, indent=2))

    summary.append({
        "image": img_path.name,
        "num_plates": len(plates),
        "elapsed_sec": round(elapsed, 3),
        "ocr_texts": [p["ocr_text"] for p in plates],
        "top_text": plates[0]["ocr_text"] if plates else "",
        "top_conf":  plates[0]["detector_conf"] if plates else 0.0,
    })
    print(f"  [{idx:2d}/{len(imgs)}] {img_path.name:20s}  plates={len(plates):2d}  elapsed={elapsed:.2f}s  top='{plates[0]['ocr_text'] if plates else ''}'", flush=True)

total = time.time() - overall_start
print(f"\nTotal time: {total:.1f}s ({total/len(imgs):.2f}s per image)", flush=True)

# Summary files
(OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nWrote: {OUT_ROOT}", flush=True)
print(f"  - summary.json", flush=True)
print(f"  - {len(imgs)} per-image JSONs", flush=True)
print(f"  - {len(imgs)} annotated PNGs in annotated/", flush=True)
print(f"  - {sum(s['num_plates'] for s in summary)} crops across crops/", flush=True)