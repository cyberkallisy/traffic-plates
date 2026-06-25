"""
TrOCR batch test on the plate crops produced by fast-alpr (same crops = direct comparison).
Reads C:/Users/gsash/Downloads/test/New folder/fast_alpr_results/crops/**/*.png,
runs TrOCR on each, writes per-crop JSON + annotated crops + summary.
"""
import sys
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

CROPS_ROOT = Path(r"C:/Users/gsash/Downloads/test/New folder/fast_alpr_results/crops")
OUT_ROOT   = Path(r"C:/Users/gsash/Downloads/test/New folder/trocr_results")
ANNOT_DIR  = OUT_ROOT / "annotated"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

crops = sorted(CROPS_ROOT.rglob("plate_*.png"))
print(f"Found {len(crops)} plate crops", flush=True)

print("\n[1/2] Loading TrOCR (microsoft/trocr-base-printed, local-only)...", flush=True)
t0 = time.time()
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402

MODEL_NAME = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, local_files_only=True).eval()
print(f"   loaded in {time.time()-t0:.1f}s", flush=True)

try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()

print(f"\n[2/2] Running TrOCR on {len(crops)} crops...\n", flush=True)
results = []
overall_start = time.time()

for idx, crop_path in enumerate(crops, 1):
    # relative path key: "<source_image>/plate_NN"
    rel = crop_path.relative_to(CROPS_ROOT)
    parts = rel.parts  # ('1', 'plate_00.png')
    source_image = parts[0]
    crop_name = parts[1]
    stem = crop_path.stem

    img = Image.open(crop_path).convert("RGB")

    crop_start = time.time()
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        gen_ids = model.generate(pixel_values, max_new_tokens=32)
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    elapsed = time.time() - crop_start

    # Annotate
    pil = img.copy()
    draw = ImageDraw.Draw(pil)
    draw.rectangle([(0, 0), (pil.size[0]-1, pil.size[1]-1)], outline="dodgerblue", width=4)
    # label below
    label = f"{text}"
    draw.rectangle([(0, 0), (pil.size[0], 32)], fill="dodgerblue")
    draw.text((4, 6), label, fill="white", font=font)
    ann_path = ANNOT_DIR / f"{source_image}__{stem}.png"
    pil.save(ann_path)

    rec = {
        "source_image": source_image,
        "crop_file": crop_path.name,
        "trocr_text": text,
        "elapsed_sec": round(elapsed, 3),
    }
    results.append(rec)
    print(f"  [{idx:2d}/{len(crops)}] {source_image}/{crop_name:14s} -> '{text}' ({elapsed:.2f}s)", flush=True)

total = time.time() - overall_start
print(f"\nTotal: {total:.1f}s ({total/max(1,len(crops)):.2f}s per crop)", flush=True)

(OUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2))

# Group by source image for convenience
by_source = {}
for r in results:
    by_source.setdefault(r["source_image"], []).append(r)
(OUT_ROOT / "summary_by_image.json").write_text(json.dumps(by_source, indent=2))

print(f"\nWrote: {OUT_ROOT}", flush=True)
print(f"  - summary.json ({len(results)} entries)", flush=True)
print(f"  - summary_by_image.json", flush=True)
print(f"  - {len(results)} annotated PNGs in annotated/", flush=True)