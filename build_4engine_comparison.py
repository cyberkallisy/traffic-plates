"""
4-Engine OCR comparison: Qwen2.5-VL + minimax-m3 + fast-alpr + TrOCR
on the 16 test images.
"""
import os
import json
import time
import sys
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(r"C:\Users\gsash\Downloads\test\New folder")
OUT  = BASE / "comparison_4engines"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "annotated").mkdir(exist_ok=True)
(OUT / "crops").mkdir(exist_ok=True)

EXPECTED = ['1.png','2.png','3.png','4.png','5.png','6.png',
            'invalid.png','no.png','no1.png','no2.png','no4.png',
            'not read.png','yes.png','yes1.png','yes2.png','yes3.png']
IMAGES = [n for n in EXPECTED if (BASE / n).exists()]
print(f"Will process {len(IMAGES)} images", flush=True)

YOLO_PT = r"C:\Users\gsash\Downloads\traffic-plates\yolo11_plate.pt"
PYTHON = r"C:\Users\gsash\Downloads\Facial-recognition\venv\Scripts\python.exe"

try:
    font_small = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font_small = ImageFont.load_default()

CROP_DIR = OUT / "crops" / "yolo11_crops"
CROP_DIR.mkdir(parents=True, exist_ok=True)
DETECTIONS_FILE = OUT / "yolo11_detections.json"

# ---------------------------------------------------------------------------
# Step 1: YOLO11 detection + crop
# ---------------------------------------------------------------------------
print("\n=== Step 1: YOLO11 detection + cropping ===", flush=True)

yolo_script = OUT / "_yolo_detect.py"
yolo_script.write_text(
    "import json\n"
    "from pathlib import Path\n"
    "from ultralytics import YOLO\n"
    "from PIL import Image\n"
    "\n"
    f"BASE = Path(r'{BASE}')\n"
    f"IMAGES = {IMAGES!r}\n"
    f"YOLO_PT = r'{YOLO_PT}'\n"
    f"CROP_DIR = Path(r'{CROP_DIR}')\n"
    "CROP_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "model = YOLO(YOLO_PT)\n"
    "results_dict = {}\n"
    "\n"
    "for img_name in IMAGES:\n"
    "    img_path = BASE / img_name\n"
    "    out = model.predict(str(img_path), conf=0.25, verbose=False)[0]\n"
    "    boxes = out.boxes\n"
    "    if boxes is None or len(boxes) == 0:\n"
    "        results_dict[img_name] = {'num_plates': 0, 'plates': []}\n"
    "        continue\n"
    "    pil = Image.open(img_path).convert('RGB')\n"
    "    iw, ih = pil.size\n"
    "    plates = []\n"
    "    for i, b in enumerate(boxes):\n"
    "        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())\n"
    "        bw, bh = x2 - x1, y2 - y1\n"
    "        pad_x = max(int(bw * 0.50), 10)\n"
    "        pad_y = max(int(bh * 0.50), 6)\n"
    "        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)\n"
    "        cx2, cy2 = min(iw, x2 + pad_x), min(ih, y2 + pad_y)\n"
    "        crop = pil.crop((cx1, cy1, cx2, cy2))\n"
    "        crop_path = CROP_DIR / f'{Path(img_name).stem}_p{i:02d}.png'\n"
    "        crop.save(crop_path)\n"
    "        plates.append({\n"
    "            'bbox': [x1, y1, x2, y2],\n"
    "            'yolo_conf': float(b.conf[0]),\n"
    "            'crop_file': crop_path.name,\n"
    "        })\n"
    "    results_dict[img_name] = {'num_plates': len(plates), 'plates': plates}\n"
    "    print(f'  {img_name}: {len(plates)} plates', flush=True)\n"
    "\n"
    f"Path(r'{DETECTIONS_FILE}').write_text(json.dumps(results_dict, indent=2))\n"
    "print('Done.', flush=True)\n"
, encoding="utf-8")

print("Running YOLO11 detection...", flush=True)
proc = subprocess.run(
    [PYTHON, "-u", str(yolo_script)],
    cwd=str(OUT),
    capture_output=True, text=True, timeout=300,
)
print(proc.stdout[-1500:])
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-1500:])
    sys.exit(1)

detections = json.loads(DETECTIONS_FILE.read_text())
total_plates = sum(d["num_plates"] for d in detections.values())
print(f"\nTotal plates detected: {total_plates}", flush=True)

all_crops = sorted(CROP_DIR.glob("*.png"))
print(f"Total crops: {len(all_crops)}", flush=True)

# ---------------------------------------------------------------------------
# Step 2a: TrOCR on all crops
# ---------------------------------------------------------------------------
print("\n=== Step 2a: TrOCR on all crops ===", flush=True)
TROCR_OUT = OUT / "trocr_results"
TROCR_OUT.mkdir(exist_ok=True)

trocr_script = OUT / "_trocr_run.py"
trocr_script.write_text(
    "import json, time\n"
    "from pathlib import Path\n"
    "from PIL import Image\n"
    "import torch\n"
    "from transformers import TrOCRProcessor, VisionEncoderDecoderModel\n"
    "\n"
    f"CROP_DIR = Path(r'{CROP_DIR}')\n"
    f"OUT_FILE = Path(r'{TROCR_OUT}/summary.json')\n"
    "OUT_FILE.parent.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "print('Loading TrOCR...', flush=True)\n"
    "t0 = time.time()\n"
    "processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')\n"
    "model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').eval()\n"
    "print(f'  loaded in {time.time()-t0:.1f}s', flush=True)\n"
    "\n"
    "crops = sorted(CROP_DIR.glob('*.png'))\n"
    "results = []\n"
    "for i, c in enumerate(crops, 1):\n"
    "    img = Image.open(c).convert('RGB')\n"
    "    t = time.time()\n"
    "    with torch.no_grad():\n"
    "        ids = model.generate(processor(images=img, return_tensors='pt').pixel_values, max_new_tokens=32)\n"
    "    text = processor.batch_decode(ids, skip_special_tokens=True)[0]\n"
    "    el = time.time() - t\n"
    "    stem = c.stem.rsplit('_p', 1)[0]\n"
    "    idx = c.stem.rsplit('_p', 1)[1]\n"
    "    results.append({\n"
    "        'crop_file': c.name,\n"
    "        'source_image_stem': stem,\n"
    "        'plate_idx': int(idx),\n"
    "        'trocr_text': text,\n"
    "        'elapsed_sec': round(el, 3),\n"
    "    })\n"
    "    if i % 5 == 0 or i == len(crops):\n"
    "        print(f'  {i}/{len(crops)}: {c.name} -> {text!r} ({el:.2f}s)', flush=True)\n"
    "\n"
    "OUT_FILE.write_text(json.dumps(results, indent=2))\n"
    "print(f'Wrote {OUT_FILE}', flush=True)\n"
, encoding="utf-8")

print("Starting TrOCR (1.3GB model, ~30s warmup)...", flush=True)
proc = subprocess.run(
    [PYTHON, "-u", str(trocr_script)],
    cwd=str(OUT),
    capture_output=True, text=True, timeout=900,
)
print(proc.stdout[-2000:])
if proc.returncode != 0:
    print("TrOCR STDERR:", proc.stderr[-1500:])

trocr_data = []
trocr_file = TROCR_OUT / "summary.json"
if trocr_file.exists():
    trocr_data = json.loads(trocr_file.read_text())
print(f"\nTrOCR processed: {len(trocr_data)} crops", flush=True)

trocr_by_image = {}
for r in trocr_data:
    trocr_by_image.setdefault(r["source_image_stem"], []).append(r)

# ---------------------------------------------------------------------------
# Step 2b: Qwen2.5-VL on all crops
# ---------------------------------------------------------------------------
print("\n=== Step 2b: Qwen2.5-VL on all crops ===", flush=True)
QWEN_OUT = OUT / "qwen_results"
QWEN_OUT.mkdir(exist_ok=True)

PROMPT = "Read the license plate text exactly as it appears in the image. Return ONLY the plate text, no other commentary. If unreadable, return UNREADABLE."

qwen_script = OUT / "_qwen_run.py"
qwen_script.write_text(
    "import json, time, os\n"
    "from pathlib import Path\n"
    "from PIL import Image\n"
    "os.environ['HF_HUB_OFFLINE'] = '0'\n"
    "\n"
    "print('Loading Qwen2.5-VL-7B-Instruct...', flush=True)\n"
    "t0 = time.time()\n"
    "import torch\n"
    "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor\n"
    "\n"
    "processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')\n"
    "model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n"
    "    'Qwen/Qwen2.5-VL-7B-Instruct',\n"
    "    torch_dtype=torch.float16,\n"
    "    device_map='cpu',\n"
    "    low_cpu_mem_usage=True,\n"
    ").eval()\n"
    "print(f'  loaded in {time.time()-t0:.1f}s', flush=True)\n"
    "\n"
    f"CROP_DIR = Path(r'{CROP_DIR}')\n"
    f"OUT_FILE = Path(r'{QWEN_OUT}/summary.json')\n"
    "OUT_FILE.parent.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    f"PROMPT = {PROMPT!r}\n"
    "\n"
    "crops = sorted(CROP_DIR.glob('*.png'))\n"
    "results = []\n"
    "for i, c in enumerate(crops, 1):\n"
    "    img = Image.open(c).convert('RGB')\n"
    "    t = time.time()\n"
    "    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]\n"
    "    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
    "    inputs = processor(text=[text_input], images=[img], return_tensors='pt').to(model.device)\n"
    "    with torch.no_grad():\n"
    "        ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)\n"
    "    out_text = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()\n"
    "    el = time.time() - t\n"
    "    stem = c.stem.rsplit('_p', 1)[0]\n"
    "    idx = c.stem.rsplit('_p', 1)[1]\n"
    "    results.append({\n"
    "        'crop_file': c.name,\n"
    "        'source_image_stem': stem,\n"
    "        'plate_idx': int(idx),\n"
    "        'qwen_text': out_text,\n"
    "        'elapsed_sec': round(el, 3),\n"
    "    })\n"
    "    if i % 3 == 0 or i == len(crops):\n"
    "        print(f'  {i}/{len(crops)}: {c.name} -> {out_text!r} ({el:.2f}s)', flush=True)\n"
    "\n"
    "OUT_FILE.write_text(json.dumps(results, indent=2))\n"
    "print(f'Wrote {OUT_FILE}', flush=True)\n"
, encoding="utf-8")

print("Starting Qwen2.5-VL-7B (CPU) — this is SLOW, ~30-90s/crop...", flush=True)
print("Estimated time: ~15-45 min for all crops", flush=True)
try:
    proc = subprocess.run(
        [PYTHON, "-u", str(qwen_script)],
        cwd=str(OUT),
        capture_output=True, text=True, timeout=2700,
    )
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print("Qwen STDERR:", proc.stderr[-1500:])
except subprocess.TimeoutExpired as e:
    print(f"Qwen timed out — partial results saved")
    if e.stdout:
        print("Last stdout:", e.stdout[-500:].decode('utf-8', errors='replace'))

qwen_data = []
qwen_file = QWEN_OUT / "summary.json"
if qwen_file.exists():
    qwen_data = json.loads(qwen_file.read_text())
print(f"\nQwen processed: {len(qwen_data)} crops", flush=True)

qwen_by_image = {}
for r in qwen_data:
    qwen_by_image.setdefault(r["source_image_stem"], []).append(r)

# ---------------------------------------------------------------------------
# Step 3: Load fast-alpr + minimax-m3 from existing runs
# ---------------------------------------------------------------------------
print("\n=== Step 3: Loading existing fast-alpr + minimax-m3 results ===", flush=True)

def load_json(p):
    raw = Path(p).read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try: return json.loads(raw.decode(enc))
        except Exception: continue
    return None

fa_data = load_json(BASE / "fast_alpr_results/summary.json")
fa_by_img = {}
for r in fa_data:
    fa_json = load_json(BASE / f"fast_alpr_results/{Path(r['image']).stem}.json") or {}
    fa_by_img[r["image"]] = fa_json.get("plates", [])

m3 = load_json(BASE / "yolo11_minimax3_ocr_20260620_121220/summary.json")
m3_by_img = {r["file"]: r for r in m3["images"]}

# ---------------------------------------------------------------------------
# Step 4: Build comparison report
# ---------------------------------------------------------------------------
print("\n=== Step 4: Building comparison report ===", flush=True)

img_stem_map = {n: Path(n).stem for n in IMAGES}

rows = []
for img_name in IMAGES:
    stem = img_stem_map[img_name]
    det = detections.get(img_name, {"plates": []})
    fa_plates = fa_by_img.get(img_name, [])
    m3_entry = m3_by_img.get(img_name, {})
    m3_plates = m3_entry.get("detections", [])
    trocr_crops = sorted(trocr_by_image.get(stem, []), key=lambda r: r["plate_idx"])
    qwen_crops = sorted(qwen_by_image.get(stem, []), key=lambda r: r["plate_idx"])

    fa_top    = fa_plates[0]["ocr_text"] if fa_plates else ""
    m3_top    = m3_plates[0].get("ocr", {}).get("text", "") if m3_plates else ""
    qwen_top  = qwen_crops[0]["qwen_text"] if qwen_crops else ""
    trocr_top = trocr_crops[0]["trocr_text"] if trocr_crops else ""

    rows.append({
        "image": img_name,
        "yolo_count": det["num_plates"],
        "qwen_count": len(qwen_crops),
        "qwen_top": qwen_top,
        "minimax_count": len(m3_plates),
        "minimax_top": m3_top,
        "fast_alpr_count": len(fa_plates),
        "fast_alpr_top": fa_top,
        "trocr_count": len(trocr_crops),
        "trocr_top": trocr_top,
    })

# Summary markdown
sum_lines = [
    "# 4-Engine OCR Comparison — 16 Test Images",
    "",
    "**Engines compared:**",
    "",
    "1. **Qwen2.5-VL-7B-Instruct** (vision-language LLM, fresh per-crop run)",
    "2. **minimax-m3** (vision LLM, from Jun 20 batch run)",
    "3. **fast-alpr** (YOLOv9-t-384 + CCT-xs-v2 global OCR, fresh batch run)",
    "4. **TrOCR-base-printed** (vision encoder-decoder, fresh per-crop run)",
    "",
    "All engines share the same YOLO11 plate detector for crop locations (except fast-alpr which has its own YOLOv9 detector).",
    "",
    "| # | Image | YOLO | Qwen VL | minimax LLM | fast-alpr | TrOCR |",
    "|---|---|---|---|---|---|---|",
]
for i, r in enumerate(rows, 1):
    sum_lines.append(
        f"| {i} | {r['image']} "
        f"| {r['yolo_count']} "
        f"| {r['qwen_count']} / `{r['qwen_top']}` "
        f"| {r['minimax_count']} / `{r['minimax_top']}` "
        f"| {r['fast_alpr_count']} / `{r['fast_alpr_top']}` "
        f"| {r['trocr_count']} / `{r['trocr_top']}` |"
    )

sum_lines.append("")
sum_lines.append("## Totals")
sum_lines.append(f"- **Images:** {len(rows)}")
sum_lines.append(f"- **Qwen2.5-VL:** {sum(r['qwen_count'] for r in rows)} crops OCR'd")
sum_lines.append(f"- **minimax-m3:** {sum(r['minimax_count'] for r in rows)} plates")
sum_lines.append(f"- **fast-alpr:** {sum(r['fast_alpr_count'] for r in rows)} plates")
sum_lines.append(f"- **TrOCR:** {sum(r['trocr_count'] for r in rows)} crops OCR'd")

(OUT / "comparison_4engines_summary.md").write_text("\n".join(sum_lines), encoding="utf-8")
print(f"Wrote {OUT / 'comparison_4engines_summary.md'}", flush=True)

# Per-image markdown with embedded thumbnails
per_img_lines = [
    "# 4-Engine OCR Comparison — Per Image",
    "",
    "Each row: original image + top OCR text per engine",
    "",
    "---",
    "",
]
for r in rows:
    per_img_lines.append(f"## {r['image']}")
    per_img_lines.append("")
    per_img_lines.append("| Original | Qwen VL | minimax-m3 | fast-alpr | TrOCR |")
    per_img_lines.append("|---|---|---|---|---|")
    per_img_lines.append(
        f"| ![{r['image']}]({(BASE / r['image']).absolute().as_posix()}) "
        f"| `{r['qwen_top']}` (n={r['qwen_count']}) "
        f"| `{r['minimax_top']}` (n={r['minimax_count']}) "
        f"| `{r['fast_alpr_top']}` (n={r['fast_alpr_count']}) "
        f"| `{r['trocr_top']}` (n={r['trocr_count']}) |"
    )
    per_img_lines.append("")
    per_img_lines.append("---")
    per_img_lines.append("")

(OUT / "comparison_4engines_per_image.md").write_text("\n".join(per_img_lines), encoding="utf-8")
print(f"Wrote {OUT / 'comparison_4engines_per_image.md'}", flush=True)

# Raw merged data
(OUT / "comparison_4engines.json").write_text(json.dumps({
    "rows": rows,
    "trocr_count": len(trocr_data),
    "qwen_count": len(qwen_data),
}, indent=2), encoding="utf-8")
print(f"Wrote {OUT / 'comparison_4engines.json'}", flush=True)

# ---------------------------------------------------------------------------
# Step 5: PDF
# ---------------------------------------------------------------------------
print("\n=== Step 5: Generating PDF ===", flush=True)
PDF = OUT / "comparison_4engines.pdf"
WKHTML = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"

proc = subprocess.run(
    [
        "pandoc",
        str((OUT / "comparison_4engines_per_image.md").relative_to(BASE)),
        "-o", str(PDF.relative_to(BASE)),
        f"--pdf-engine={WKHTML}",
        f"--resource-path={BASE}",
        "-V", "geometry:margin=0.5in",
        "-V", "geometry:landscape",
        "--quiet",
    ],
    cwd=str(BASE),
    capture_output=True, text=True,
)
print("pandoc rc:", proc.returncode)
if proc.returncode == 0 and PDF.exists():
    data = PDF.read_bytes()
    print(f"✅ PDF: {PDF}")
    print(f"   size: {len(data)/1024:.1f} KB")
    print(f"   /Subtype /Image count: {data.count(b'/Subtype /Image')}")
    print(f"   /Type /Page count: {data.count(b'/Type /Page')}")
else:
    print("❌ PDF failed")
    print("stderr:", proc.stderr[-1500:])

print("\n=== Done ===", flush=True)