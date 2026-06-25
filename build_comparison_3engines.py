"""
Build a 3-engine OCR comparison PDF for all 16 test images.

Engines compared:
  1. fast-alpr (YOLOv9-t-384-end2end + CCT-xs-v2)         [fast_alpr_results/]
  2. YOLO11 + PaddleOCR mobile only                       [yolo11_paddleocr_20260622_071731/]
  3. YOLO11 + minimax-m3 (vision LLM) OCR                 [yolo11_minimax3_ocr_20260620_121220/]

Output:
  - comparison_3engines_per_image.md   (markdown with image refs)
  - comparison_3engines_summary.md     (text-only tables, no images)
  - comparison_3engines.pdf            (via pandoc + wkhtmltopdf)
  - comparison_3engines.json           (raw merged data)
"""
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT  = BASE / "comparison_3engines"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "annotated").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load all three engines (handling different schemas)
# ---------------------------------------------------------------------------
def load_json(p):
    raw = Path(p).read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    return None


# (a) fast-alpr: list of {image, num_plates, elapsed_sec, ocr_texts[], top_text}
fa_data = load_json(BASE / "fast_alpr_results/summary.json")
fast_alpr = {r["image"]: r for r in fa_data}

# (b) PaddleOCR: dict with results: [{image, plates: [{plate_idx, yolo_conf, ocr_text, ...}]}]
paddle = load_json(BASE / "yolo11_paddleocr_20260622_071731/summary.json")
paddle_by_img = {r["image"]: r for r in paddle["results"]}

# (c) minimax-m3: dict with images: [{file, detections: [{bbox, confidence, ocr_text, ocr_confidence}]}]
m3 = load_json(BASE / "yolo11_minimax3_ocr_20260620_121220/summary.json")
m3_by_img = {r["file"]: r for r in m3["images"]}

# ---------------------------------------------------------------------------
# 2. Build per-image annotated side-by-side image
# ---------------------------------------------------------------------------
imgs = sorted(fast_alpr.keys())
print(f"Building side-by-side annotated images for {len(imgs)} images...", flush=True)

try:
    font_big   = ImageFont.truetype("arial.ttf", 22)
    font_small = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()


def annotate(img_path: Path, plate_records, color, out_path: Path):
    """Draw plates from `plate_records` onto a copy of img_path."""
    pil = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for pr in plate_records:
        x1, y1, x2, y2 = pr["bbox"]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{pr['text']} ({pr['conf']:.2f})"
        try:
            tw = int(draw.textlength(label, font=font_small))
        except Exception:
            tw = len(label) * 10
        draw.rectangle([(x1, max(0, y1 - 24)), (x1 + tw + 6, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - 22)), label, fill="black", font=font_small)
    pil.save(out_path)


def extract_plates_fast_alpr(r):
    return [{
        "bbox": (d := load_json(BASE / f"fast_alpr_results/{Path(r['image']).stem}.json")
                  )["plates"][i]["bbox"],
        "text": r["ocr_texts"][i],
        "conf": load_json(BASE / f"fast_alpr_results/{Path(r['image']).stem}.json")
                  ["plates"][i]["detector_conf"],
    } for i in range(r["num_plates"])]


def best_text(paddle_plate):
    """Paddle stores OCR text under .ocr.text, with .ocr.raw[] as fallbacks."""
    ocr = paddle_plate.get("ocr", {}) or {}
    txt = ocr.get("text", "") or ""
    if not txt:
        for r in ocr.get("raw", []) or []:
            if r.get("text"):
                txt = r["text"]
                break
    return txt

# Build per-engine rows for the markdown table
md_rows = []
merged_data = []

for img_name in imgs:
    img_path = BASE / img_name
    fa = fast_alpr.get(img_name, {})
    pd_ = paddle_by_img.get(img_name, {})
    m3_ = m3_by_img.get(img_name, {})

    # fast-alpr details (per-image JSON has bboxes)
    fa_json = load_json(BASE / f"fast_alpr_results/{Path(img_name).stem}.json") or {}
    fa_plates = fa_json.get("plates", [])
    fa_records = [{"bbox": p["bbox"], "text": p["ocr_text"], "conf": p["detector_conf"]} for p in fa_plates]

    # Paddle plates
    pd_plates = pd_.get("plates", [])
    pd_records = [{"bbox": p["bbox"], "text": best_text(p), "conf": p.get("yolo_conf", 0)} for p in pd_plates]

    # minimax plates
    m3_dets = m3_.get("detections", [])
    m3_records = [{"bbox": d["bbox_xyxy"], "text": d.get("ocr", {}).get("text", ""), "conf": d.get("confidence", 0)} for d in m3_dets]

    # Annotated per-engine images
    fa_ann = OUT / "annotated" / f"{Path(img_name).stem}__fast_alpr.png"
    pd_ann = OUT / "annotated" / f"{Path(img_name).stem}__paddle.png"
    m3_ann = OUT / "annotated" / f"{Path(img_name).stem}__minimax.png"
    annotate(img_path, fa_records, "lime",      fa_ann)
    annotate(img_path, pd_records, "dodgerblue", pd_ann)
    annotate(img_path, m3_records, "orange",   m3_ann)

    # Best text per engine (highest detector conf)
    fa_top = fa_records[0]["text"] if fa_records else ""
    pd_top = pd_records[0]["text"] if pd_records else ""
    m3_top = m3_records[0]["text"] if m3_records else ""

    # Markdown row — table-only summary
    md_rows.append({
        "image": img_name,
        "fast_alpr_count": len(fa_records),
        "fast_alpr_top":   fa_top,
        "paddle_count":    len(pd_records),
        "paddle_top":      pd_top,
        "minimax_count":   len(m3_records),
        "minimax_top":     m3_top,
    })

    merged_data.append({
        "image": img_name,
        "fast_alpr": fa_records,
        "paddle":    pd_records,
        "minimax":   m3_records,
    })

# ---------------------------------------------------------------------------
# 3. Write markdown with embedded annotated thumbnails
# ---------------------------------------------------------------------------
# Use absolute path references so pandoc finds images regardless of cwd
MD_FULL = OUT / "comparison_3engines_per_image.md"
lines = [
    "# 3-Engine OCR Comparison — 16 Test Images",
    "",
    "**Engines compared:**",
    "",
    "1. **fast-alpr** — YOLOv9-t-384 license-plate-end2end + CCT-xs-v2 global OCR",
    "2. **YOLO11 + PaddleOCR mobile** — morsetechlab/yolov11 + PaddleOCR 2.7 mobile (en, angle-cls)",
    "3. **YOLO11 + minimax-m3 vision LLM** — morsetechlab/yolov11 + minimax-m3 multimodal OCR",
    "",
    "For each image: original | fast-alpr | paddleocr | minimax-m3",
    "",
    "---",
    "",
]

for row in md_rows:
    stem = Path(row["image"]).stem
    orig_abs   = (BASE / row["image"]).absolute().as_posix()
    fa_abs     = (OUT / "annotated" / f"{stem}__fast_alpr.png").absolute().as_posix()
    pd_abs     = (OUT / "annotated" / f"{stem}__paddle.png").absolute().as_posix()
    m3_abs     = (OUT / "annotated" / f"{stem}__minimax.png").absolute().as_posix()
    lines.append(f"## {row['image']}")
    lines.append("")
    lines.append("| Original | fast-alpr | PaddleOCR mobile | minimax-m3 |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| ![{row['image']}]({orig_abs}) "
        f"| ![fa]({fa_abs}) "
        f"| ![pd]({pd_abs}) "
        f"| ![m3]({m3_abs}) |"
    )
    lines.append("")
    lines.append("**Reads (top per engine):**")
    lines.append(f"- fast-alpr: `{row['fast_alpr_top']}` ({row['fast_alpr_count']} plates)")
    lines.append(f"- PaddleOCR: `{row['paddle_top']}` ({row['paddle_count']} plates)")
    lines.append(f"- minimax-m3: `{row['minimax_top']}` ({row['minimax_count']} plates)")
    lines.append("")
    lines.append("---")
    lines.append("")

MD_FULL.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {MD_FULL}", flush=True)

# ---------------------------------------------------------------------------
# 4. Write summary table (text-only, easy to skim)
# ---------------------------------------------------------------------------
MD_SUM = OUT / "comparison_3engines_summary.md"
sum_lines = [
    "# 3-Engine OCR Comparison — Summary",
    "",
    "| # | Image | fast-alpr (count / top) | PaddleOCR mobile (count / top) | minimax-m3 (count / top) |",
    "|---|---|---|---|---|",
]
for i, r in enumerate(md_rows, 1):
    sum_lines.append(
        f"| {i} | {r['image']} "
        f"| {r['fast_alpr_count']} / `{r['fast_alpr_top']}` "
        f"| {r['paddle_count']} / `{r['paddle_top']}` "
        f"| {r['minimax_count']} / `{r['minimax_top']}` |"
    )
# Totals
sum_lines.append("")
sum_lines.append("## Totals")
sum_lines.append(f"- **Images:** {len(md_rows)}")
sum_lines.append(f"- **fast-alpr** detected {sum(r['fast_alpr_count'] for r in md_rows)} plates")
sum_lines.append(f"- **PaddleOCR** detected {sum(r['paddle_count'] for r in md_rows)} plates")
sum_lines.append(f"- **minimax-m3** detected {sum(r['minimax_count'] for r in md_rows)} plates")
MD_SUM.write_text("\n".join(sum_lines), encoding="utf-8")
print(f"Wrote {MD_SUM}", flush=True)

# ---------------------------------------------------------------------------
# 5. Write raw JSON for programmatic use
# ---------------------------------------------------------------------------
(OUT / "comparison_3engines.json").write_text(json.dumps(merged_data, indent=2), encoding="utf-8")
print(f"Wrote {OUT / 'comparison_3engines.json'}", flush=True)

# ---------------------------------------------------------------------------
# 6. Convert to PDF via pandoc + wkhtmltopdf
# ---------------------------------------------------------------------------
import subprocess

PDF = OUT / "comparison_3engines.pdf"
# Use absolute paths inside OUT so wkhtmltopdf finds images
# Pandoc needs to be run from BASE so relative image refs work
proc = subprocess.run(
    [
        "pandoc",
        str(MD_FULL.relative_to(BASE)),
        "-o", str(PDF.relative_to(BASE)),
        "--pdf-engine=" + r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
        "--resource-path", str(BASE) + ";" + str(OUT),
        "-V", "geometry:margin=0.5in",
        "-V", "geometry:landscape",
        "--quiet",
    ],
    cwd=str(BASE),
    capture_output=True, text=True,
)
print("pandoc stdout:", proc.stdout)
print("pandoc stderr:", proc.stderr)
print("pandoc rc:", proc.returncode)

if PDF.exists():
    print(f"\n✅ PDF: {PDF}  ({PDF.stat().st_size/1024:.1f} KB)")
else:
    print(f"\n❌ PDF generation failed")