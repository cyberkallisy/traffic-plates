"""
YOLO 11 detection + TrOCR OCR pipeline for the 16 traffic-plates test images.

Pipeline per image:
  1. YOLO 11 plate detection (project's local copy)
  2. Crop each plate (with padding)
  3. TrOCR (microsoft/trocr-base-printed) on each crop → text
  4. Annotate full image with bbox + TrOCR text
  5. Save per-image JSON, crops, annotated PNG

After all 16 images:
  Generate two PDFs:
    - yolo11_tocr_results.pdf : columns [Image | YOLO bbox+conf | TrOCR text]
    - yolo11_detection.pdf   : columns [Image | YOLO bbox+conf]

Output folder:
  C:/Users/gsash/Downloads/traffic-plates/yolo11_tocr_results_<timestamp>/
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force pure-python protobuf (this machine's torch/protobuf combo needs it)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak,
)
from reportlab.lib.enums import TA_LEFT
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
YOLO_PT = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo11_plate.pt")
TROCR_MODEL = "microsoft/trocr-base-printed"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = Path(r"C:/Users/gsash/Downloads/traffic-plates") / f"yolo11_tocr_results_{STAMP}"
ANNOT_DIR = OUT_ROOT / "annotated"
CROP_DIR = OUT_ROOT / "crops"
JSON_DIR = OUT_ROOT / "json"
for d in (OUT_ROOT, ANNOT_DIR, CROP_DIR, JSON_DIR):
    d.mkdir(parents=True, exist_ok=True)

YOLO_CONF = 0.25
PAD_FRAC = 0.50
EXT_OK = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


def safe_text(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (s or ""))[:30] or "notext"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    images = collect_images(INPUT_DIR)
    print(f"[run] {len(images)} images in {INPUT_DIR}")

    # ----- Load YOLO 11 -----
    print(f"[run] loading YOLO 11 from {YOLO_PT} ...")
    yolo = YOLO(str(YOLO_PT))
    print(f"[run] YOLO 11 class names: {yolo.names}")

    # ----- Load TrOCR -----
    print(f"[run] loading TrOCR ({TROCR_MODEL}) ...")
    t_load = time.time()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[run] TrOCR ready on {device} in {time.time() - t_load:.1f}s")

    def trocr_read(crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr.size == 0:
            return "", 0.0
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
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
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                conf = float(torch.sigmoid(outputs.sequences_scores).item())
            else:
                conf = 0.0
        return text, conf

    # ----- Per-image processing -----
    summary: list[dict] = []
    grand_t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        t1 = time.time()
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[run] [{idx}/{len(images)}] {img_path.name}: cannot read")
            summary.append({"file": img_path.name, "error": "cannot_read",
                            "plates": [], "elapsed_s": 0.0})
            (JSON_DIR / (img_path.stem + ".json")).write_text(
                json.dumps(summary[-1], indent=2))
            continue

        h, w = bgr.shape[:2]
        yres = yolo.predict(bgr, conf=YOLO_CONF, verbose=False)[0]
        plates: list[dict] = []
        annotated = bgr.copy()

        for i, box in enumerate(yres.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            det_conf = float(box.conf[0].cpu().numpy())

            crop = crop_with_pad(bgr, x1, y1, x2, y2, PAD_FRAC)
            text, ocr_conf = trocr_read(crop)

            # annotated drawing (bbox + label)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
            label = f"{text or '(no text)'} ({ocr_conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(y1 - 8, th + 4)
            cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + int(tw) + 6, ty + 2),
                          (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            crop_filename = f"{img_path.stem}__{i:02d}__{safe_text(text)}.jpg"
            crop_path = CROP_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)

            plates.append({
                "idx": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": round(det_conf, 4),
                "text": text,
                "ocr_conf": round(ocr_conf, 4),
                "crop_file": crop_filename,
            })

        # annotated save
        annotated_path = ANNOT_DIR / (img_path.stem + ".jpg")
        cv2.imwrite(str(annotated_path), annotated)

        elapsed = time.time() - t1
        summary.append({
            "file": img_path.name,
            "width": w,
            "height": h,
            "num_plates": len(plates),
            "plates": plates,
            "annotated_file": annotated_path.name,
            "elapsed_s": round(elapsed, 3),
        })
        (JSON_DIR / (img_path.stem + ".json")).write_text(
            json.dumps(summary[-1], indent=2))
        print(
            f"[run] [{idx:02d}/{len(images)}] {img_path.name}: "
            f"{len(plates)} plate(s) "
            f"texts={[p['text'] for p in plates]} ({elapsed:.2f}s)",
            flush=True,
        )

    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "engine": "YOLO11 + TrOCR",
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
    print(f"\n[run] DONE — {len(images)} images, {total_plates} plates, "
          f"{time.time() - grand_t0:.1f}s total")
    print(f"[run] results -> {OUT_ROOT}")

    # ----- Build PDFs -----
    print("\n[run] building PDFs...")
    build_pdfs(OUT_ROOT, summary, images)
    print(f"[run] PDFs saved to {OUT_ROOT}")
    return 0


# -----------------------------------------------------------------------------
# PDF builders
# -----------------------------------------------------------------------------
def _thumb_for_pdf(path: Path, max_w_px: int = 380, max_h_px: int = 280):
    """Open image, scale to fit, return RLImage + original size string."""
    pil = Image.open(path).convert("RGB")
    w, h = pil.size
    scale = min(max_w_px / w, max_h_px / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    tmp_path = path.with_suffix(".pdf_thumb.jpg")
    pil.save(tmp_path, "JPEG", quality=85)
    return tmp_path, new_w, new_h


def _make_pdf(
    pdf_path: Path,
    title: str,
    rows: list[dict],
    include_ocr_col: bool,
):
    """rows: list of {file, annotated_path, plates}."""
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title=title,
    )
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "cell", parent=styles["BodyText"], fontSize=8, leading=10,
        alignment=TA_LEFT, wordWrap="CJK",
    )
    title_style = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=16, alignment=TA_LEFT,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "sub", parent=styles["BodyText"], fontSize=9,
        textColor=colors.grey, spaceAfter=8,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
        f"16 test images from C:/Users/gsash/Downloads/test/New folder — "
        f"YOLO 11 plate detector (yolo11_plate.pt, conf={YOLO_CONF})"
        + (f" + TrOCR ({TROCR_MODEL})" if include_ocr_col else " (detection only)"),
        sub_style,
    ))

    # Table header
    if include_ocr_col:
        header = ["#", "Image (annotated)", "YOLO 11 Detection (bbox + conf)", "TrOCR Text"]
    else:
        header = ["#", "Image (annotated)", "YOLO 11 Detection (bbox + conf)"]
    data = [header]

    for idx, row in enumerate(rows, 1):
        # Image cell (thumbnail of annotated)
        annotated_path = ANNOT_DIR / row["annotated_file"]
        thumb_path, tw, th = _thumb_for_pdf(annotated_path)
        img_cell = RLImage(str(thumb_path), width=tw, height=th)

        # Detection cell — list of bbox + conf per plate
        det_lines = []
        for j, p in enumerate(row["plates"], 1):
            x1, y1, x2, y2 = p["bbox"]
            det_lines.append(
                f"<b>Plate {j}</b>: [{x1},{y1}]–[{x2},{y2}] "
                f"conf={p['det_conf']:.2f}"
            )
        if not det_lines:
            det_lines.append("<i>No plates detected</i>")
        det_cell = Paragraph("<br/>".join(det_lines), cell_style)

        if include_ocr_col:
            ocr_lines = []
            for j, p in enumerate(row["plates"], 1):
                ocr_lines.append(
                    f"<b>Plate {j}</b>: "
                    f"{p['text'] if p['text'] else '<i>(empty)</i>'}"
                    f"  <font color='grey'>(ocr_conf={p['ocr_conf']:.2f})</font>"
                )
            if not ocr_lines:
                ocr_lines.append("<i>—</i>")
            ocr_cell = Paragraph("<br/>".join(ocr_lines), cell_style)
            data.append([str(idx), img_cell, det_cell, ocr_cell])
        else:
            data.append([str(idx), img_cell, det_cell])

    # Column widths (landscape A4 ~ 27.7 cm usable)
    col_widths = [0.8 * cm, 11.5 * cm, 7.5 * cm] + ([7.5 * cm] if include_ocr_col else [])

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F6F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    # Summary page
    story.append(PageBreak())
    story.append(Paragraph("Summary", title_style))
    total_plates = sum(len(r["plates"]) for r in rows)
    summary_lines = [
        f"<b>Total images processed:</b> {len(rows)}",
        f"<b>Total plates detected:</b> {total_plates}",
        f"<b>Images with plates:</b> {sum(1 for r in rows if r['plates'])}",
        f"<b>YOLO 11 model:</b> {YOLO_PT}",
        f"<b>YOLO confidence threshold:</b> {YOLO_CONF}",
    ]
    if include_ocr_col:
        summary_lines.insert(2, f"<b>TrOCR model:</b> {TROCR_MODEL}")
        non_empty = sum(
            1 for r in rows for p in r["plates"] if p["text"]
        )
        summary_lines.insert(3, f"<b>Plates with OCR text:</b> {non_empty}")
    story.append(Paragraph("<br/>".join(summary_lines), styles["BodyText"]))

    # Per-image one-line summary table
    story.append(Spacer(1, 0.5 * cm))
    sum_header = ["#", "Image", "Plates", "Top text (TrOCR)"] if include_ocr_col \
        else ["#", "Image", "Plates", "Top detection"]
    sum_data = [sum_header]
    for idx, row in enumerate(rows, 1):
        plates_n = len(row["plates"])
        if include_ocr_col:
            top = row["plates"][0]["text"] if row["plates"] else "—"
        else:
            p0 = row["plates"][0] if row["plates"] else None
            top = f"[{p0['bbox'][0]},{p0['bbox'][1]}]–[{p0['bbox'][2]},{p0['bbox'][3]}] {p0['det_conf']:.2f}" \
                if p0 else "—"
        sum_data.append([
            str(idx), row["file"], str(plates_n),
            Paragraph(top, cell_style),
        ])
    sum_table = Table(sum_data, colWidths=[0.8*cm, 6*cm, 1.5*cm, 17*cm])
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F6F7")]),
    ]))
    story.append(sum_table)

    doc.build(story)


def build_pdfs(out_root: Path, summary: list[dict], images: list[Path]):
    """Build both PDFs (TrOCR+YOLO and YOLO-only)."""
    # rows in the same order as collect_images() — that is the input order
    annotated_by_stem = {
        s["file"].rsplit(".", 1)[0]: s for s in summary
    }
    rows = []
    for img_path in images:
        stem = img_path.stem
        rec = annotated_by_stem.get(stem)
        if rec is None:
            rec = {
                "file": img_path.name,
                "annotated_file": img_path.name,
                "plates": [],
            }
        else:
            rec = dict(rec)
            rec["annotated_file"] = rec.get("annotated_file", img_path.name)
        rows.append(rec)

    pdf1 = out_root / "yolo11_tocr_results.pdf"
    _make_pdf(
        pdf1,
        "YOLO 11 + TrOCR — Traffic Plate Results",
        rows,
        include_ocr_col=True,
    )
    print(f"[pdf] wrote {pdf1}")

    pdf2 = out_root / "yolo11_detection.pdf"
    _make_pdf(
        pdf2,
        "YOLO 11 — Plate Detection Results",
        rows,
        include_ocr_col=False,
    )
    print(f"[pdf] wrote {pdf2}")


if __name__ == "__main__":
    sys.exit(main())