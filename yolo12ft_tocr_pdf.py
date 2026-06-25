"""
YOLO12 fine-tuned plate detector + TrOCR (base-printed) — test on 16 images
in C:/Users/gsash/Downloads/test/New folder and save PDF report in the same folder.

Pipeline:
  fine-tuned YOLO12 (yolo12_plate.pt) → detect license plates → crop (50% pad)
       ↓
  TrOCR-base-printed → read plate text
       ↓
  per-image JSON + annotated JPG + crops + summary
       ↓
  PDF report (image + bbox + OCR result)

Run with:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  "C:/Users/gsash/Downloads/Facial-recognition/venv/Scripts/python.exe" \
      yolo12ft_tocr_pdf.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    PageBreak,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR    = Path(r"C:/Users/gsash/Downloads/test/New folder")
YOLO_PT      = Path(r"C:/Users/gsash/Downloads/traffic-plates/yolo12_plate.pt")  # fine-tuned
TROCR_MODEL  = "microsoft/trocr-base-printed"

YOLO_CONF    = 0.25
PAD_FRAC     = 0.50
EXT_OK       = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Output: alongside input (timestamped sub-folder to keep clean)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = INPUT_DIR / f"yolo12ft_tocr_results_{ts}"


def collect_images(d: Path) -> list[Path]:
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in EXT_OK
    )


def crop_with_pad(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(bw * pad_frac), 8)
    pad_y = max(int(bh * pad_frac), 6)
    return img[max(0, y1 - pad_y):min(h, y2 + pad_y),
               max(0, x1 - pad_x):min(w, x2 + pad_x)].copy()


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "annotated").mkdir(exist_ok=True)
    (OUT_ROOT / "json").mkdir(exist_ok=True)
    (OUT_ROOT / "crops").mkdir(exist_ok=True)

    images = collect_images(INPUT_DIR)
    print(f"[Pipeline] {len(images)} images in {INPUT_DIR}", flush=True)

    # --- Models ------------------------------------------------------------
    print(f"[Pipeline] loading fine-tuned YOLO12 from {YOLO_PT} ...", flush=True)
    yolo = YOLO(str(YOLO_PT))

    print(f"[Pipeline] loading TrOCR ({TROCR_MODEL}) ...", flush=True)
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[Pipeline] TrOCR ready on {device} in {time.time() - t0:.1f}s", flush=True)

    def trocr_read(crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr.size == 0:
            return "", 0.0
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        target_long = 384
        w, h = pil_img.size
        long_side = max(w, h)
        if long_side < target_long:
            scale = target_long / long_side
            pil_img = pil_img.resize(
                (int(round(w * scale)), int(round(h * scale))), Image.LANCZOS
            )
        with torch.no_grad():
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
            outputs = model.generate(
                pixel_values,
                max_new_tokens=16,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )
            text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                conf = float(torch.sigmoid(outputs.sequences_scores).item())
            else:
                conf = 0.0
        return text, conf

    # --- Process each image ------------------------------------------------
    summary: list[dict] = []
    grand_t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        t1 = time.time()
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[Pipeline] {img_path.name}: cannot read", flush=True)
            summary.append({"file": img_path.name, "error": "cannot_read",
                            "plates": [], "elapsed_s": 0.0})
            (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
                json.dumps(summary[-1], indent=2))
            continue

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

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{text or '(no text)'} ({ocr_conf:.2f})"
            ty = max(15, y1 - 8)
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            safe_text = "".join(ch if ch.isalnum() else "_" for ch in text)[:24] or "notext"
            crop_path = OUT_ROOT / "crops" / f"{img_path.stem}__{i:02d}__{safe_text}.jpg"
            cv2.imwrite(str(crop_path), crop)

        cv2.imwrite(str(OUT_ROOT / "annotated" / (img_path.stem + ".jpg")), annotated)

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
            f"[Pipeline] [{idx:02d}/{len(images)}] {img_path.name}: "
            f"{len(plates)} plate(s) texts={[p['text'] for p in plates]} "
            f"({elapsed:.2f}s)",
            flush=True,
        )

    total_elapsed = time.time() - grand_t0
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "engine": "YOLO12-finetuned + TrOCR-base-printed",
        "yolo_model": str(YOLO_PT),
        "trocr_model": TROCR_MODEL,
        "conf_thresh": YOLO_CONF,
        "pad_frac": PAD_FRAC,
        "device": device,
        "num_images": len(images),
        "total_elapsed_s": round(total_elapsed, 3),
        "results": summary,
    }, indent=2))

    total_plates = sum(len(r["plates"]) for r in summary)
    print(f"\n[Pipeline] DONE — {len(images)} images, {total_plates} plates, "
          f"{total_elapsed:.1f}s total", flush=True)
    print(f"[Pipeline] results → {OUT_ROOT}", flush=True)

    # --- Build PDF ---------------------------------------------------------
    print(f"\n[Pipeline] building PDF ...", flush=True)
    pdf_path = INPUT_DIR / f"yolo12ft_tocr_results_{ts}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=16,
                                 alignment=1, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                        spaceAfter=6, textColor=colors.HexColor("#1f4e79"))
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8,
                                leading=10)
    cell_center = ParagraphStyle("CellC", parent=cell_style, alignment=1)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=0.4 * inch, rightMargin=0.4 * inch,
        topMargin=0.4 * inch, bottomMargin=0.4 * inch,
    )
    flow = []
    flow.append(Paragraph(
        "YOLO12 (fine-tuned) + TrOCR-base-printed — License Plate Recognition",
        title_style))
    flow.append(Paragraph(
        f"Source: {INPUT_DIR} | YOLO: yolo12_plate.pt (fine-tuned) | "
        f"TrOCR: microsoft/trocr-base-printed | Device: {device}",
        cell_style))
    flow.append(Paragraph(
        f"{len(images)} images · {total_plates} plates · {total_elapsed:.1f}s total",
        cell_style))
    flow.append(Spacer(1, 0.15 * inch))

    # Summary table
    flow.append(Paragraph("Summary", h2))
    sum_rows = [["#", "Image", "Plates", "Top OCR text", "Time (s)"]]
    for i, r in enumerate(summary, 1):
        plates = r.get("plates", [])
        top = plates[0]["text"] if plates and plates[0]["text"] else "—"
        if len(plates) > 1:
            top += f" (+{len(plates)-1} more)"
        sum_rows.append([
            str(i), r["file"], str(len(plates)),
            Paragraph(top or "<i>(none)</i>", cell_style),
            f"{r.get('elapsed_s', 0):.2f}",
        ])
    t = Table(sum_rows, colWidths=[0.3*inch, 1.6*inch, 0.6*inch, 5.0*inch, 0.7*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 0.15 * inch))

    # Per-image pages
    for i, r in enumerate(summary, 1):
        flow.append(PageBreak())
        flow.append(Paragraph(f"<b>#{i} — {r['file']}</b>", h2))
        plates = r.get("plates", [])
        annot_path = OUT_ROOT / "annotated" / (Path(r["file"]).stem + ".jpg")
        if annot_path.exists():
            try:
                flow.append(RLImage(str(annot_path), width=6.5*inch, height=4.0*inch,
                                    kind="proportional"))
            except Exception as e:
                flow.append(Paragraph(f"<i>(annotated image failed: {e})</i>", cell_style))
        else:
            flow.append(Paragraph("<i>(no annotated image)</i>", cell_style))

        # detections table for this image
        det_rows = [["#", "BBox (x1,y1,x2,y2)", "YOLO conf", "OCR text", "OCR conf"]]
        if not plates:
            det_rows.append(["—", "—", "—", "<i>(no plates detected)</i>", "—"])
        else:
            for p in plates:
                bb = p["bbox"]
                det_rows.append([
                    str(p["idx"]),
                    f"[{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}]",
                    f"{p['det_conf']:.3f}",
                    Paragraph(p["text"] or "<i>(none)</i>", cell_style),
                    f"{p['ocr_conf']:.3f}",
                ])
        det_table = Table(det_rows,
                          colWidths=[0.4*inch, 1.8*inch, 0.8*inch, 3.5*inch, 0.8*inch])
        det_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e75b6")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.white]),
        ]))
        flow.append(Spacer(1, 0.1 * inch))
        flow.append(det_table)
        flow.append(Paragraph(f"<i>Elapsed: {r.get('elapsed_s', 0):.2f}s</i>", cell_style))

    doc.build(flow)
    print(f"[Pipeline] PDF → {pdf_path}", flush=True)
    print(f"[Pipeline] PDF size: {pdf_path.stat().st_size / 1024:.1f} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())