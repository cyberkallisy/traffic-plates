"""
OCR-only pipeline (no detection) for already-cropped name-plate images.

Input  : C:/Users/gsash/Downloads/traffic-plates/name plate/crops/*.jpg
Output : <OUT>/name_plate_ocr.pdf          ← 2-column table: image_name | result
         <OUT>/summary.json                  ← per-image raw records
         <OUT>/per_image/<name>.json         ← per-image OCR record

OCR engine : fast-alpr / fast-plate-ocr / DefaultOCR (cct-xs-v2-global-model)
            → loaded directly via DefaultOCR.predict(cropped_plate_bgr),
              bypassing the YOLO detector entirely (since the input
              is already a cropped plate).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Paranoia: pure-python protobuf before any paddle/torch import
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2

from fast_alpr.default_ocr import DefaultOCR

# reportlab for the PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.lib.enums import TA_LEFT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = Path(r"C:/Users/gsash/Downloads/traffic-plates/name plate/crops")
OUT_ROOT  = Path(r"C:/Users/gsash/Downloads/traffic-plates/name_plate_ocr_results")
OCR_MODEL = "cct-xs-v2-global-model"     # fast-alpr OCR (global/European chars)
DEVICE    = "cpu"
EXT_OK    = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

PDF_TITLE = "Name-Plate OCR Results — fast-alpr (OCR only, no detection)"


# ---------------------------------------------------------------------------
# Collect + OCR
# ---------------------------------------------------------------------------
def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXT_OK
    )


def _flatten_conf(raw) -> float:
    """fast-alpr returns confidence as float OR list[float] — handle both."""
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return 0.0
        return float(sum(raw) / len(raw))
    try:
        return float(raw)
    except Exception:
        return 0.0


def run_ocr(images: list[Path]) -> list[dict]:
    (OUT_ROOT / "per_image").mkdir(parents=True, exist_ok=True)

    print(f"[ocr] {len(images)} images in {INPUT_DIR}")
    print(f"[ocr] loading {OCR_MODEL} on {DEVICE} ...")
    t0 = time.time()
    ocr = DefaultOCR(hub_ocr_model=OCR_MODEL, device=DEVICE)
    print(f"[ocr] DefaultOCR ready in {time.time() - t0:.1f}s")

    grand_t0 = time.time()
    summary: list[dict] = []
    for idx, img_path in enumerate(images, 1):
        t1 = time.time()
        rec: dict = {"file": img_path.name, "path": str(img_path)}
        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imread returned None")
            ocr_res = ocr.predict(bgr)
            if ocr_res is None:
                rec.update({"text": "", "ocr_conf": 0.0, "region": None,
                            "region_conf": None, "error": "ocr returned None"})
            else:
                rec.update({
                    "text":         str(ocr_res.text or ""),
                    "ocr_conf":     _flatten_conf(ocr_res.confidence),
                    "region":       ocr_res.region,
                    "region_conf":  (float(ocr_res.region_confidence)
                                     if ocr_res.region_confidence is not None else None),
                })
        except Exception as e:
            rec.update({"text": "", "ocr_conf": 0.0, "region": None,
                        "region_conf": None, "error": f"{type(e).__name__}: {e}"})

        rec["elapsed_s"] = round(time.time() - t1, 3)
        (OUT_ROOT / "per_image" / (img_path.stem + ".json")).write_text(
            json.dumps(rec, indent=2, default=str)
        )
        summary.append(rec)
        preview = rec.get("text") or "(empty)"
        print(f"[ocr] [{idx:02d}/{len(images)}] {img_path.name}: "
              f"text={preview!r}  conf={rec.get('ocr_conf', 0):.2f}  "
              f"({rec['elapsed_s']:.2f}s)")

    total_elapsed = time.time() - grand_t0
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "engine":         "fast-alpr (OCR only via DefaultOCR)",
        "ocr_model":      OCR_MODEL,
        "device":         DEVICE,
        "input_dir":      str(INPUT_DIR),
        "num_images":     len(images),
        "total_elapsed_s": round(total_elapsed, 3),
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "results":        summary,
    }, indent=2, default=str))

    nonempty = sum(1 for r in summary if r.get("text"))
    print(f"\n[ocr] DONE — {len(images)} images, {nonempty} with OCR text, "
          f"{total_elapsed:.1f}s total")
    return summary


# ---------------------------------------------------------------------------
# PDF builder — 2 columns: image_name | result
# ---------------------------------------------------------------------------
def build_pdf(summary: list[dict], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title=PDF_TITLE, author="Hermes Agent",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=16, leading=20, textColor=HexColor("#1f2937"),
        spaceAfter=4, alignment=TA_LEFT,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, textColor=grey, leading=12, alignment=TA_LEFT,
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=HexColor("#111827"),
    )
    cell = ParagraphStyle(
        "Cell", parent=styles["Normal"],
        fontSize=10, leading=12, textColor=HexColor("#111827"),
    )
    cell_mono = ParagraphStyle(
        "CellMono", parent=styles["Code"],
        fontSize=9, leading=12, fontName="Courier",
        textColor=HexColor("#111827"),
    )
    header_cell = ParagraphStyle(
        "Hdr", parent=styles["Normal"],
        fontSize=10, leading=12, fontName="Helvetica-Bold",
        textColor=HexColor("#ffffff"),
    )

    story = []
    story.append(Paragraph(PDF_TITLE, title_style))
    story.append(Paragraph(
        f"OCR engine: <b>{OCR_MODEL}</b> (fast-alpr, OCR only — no detection) "
        f"&nbsp;|&nbsp; Device: <b>{DEVICE}</b>",
        meta_style,
    ))
    story.append(Paragraph(
        f"Input: <font name='Courier'>{INPUT_DIR}</font><br/>"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        meta_style,
    ))
    story.append(Spacer(1, 0.4 * cm))

    nonempty = sum(1 for r in summary if r.get("text"))
    empty    = len(summary) - nonempty
    story.append(Paragraph(
        f"<b>{len(summary)}</b> images &nbsp;|&nbsp; "
        f"<b>{nonempty}</b> with OCR text &nbsp;|&nbsp; "
        f"<b>{empty}</b> empty/blank",
        body,
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ---- the 2-column table ---------------------------------------------
    rows = [[
        Paragraph("#",        header_cell),
        Paragraph("image_name", header_cell),
        Paragraph("result",   header_cell),
    ]]
    for i, r in enumerate(summary, 1):
        text = r.get("text", "") or ""
        if not text:
            text_disp = "<i>(empty)</i>"
        else:
            text_disp = text

        # confidence badge (small)
        conf = float(r.get("ocr_conf", 0.0) or 0.0)
        if text:
            text_disp = f"{text_disp} <font color='#6b7280' size='8'>[conf {conf:.2f}]</font>"

        err = r.get("error")
        if err:
            text_disp += f"<br/><font color='#dc2626' size='8'>{err}</font>"

        rows.append([
            Paragraph(str(i), cell),
            Paragraph(f"<font name='Courier' size='9'>{r['file']}</font>", cell_mono),
            Paragraph(text_disp, cell),
        ])

    # A4 width 21cm minus 3cm margins = 18cm usable
    tbl = Table(
        rows,
        colWidths=[0.9 * cm, 6.5 * cm, 10.6 * cm],
        hAlign="LEFT",
        repeatRows=1,
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  HexColor("#1d4ed8")),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  HexColor("#ffffff")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [HexColor("#ffffff"), HexColor("#f9fafb")]),
        ("BOX",          (0, 0), (-1, -1), 0.5, HexColor("#9ca3af")),
        ("INNERGRID",    (0, 0), (-1, -1), 0.25, HexColor("#d1d5db")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)

    doc.build(story)
    print(f"[pdf] wrote {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    images = collect_images(INPUT_DIR)
    if not images:
        print(f"[error] no images found in {INPUT_DIR}", file=sys.stderr)
        return 1

    summary = run_ocr(images)

    pdf_path = OUT_ROOT / "name_plate_ocr.pdf"
    build_pdf(summary, pdf_path)

    nonempty = sum(1 for r in summary if r.get("text"))
    print("\n=== Summary ===")
    print(f"Images processed : {len(summary)}")
    print(f"OCR text found   : {nonempty}")
    print(f"Output folder    : {OUT_ROOT}")
    print(f"PDF              : {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
