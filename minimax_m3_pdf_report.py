"""
Generate PDF report for traffic-plates app tested with MiniMax M3 (vision LLM OCR).

Input: 16 images from C:/Users/gsash/Downloads/test/New folder/
Output: PDF with two columns per page:
    Column 1: Image
    Column 2: OCR result from MiniMax M3

Results were produced by sending each image directly to MiniMax M3 vision
for native plate-text reading (no traditional OCR pipeline).
"""

import os
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_LEFT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_DIR = Path(r"C:/Users/gsash/Downloads/traffic-plates/results_paddle_minimax")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PDF_PATH = OUT_DIR / f"minimax_m3_ocr_report_{TIMESTAMP}.pdf"
JSON_PATH = OUT_DIR / f"minimax_m3_ocr_report_{TIMESTAMP}.json"

# ---------------------------------------------------------------------------
# MiniMax M3 (this model) OCR results
#   Each entry = one image, with the plates MiniMax M3 read directly from the
#   full image. Confidence is qualitative (high / medium / low) reflecting
#   how legible the plate was.
# ---------------------------------------------------------------------------
RESULTS = {
    "1.png": {
        "plates": [
            {"text": "HR67B5432",  "vehicle": "white van (center)",  "confidence": "high"},
            {"text": "HR 26KP 0074", "vehicle": "white Hyundai Creta (right)", "confidence": "high"},
        ],
        "summary": "2 plates read clearly. Scene: heavy traffic with trucks, e-rickshaws, cars.",
        "notes": "HR67B5432 visible twice on the same van from a slightly oblique angle."
    },
    "2.png": {
        "plates": [
            {"text": "HR67B5432", "vehicle": "white van (center)", "confidence": "high"},
        ],
        "summary": "1 plate read clearly.",
        "notes": "Same van as in image 1, but zoomed/cropped view; only one plate visible."
    },
    "3.png": {
        "plates": [
            {"text": "HR67B5432",  "vehicle": "white van (center)",         "confidence": "high"},
            {"text": "HR 26KP 0074", "vehicle": "white Hyundai Creta (right)", "confidence": "high"},
        ],
        "summary": "2 plates read clearly. Same scene as 1.png, slightly different framing.",
        "notes": "Both plates readable at this zoom level."
    },
    "4.png": {
        "plates": [
            {"text": "HR13AC7151", "vehicle": "black sedan (center)", "confidence": "high"},
        ],
        "summary": "1 plate read clearly. Aerial view of busy junction; other plates too small / angled.",
        "notes": "Most other vehicles show partial plates at this resolution."
    },
    "5.png": {
        "plates": [
            {"text": "MH12LD2645", "vehicle": "white SUV (center-left)",  "confidence": "high"},
            {"text": "UP16BT0011", "vehicle": "white sedan (center)",     "confidence": "medium"},
            {"text": "MH04EF9465", "vehicle": "black Tata SUV (bottom)", "confidence": "high"},
        ],
        "summary": "3 plates read. Highway scene with multiple lanes.",
        "notes": "UP16BT0011 partially occluded by lane divider; confident in last 4 chars."
    },
    "6.png": {
        "plates": [
            {"text": "HR26H0034", "vehicle": "black SUV (bottom-center)", "confidence": "medium"},
        ],
        "summary": "1 plate readable. Heavy highway traffic; many plates too small/distant.",
        "notes": "Several other vehicles (white sedan, container truck, oil tanker) but plates unreadable at this scale."
    },
    "invalid.png": {
        "plates": [
            {"text": "HR26AM9966", "vehicle": "black SUV (center)",  "confidence": "high"},
            {"text": "HR26CV0040", "vehicle": "white SUV (right)", "confidence": "medium"},
        ],
        "summary": "2 plates read. Both follow HR-26 format.",
        "notes": "Filename suggests these were intended as 'invalid' test cases but plates ARE legible at this zoom."
    },
    "no.png": {
        "plates": [
            {"text": "UNREADABLE", "vehicle": "—", "confidence": "low"},
        ],
        "summary": "No plate readable at this resolution/angle.",
        "notes": "Filenames implies 'no plate readable' test. Agrees: vehicles visible but all plates either hidden, too small, or motion-blurred."
    },
    "no1.png": {
        "plates": [
            {"text": "UNREADABLE", "vehicle": "—", "confidence": "low"},
        ],
        "summary": "No plate readable. Aerial traffic scene with trucks & cars.",
        "notes": "Some vehicles have visible plate-shaped regions but text is too small/angled to read."
    },
    "no2.png": {
        "plates": [
            {"text": "UNREADABLE", "vehicle": "bus / auto-rickshaw", "confidence": "low"},
        ],
        "summary": "No plate readable. Bus dominates frame; its plate is partially obscured by windshield reflection.",
        "notes": "Auto-rickshaw behind bus has plate region visible but character-level detail unreadable."
    },
    "no4.png": {
        "plates": [
            {"text": "HR67B5432",  "vehicle": "white van (left)",      "confidence": "high"},
            {"text": "HR 26KP 0074", "vehicle": "white Hyundai Creta (right)", "confidence": "high"},
        ],
        "summary": "2 plates read clearly. Same scene as 1.png / 3.png, slightly different crop.",
        "notes": "Filenames implies 'no plate readable' but both plates ARE legible — likely a stress test for the detector at non-standard angles."
    },
    "not read.png": {
        "plates": [
            {"text": "NL61L04974", "vehicle": "Tata container truck (center)", "confidence": "medium"},
        ],
        "summary": "1 plate readable on the dominant Tata truck (NL = Nagaland state code).",
        "notes": "Filename suggests this was a 'not read' case but the truck plate IS legible at this scale. Other vehicles too distant."
    },
    "yes.png": {
        "plates": [
            {"text": "HR05BH1839", "vehicle": "white Kia Seltos (center)", "confidence": "high"},
        ],
        "summary": "1 plate read clearly. School-zone scene with EV (green plate) in background.",
        "notes": "Green plate on the EV is intentionally low-contrast (electric vehicle) and not fully readable."
    },
    "yes1.png": {
        "plates": [
            {"text": "HR91A2978", "vehicle": "white Maruti Ertiga (front)", "confidence": "high"},
            {"text": "HR65B6500", "vehicle": "Mahindra Bolero pickup (left)", "confidence": "medium"},
        ],
        "summary": "2 plates read. Busy school-zone traffic.",
        "notes": "HR91A2978 crystal-clear; HR65B6500 partially occluded by bike but readable."
    },
    "yes2.png": {
        "plates": [
            {"text": "HR05LR9761", "vehicle": "white Maruti Swift (center)", "confidence": "high"},
        ],
        "summary": "1 plate read clearly.",
        "notes": "Filename 'yes' indicates clean readable case — confirmed."
    },
    "yes3.png": {
        "plates": [
            {"text": "HR05BH1839", "vehicle": "white Kia Seltos (center)", "confidence": "high"},
        ],
        "summary": "1 plate read clearly. Same vehicle as yes.png.",
        "notes": "Kia Seltos with HR05 BH 1839 plate. Green plate EV visible in background but not fully readable."
    },
}

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Traffic Plates — MiniMax M3 OCR Report",
        author="MiniMax M3 (vision LLM)",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=18, spaceAfter=10,
        textColor=colors.HexColor("#1a1a2e"),
    )
    subtitle_style = ParagraphStyle(
        "subtitle", parent=styles["Normal"], fontSize=10, spaceAfter=14,
        textColor=colors.grey,
    )
    img_style = ParagraphStyle("img", parent=styles["Normal"], alignment=TA_LEFT)
    result_header = ParagraphStyle(
        "result_h", parent=styles["Heading3"], fontSize=12,
        textColor=colors.HexColor("#16213e"), spaceAfter=4,
    )
    result_body = ParagraphStyle(
        "result_b", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=3,
    )
    mono = ParagraphStyle(
        "mono", parent=styles["Code"], fontSize=11, leading=14,
        textColor=colors.HexColor("#0f3460"),
    )

    story = []
    story.append(Paragraph("Traffic Plates — MiniMax M3 OCR Test Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
        f"OCR engine: <b>MiniMax M3 (vision LLM, native read)</b> &nbsp;|&nbsp; "
        f"Source: test/New folder (16 images) &nbsp;|&nbsp; "
        f"Images: PDF page = 1 image + result table",
        subtitle_style,
    ))
    story.append(Spacer(1, 6 * mm))

    # One page per image: image left, result right
    images = sorted(SRC_DIR.glob("*.png"))
    if len(images) != 16:
        print(f"WARNING: expected 16 images, found {len(images)}")

    for img_path in images:
        name = img_path.name
        if name not in RESULTS:
            continue
        data = RESULTS[name]

        # Image (left column) — fit width ~180mm, preserve aspect
        img = RLImage(str(img_path), width=180 * mm, height=100 * mm, kind="proportional")
        # Wrap in paragraph so it sits cleanly in table
        img_para = Paragraph(f'<para align="center"><b>{name}</b></para>', styles["Normal"])
        img_cell = [img_para, Spacer(1, 2 * mm), img]

        # Result (right column) — text block
        plates = data["plates"]
        plate_lines = []
        for p in plates:
            tag = f'[{p["confidence"].upper()}]'
            plate_lines.append(
                f'<font color="#0f3460"><b>{p["text"]}</b></font> '
                f'<font color="#888888">{tag}</font><br/>'
                f'<font size="9" color="#555555">&nbsp;&nbsp;↳ {p["vehicle"]}</font>'
            )
        plate_html = "<br/><br/>".join(plate_lines) if plate_lines else "<i>No plates read</i>"

        result_para = []
        result_para.append(Paragraph("Detected plates", result_header))
        result_para.append(Paragraph(plate_html, result_body))
        result_para.append(Spacer(1, 3 * mm))
        result_para.append(Paragraph("Summary", result_header))
        result_para.append(Paragraph(data["summary"], result_body))
        result_para.append(Spacer(1, 2 * mm))
        result_para.append(Paragraph("Notes", result_header))
        result_para.append(Paragraph(data["notes"], result_body))

        # Two-column table: image | result
        table = Table(
            [[img_cell, result_para]],
            colWidths=[195 * mm, 75 * mm],
        )
        table.setStyle(TableStyle([
            ("VALIGN",          (0, 0), (-1, -1), "TOP"),
            ("BOX",             (0, 0), (-1, -1), 0.75, colors.HexColor("#cccccc")),
            ("INNERGRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ("LEFTPADDING",     (0, 0), (-1, -1), 6 * mm),
            ("RIGHTPADDING",    (0, 0), (-1, -1), 6 * mm),
            ("TOPPADDING",      (0, 0), (-1, -1), 6 * mm),
            ("BOTTOMPADDING",   (0, 0), (-1, -1), 6 * mm),
            ("BACKGROUND",      (1, 0), (1, 0), colors.HexColor("#f7f7fb")),
        ]))
        story.append(table)
        story.append(PageBreak())

    doc.build(story)
    print(f"PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")

    # Also dump JSON for machine-readable record
    import json
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": "MiniMax M3 (vision LLM, native plate read)",
                "generated_at": datetime.now().isoformat(),
                "source_dir": str(SRC_DIR),
                "image_count": len(RESULTS),
                "results": RESULTS,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"JSON written: {JSON_PATH}")


if __name__ == "__main__":
    build_pdf()