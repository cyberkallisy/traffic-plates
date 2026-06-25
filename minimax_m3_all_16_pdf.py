"""
Compact PDF: all 16 traffic-plate images + MiniMax M3 OCR results.
Layout:
  Page 1: Index — 4x4 grid of thumbnails with plate labels
  Pages 2-9: Detail pages, 2 images per page (image left, result right)
"""

import json
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, NextPageTemplate, PageBreak,
    Paragraph, Spacer, Image as RLImage, Table, TableStyle, KeepTogether,
    FrameBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

SRC_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_DIR = Path(r"C:/Users/gsash/Downloads/traffic-plates/results_paddle_minimax")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PDF_PATH = OUT_DIR / f"minimax_m3_all_16_images_{TIMESTAMP}.pdf"

# Same MiniMax M3 OCR results as the per-image PDF
RESULTS = {
    "1.png": {"plates": [{"text": "HR67B5432", "vehicle": "white van (center)", "confidence": "high"}, {"text": "HR 26KP 0074", "vehicle": "white Hyundai Creta (right)", "confidence": "high"}], "summary": "2 plates read clearly. Heavy traffic scene.", "notes": "HR67B5432 visible twice on the same van from a slightly oblique angle."},
    "2.png": {"plates": [{"text": "HR67B5432", "vehicle": "white van (center)", "confidence": "high"}], "summary": "1 plate read clearly.", "notes": "Same van as in image 1, but zoomed view."},
    "3.png": {"plates": [{"text": "HR67B5432", "vehicle": "white van (center)", "confidence": "high"}, {"text": "HR 26KP 0074", "vehicle": "white Hyundai Creta (right)", "confidence": "high"}], "summary": "2 plates read clearly. Same scene as 1.png.", "notes": "Both plates readable at this zoom level."},
    "4.png": {"plates": [{"text": "HR13AC7151", "vehicle": "black sedan (center)", "confidence": "high"}], "summary": "1 plate read clearly. Aerial view of busy junction.", "notes": "Other vehicles' plates too small/angled to read."},
    "5.png": {"plates": [{"text": "MH12LD2645", "vehicle": "white SUV (center-left)", "confidence": "high"}, {"text": "UP16BT0011", "vehicle": "white sedan (center)", "confidence": "medium"}, {"text": "MH04EF9465", "vehicle": "black Tata SUV (bottom)", "confidence": "high"}], "summary": "3 plates read. Highway scene with multiple lanes.", "notes": "UP16BT0011 partially occluded by lane divider."},
    "6.png": {"plates": [{"text": "HR26H0034", "vehicle": "black SUV (bottom-center)", "confidence": "medium"}], "summary": "1 plate readable. Heavy highway traffic.", "notes": "Several other vehicles but plates unreadable at this scale."},
    "invalid.png": {"plates": [{"text": "HR26AM9966", "vehicle": "black SUV (center)", "confidence": "high"}, {"text": "HR26CV0040", "vehicle": "white SUV (right)", "confidence": "medium"}], "summary": "2 plates read. Both follow HR-26 format.", "notes": "Filename suggests 'invalid' but plates ARE legible at this zoom."},
    "no.png": {"plates": [{"text": "UNREADABLE", "vehicle": "—", "confidence": "low"}], "summary": "No plate readable at this resolution/angle.", "notes": "Filename implies 'no plate readable'. Confirmed."},
    "no1.png": {"plates": [{"text": "UNREADABLE", "vehicle": "—", "confidence": "low"}], "summary": "No plate readable. Aerial traffic scene.", "notes": "Some vehicles have plate-shaped regions but text is too small/angled."},
    "no2.png": {"plates": [{"text": "UNREADABLE", "vehicle": "bus / auto-rickshaw", "confidence": "low"}], "summary": "No plate readable. Bus dominates frame.", "notes": "Bus plate obscured by windshield reflection."},
    "no4.png": {"plates": [{"text": "HR67B5432", "vehicle": "white van (left)", "confidence": "high"}, {"text": "HR 26KP 0074", "vehicle": "white Hyundai Creta (right)", "confidence": "high"}], "summary": "2 plates read clearly. Same scene as 1.png / 3.png.", "notes": "Filename implies 'no plate readable' but both plates ARE legible."},
    "not read.png": {"plates": [{"text": "NL61L04974", "vehicle": "Tata container truck (center)", "confidence": "medium"}], "summary": "1 plate readable on the dominant Tata truck.", "notes": "NL = Nagaland state code. Other vehicles too distant."},
    "yes.png": {"plates": [{"text": "HR05BH1839", "vehicle": "white Kia Seltos (center)", "confidence": "high"}], "summary": "1 plate read clearly. School-zone scene.", "notes": "Green plate EV in background is low-contrast."},
    "yes1.png": {"plates": [{"text": "HR91A2978", "vehicle": "white Maruti Ertiga (front)", "confidence": "high"}, {"text": "HR65B6500", "vehicle": "Mahindra Bolero pickup (left)", "confidence": "medium"}], "summary": "2 plates read. Busy school-zone traffic.", "notes": "HR91A2978 crystal-clear; HR65B6500 partially occluded."},
    "yes2.png": {"plates": [{"text": "HR05LR9761", "vehicle": "white Maruti Swift (center)", "confidence": "high"}], "summary": "1 plate read clearly.", "notes": "Filename 'yes' indicates clean readable case — confirmed."},
    "yes3.png": {"plates": [{"text": "HR05BH1839", "vehicle": "white Kia Seltos (center)", "confidence": "high"}], "summary": "1 plate read clearly. Same vehicle as yes.png.", "notes": "Kia Seltos with HR05 BH 1839 plate."},
}


def build_pdf():
    # Use BaseDocTemplate with explicit page templates for cleaner control
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        leftMargin=8 * mm,
        rightMargin=8 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm,
        title="Traffic Plates — MiniMax M3 OCR — All 16 Images",
        author="MiniMax M3 (vision LLM)",
    )

    # Two page templates: index page (portrait-ish layout) + detail pages
    page_w, page_h = landscape(A4)
    frame_index = Frame(
        doc.leftMargin, doc.bottomMargin,
        page_w - doc.leftMargin - doc.rightMargin,
        page_h - doc.topMargin - doc.bottomMargin,
        id="index", showBoundary=0,
    )
    frame_detail = Frame(
        doc.leftMargin, doc.bottomMargin,
        page_w - doc.leftMargin - doc.rightMargin,
        page_h - doc.topMargin - doc.bottomMargin,
        id="detail", showBoundary=0,
    )
    doc.addPageTemplates([
        PageTemplate(id="Index", frames=[frame_index]),
        PageTemplate(id="Detail", frames=[frame_detail]),
    ])

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    sub_style = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, spaceAfter=8, textColor=colors.grey)
    cap_style = ParagraphStyle("cap", parent=styles["Normal"], fontSize=8, alignment=TA_CENTER, leading=10)
    plate_text_style = ParagraphStyle("plate", parent=styles["Normal"], fontSize=10, leading=12)
    head_style = ParagraphStyle("head", parent=styles["Heading4"], fontSize=10, textColor=colors.HexColor("#16213e"), spaceAfter=2, leading=12)
    note_style = ParagraphStyle("note", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#555555"))

    story = []

    # ============================================================
    # INDEX PAGE — switch to Index template, build 4x4 grid as a Table
    # ============================================================
    story.append(NextPageTemplate("Index"))
    story.append(Paragraph("Traffic Plates — MiniMax M3 OCR — All 16 Images", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
        f"OCR engine: <b>MiniMax M3 (vision LLM, native read)</b> &nbsp;|&nbsp; "
        f"Source: test/New folder (16 images) &nbsp;|&nbsp; "
        f"Layout: <b>4×4 index grid + 2 images per detail page (8 detail pages)</b>",
        sub_style,
    ))

    # 4x4 grid of small thumbnails — each cell is a single Paragraph (Image is too tall for a small cell)
    image_names = sorted(RESULTS.keys())
    thumb_w = 65 * mm
    thumb_h = 37 * mm
    grid_data = []
    for row_start in range(0, len(image_names), 4):
        row = []
        for name in image_names[row_start:row_start + 4]:
            data = RESULTS[name]
            primary = data["plates"][0]["text"] if data["plates"] else "?"
            cell_html = (
                f'<para align="CENTER">'
                f'<b>{name}</b><br/>'
                f'<font color="#0f3460" size="9"><b>{primary}</b></font>'
                f'</para>'
            )
            row.append(Paragraph(cell_html, cap_style))
        # Pad to 4
        while len(row) < 4:
            row.append(Paragraph("", cap_style))
        grid_data.append(row)

    grid_table = Table(grid_data, colWidths=[68 * mm] * 4, rowHeights=[20 * mm] * 4)
    grid_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    story.append(grid_table)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "<b>Reading rate:</b> 13 of 16 images had readable plates (the 3 <i>no*.png</i> "
        "files were unreadable by design). Filename labels do NOT predict readability — "
        "<i>no4.png</i>, <i>invalid.png</i>, and <i>not read.png</i> all yielded clear "
        "plate reads at the image scale provided.",
        note_style,
    ))

    # ============================================================
    # DETAIL PAGES — switch to Detail template, 2 images per page
    # ============================================================
    story.append(NextPageTemplate("Detail"))
    story.append(PageBreak())

    img_w = 130 * mm
    img_h = 73 * mm

    for page_idx in range(0, len(image_names), 2):
        # Switch to detail template for each detail page
        if page_idx > 0:
            story.append(PageBreak())

        # Build two flat Image+Paragraph columns side by side using FrameBreak
        # Actually, easier: build a 2-column table where each cell is a SINGLE flowable (an Image)
        # then a separate row of paragraphs below
        slot1 = image_names[page_idx]
        slot2 = image_names[page_idx + 1] if page_idx + 1 < len(image_names) else None

        def img_block(name):
            return RLImage(str(SRC_DIR / name), width=img_w, height=img_h, kind="proportional")

        def result_block(name):
            data = RESULTS[name]
            plates = data["plates"]
            plate_lines = []
            for p in plates:
                tag = f'[{p["confidence"].upper()}]'
                plate_lines.append(
                    f'<font color="#0f3460" size="11"><b>{p["text"]}</b></font> '
                    f'<font color="#888888" size="8">{tag}</font><br/>'
                    f'<font size="8" color="#555555">&nbsp;&nbsp;↳ {p["vehicle"]}</font>'
                )
            plate_html = "<br/>".join(plate_lines) if plate_lines else "<i>No plates read</i>"

            # Return as a single nested Table that mimics the multi-paragraph block
            inner = Table(
                [[Paragraph("Plates read", head_style)],
                 [Paragraph(plate_html, plate_text_style)],
                 [Paragraph("Summary", head_style)],
                 [Paragraph(data["summary"], plate_text_style)],
                 [Paragraph("Notes", head_style)],
                 [Paragraph(data["notes"], note_style)]],
                colWidths=[130 * mm],
            )
            inner.setStyle(TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            return inner

        # Build outer 2-col table for this page
        # Row 0 = images, Row 1 = results
        img_row = [img_block(slot1), img_block(slot2) if slot2 else Paragraph("", styles["Normal"])]
        result_row = [result_block(slot1), result_block(slot2) if slot2 else Paragraph("", styles["Normal"])]

        # Filename caption row above images
        cap_row = [
            Paragraph(f'<b>{slot1}</b>', ParagraphStyle("cap2", parent=cap_style, alignment=TA_CENTER, fontSize=10)),
            Paragraph(f'<b>{slot2}</b>' if slot2 else "", ParagraphStyle("cap2", parent=cap_style, alignment=TA_CENTER, fontSize=10)),
        ]

        page_table = Table(
            [cap_row, img_row, result_row],
            colWidths=[138 * mm, 138 * mm],
        )
        page_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cccccc")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
            ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fafaff")),
        ]))
        story.append(page_table)

    doc.build(story)
    print(f"PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf()