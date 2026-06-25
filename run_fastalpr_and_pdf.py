"""
Run fast-alpr on the 16 traffic-plates test images AND build a PDF report.

PDF layout (per page, landscape A4):
  +-----------------------------------------------------------------+
  |  [ image ]                  |  Result from fast-alpr            |
  |  (annotated, scaled to fit) |  - File: 1.png                   |
  |                             |  - Plates: 2                     |
  |                             |  - #1: text="WN8410OH"           |
  |                             |         bbox=[918,472,990,494]   |
  |                             |         det_conf=0.79            |
  |                             |         ocr_conf=0.51            |
  |                             |         region=France (0.30)     |
  |                             |  - #2: text="NR67443" ...        |
  |                             |  - Elapsed: 0.45s                |
  +-----------------------------------------------------------------+

Output:
  - <OUT>/annotated/<name>.jpg          (fast-alpr annotated images)
  - <OUT>/json/<name>.json              (per-image results)
  - <OUT>/crops/<name>__<i>__<text>.jpg  (per-plate crops)
  - <OUT>/summary.json                  (overall summary)
  - <OUT>/fast_alpr_results.pdf         (final PDF report) <-- THE DELIVERABLE

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ^
    "C:/Users/gsash/Downloads/Facial-recognition/venv/Scripts/python.exe" ^
        run_fastalpr_and_pdf.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force pure-Python protobuf before any paddle/torch import (paranoia)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
from PIL import Image as PILImage  # for PDF rendering

from fast_alpr import ALPR

# reportlab is what we use for the PDF
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.enums import TA_LEFT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")
OUT_ROOT = Path(r"C:/Users/gsash/Downloads/traffic-plates/fastalpr_results_20260622_120000")

DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"
CONF_THRESH = 0.25

EXT_OK = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(input_dir: Path) -> list[Path]:
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXT_OK
    )
    return files


def run_fast_alpr(images: list[Path]) -> list[dict]:
    """Run fast-alpr on every image, save annotated/crops/json, return summary."""
    (OUT_ROOT / "annotated").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "json").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "crops").mkdir(parents=True, exist_ok=True)

    print(f"[fast-alpr] {len(images)} images in {INPUT_DIR}")
    print(f"[fast-alpr] loading detector={DETECTOR_MODEL} ocr={OCR_MODEL} device=cpu ...")
    t0 = time.time()
    alpr = ALPR(
        detector_model=DETECTOR_MODEL,
        ocr_model=OCR_MODEL,
        detector_conf_thresh=CONF_THRESH,
        ocr_device="cpu",
    )
    print(f"[fast-alpr] ALPR ready in {time.time() - t0:.1f}s")

    summary: list[dict] = []
    grand_t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        t1 = time.time()
        try:
            draw = alpr.draw_predictions(str(img_path))
        except Exception as e:
            print(f"[fast-alpr] {img_path.name}: ERROR {type(e).__name__}: {e}")
            rec = {
                "file": img_path.name,
                "error": f"{type(e).__name__}: {e}",
                "plates": [],
                "elapsed_s": round(time.time() - t1, 3),
            }
            (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
                json.dumps(rec, indent=2)
            )
            summary.append(rec)
            continue

        results = draw.results
        annotated_path = OUT_ROOT / "annotated" / (img_path.stem + ".jpg")
        cv2.imwrite(str(annotated_path), draw.image)

        plates: list[dict] = []
        for i, r in enumerate(results):
            bb = r.detection.bounding_box
            x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
            det_conf = float(r.detection.confidence)
            label = str(r.detection.label)

            if r.ocr is not None:
                text = str(r.ocr.text)
                ocr_conf_raw = r.ocr.confidence
                if isinstance(ocr_conf_raw, (list, tuple)):
                    ocr_conf = float(sum(ocr_conf_raw) / max(1, len(ocr_conf_raw)))
                else:
                    ocr_conf = float(ocr_conf_raw)
                region = r.ocr.region
                region_conf = r.ocr.region_confidence
            else:
                text = ""
                ocr_conf = 0.0
                region = None
                region_conf = None

            plates.append({
                "idx": i,
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": det_conf,
                "text": text,
                "ocr_conf": ocr_conf,
                "region": region,
                "region_conf": region_conf,
            })

            # Crop from annotated (so user can see bbox)
            h, w = draw.image.shape[:2]
            cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
            cx2 = min(w, int(x2)); cy2 = min(h, int(y2))
            crop = draw.image[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                safe_text = "".join(
                    ch if ch.isalnum() else "_" for ch in text
                )[:24] or "notext"
                crop_path = OUT_ROOT / "crops" / f"{img_path.stem}__{i:02d}__{safe_text}.jpg"
                cv2.imwrite(str(crop_path), crop)

        elapsed = time.time() - t1
        rec = {
            "file": img_path.name,
            "num_plates": len(plates),
            "plates": plates,
            "elapsed_s": round(elapsed, 3),
        }
        summary.append(rec)
        (OUT_ROOT / "json" / (img_path.stem + ".json")).write_text(
            json.dumps(rec, indent=2)
        )
        print(
            f"[fast-alpr] [{idx:02d}/{len(images)}] {img_path.name}: "
            f"{len(plates)} plate(s) "
            f"texts={[p['text'] for p in plates]} "
            f"({elapsed:.2f}s)"
        )

    total_elapsed = time.time() - grand_t0
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "engine": "fast-alpr",
        "detector": DETECTOR_MODEL,
        "ocr": OCR_MODEL,
        "conf_thresh": CONF_THRESH,
        "num_images": len(images),
        "total_elapsed_s": round(total_elapsed, 3),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": summary,
    }, indent=2))

    total_plates = sum(len(r["plates"]) for r in summary)
    print(f"\n[fast-alpr] DONE — {len(images)} images, {total_plates} plates, "
          f"{total_elapsed:.1f}s total")
    print(f"[fast-alpr] results → {OUT_ROOT}")
    return summary


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------
def _fmt_conf(c) -> str:
    try:
        return f"{float(c):.2f}"
    except Exception:
        return str(c)


def _fmt_region(region, conf) -> str:
    if region is None:
        return "-"
    if conf is None:
        return str(region)
    return f"{region} ({_fmt_conf(conf)})"


def build_pdf(summary: list[dict], pdf_path: Path) -> None:
    """One page per image. Two columns: image | fast-alpr result."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=0.8 * cm, rightMargin=0.8 * cm,
        topMargin=0.8 * cm, bottomMargin=0.8 * cm,
        title="Fast-ALPR Test Results — 16 Traffic Plate Images",
        author="Hermes Agent",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBig", parent=styles["Title"],
        fontSize=20, leading=24, textColor=HexColor("#1f2937"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, textColor=grey, leading=12,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, leading=16, textColor=HexColor("#1d4ed8"),
        spaceBefore=4, spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=HexColor("#111827"),
    )
    mono = ParagraphStyle(
        "Mono", parent=styles["Code"],
        fontSize=9, leading=12, fontName="Courier",
    )

    story = []

    # ---- Cover page ---------------------------------------------------------
    total_plates = sum(len(r.get("plates", [])) for r in summary)
    total_imgs = len(summary)
    with_plate = sum(1 for r in summary if r.get("plates"))
    no_plate = total_imgs - with_plate

    story.append(Paragraph("Fast-ALPR Test Results", title_style))
    story.append(Paragraph(
        f"Detector: <b>{DETECTOR_MODEL}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"OCR: <b>{OCR_MODEL}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Conf ≥ {CONF_THRESH}",
        meta_style,
    ))
    story.append(Paragraph(
        f"Input: <font name='Courier'>{INPUT_DIR}</font> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        meta_style,
    ))
    story.append(Spacer(1, 0.4 * cm))

    summary_table = Table(
        [
            ["Total images", f"{total_imgs}"],
            ["Images with plates detected", f"{with_plate}"],
            ["Images with no plate detected", f"{no_plate}"],
            ["Total plates found", f"{total_plates}"],
        ],
        colWidths=[7 * cm, 4 * cm],
        hAlign="LEFT",
    )
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#f3f4f6")),
        ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#111827")),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#9ca3af")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, HexColor("#d1d5db")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5 * cm))

    # Cover-page compact list
    cover_rows = [["#", "Image", "Plates", "Top text", "Det.conf"]]
    for i, r in enumerate(summary, 1):
        plates = r.get("plates", [])
        top_text = plates[0]["text"] if plates else "(none)"
        top_conf = _fmt_conf(plates[0]["det_conf"]) if plates else "-"
        cover_rows.append([str(i), r["file"], str(len(plates)), top_text, top_conf])

    cover_tbl = Table(cover_rows, colWidths=[1 * cm, 5 * cm, 1.5 * cm, 8 * cm, 2 * cm], hAlign="LEFT")
    cover_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f9fafb")]),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#9ca3af")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cover_tbl)

    # ---- Per-image pages ----------------------------------------------------
    page_w, page_h = landscape(A4)
    img_max_w = 12.5 * cm       # left column
    img_max_h = 16.5 * cm       # vertical room on landscape A4 (~21cm tall, minus margins)
    text_col_w = page_w - 0.8 * cm - 0.8 * cm - img_max_w - 0.4 * cm  # right column

    for i, rec in enumerate(summary, 1):
        story.append(PageBreak())

        file_name = rec["file"]
        annotated_path = OUT_ROOT / "annotated" / f"{Path(file_name).stem}.jpg"
        if not annotated_path.exists():
            annotated_path = INPUT_DIR / file_name  # fallback to original

        # Image (scaled to fit)
        try:
            with PILImage.open(annotated_path) as pil:
                iw, ih = pil.size
            # convert px -> points (reportlab uses 72dpi, screenshots ~96dpi)
            pts_w = iw * 0.75
            pts_h = ih * 0.75
            scale = min(img_max_w / pts_w, img_max_h / pts_h, 1.0)
            draw_w = pts_w * scale
            draw_h = pts_h * scale
            rl_img = RLImage(str(annotated_path), width=draw_w, height=draw_h)
        except Exception as e:
            rl_img = Paragraph(f"(image not available: {e})", body)

        # Right column: details
        right_flow = []
        right_flow.append(Paragraph(
            f"<b>Image {i} of {total_imgs}:</b> "
            f"<font name='Courier'>{file_name}</font>", h2
        ))

        if "error" in rec:
            right_flow.append(Paragraph(
                f"<b>Error:</b> <font color='#dc2626'>{rec['error']}</font>", body
            ))
            right_flow.append(Paragraph(
                f"Elapsed: {_fmt_conf(rec.get('elapsed_s', 0))}s", body
            ))
        else:
            plates = rec.get("plates", [])
            right_flow.append(Paragraph(
                f"<b>Plates detected:</b> {len(plates)} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Elapsed:</b> {_fmt_conf(rec.get('elapsed_s', 0))}s",
                body,
            ))

            if not plates:
                right_flow.append(Paragraph(
                    "<i>No plates found in this image.</i>", body
                ))
            else:
                rows = [["#", "Text", "Det.conf", "OCR.conf", "Region (conf)", "BBox (x1,y1,x2,y2)"]]
                for p in plates:
                    bbox_str = str(p["bbox"])
                    rows.append([
                        str(p["idx"] + 1),
                        Paragraph(p["text"] or "<i>(empty)</i>", body),
                        _fmt_conf(p["det_conf"]),
                        _fmt_conf(p["ocr_conf"]),
                        Paragraph(_fmt_region(p.get("region"), p.get("region_conf")), body),
                        Paragraph(f"<font name='Courier' size='8'>{bbox_str}</font>", body),
                    ])
                tbl = Table(
                    rows,
                    colWidths=[
                        0.6 * cm,           # #
                        3.4 * cm,           # text
                        1.4 * cm,           # det_conf
                        1.4 * cm,           # ocr_conf
                        2.8 * cm,           # region
                        4.4 * cm,           # bbox
                    ],
                    hAlign="LEFT",
                )
                tbl.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1d4ed8")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                        [HexColor("#ffffff"), HexColor("#f9fafb")]),
                    ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#9ca3af")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                right_flow.append(tbl)

        # Wrap right column into a fixed-width cell using a 1-cell Table
        right_container = Table([[right_flow]], colWidths=[text_col_w])
        right_container.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#d1d5db")),
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#ffffff")),
        ]))

        two_col = Table(
            [[rl_img, right_container]],
            colWidths=[img_max_w + 0.4 * cm, text_col_w],
            hAlign="LEFT",
        )
        two_col.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(two_col)

    doc.build(story)
    print(f"[pdf] wrote {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    images = collect_images(INPUT_DIR)
    if len(images) != 16:
        print(f"[warn] expected 16 images, found {len(images)} — proceeding anyway")

    summary = run_fast_alpr(images)

    pdf_path = OUT_ROOT / "fast_alpr_results.pdf"
    build_pdf(summary, pdf_path)

    print("\n=== Summary ===")
    print(f"Images processed : {len(summary)}")
    print(f"Total plates     : {sum(len(r.get('plates', [])) for r in summary)}")
    print(f"Output folder    : {OUT_ROOT}")
    print(f"PDF              : {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())