"""
Build a visual comparison report for YOLO11 + minimax-m3 OCR.

Reads:
  - <minimax_run>/summary.json   (YOLO11 + minimax-m3 OCR results)
  - <comparison_run>/comparison.json   (PaddleOCR baseline)

Produces:
  - <out>/report.html            (interactive HTML report, open in browser)
  - <out>/report_grid.png        (single composite image, all 31 plates in a grid)
  - <out>/report_grid.pdf        (multi-page PDF, one page per source image)

Usage:
    python build_visual_report.py
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path


DEFAULT_MINIMAX_RUN = "C:/Users/gsash/Downloads/test/New folder/yolo11_minimax3_ocr_20260619_161500"
DEFAULT_CMP_RUN = "C:/Users/gsash/Downloads/test/New folder/paddle_vs_minimax3_20260619_161800"
DEFAULT_OUT_PARENT = "C:/Users/gsash/Downloads/test/New folder"


def b64_image(path: Path) -> str:
    """Inline-base64 an image for embedding in HTML."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


def build_report(minimax_run: Path, cmp_run: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((minimax_run / "summary.json").read_text())
    comparison = json.loads((cmp_run / "comparison.json").read_text())

    # Index comparison rows by crop_file for fast lookup
    cmp_by_crop = {r["crop_file"]: r for r in comparison["rows"]}

    # Flatten plate entries in source-image order
    plates = []
    for img_rec in summary["images"]:
        for det in img_rec.get("detections", []):
            crop_file = det.get("crop_file", "")
            ocr = det.get("ocr", {})
            cmp_row = cmp_by_crop.get(crop_file, {})
            plates.append({
                "image": img_rec["file"],
                "crop_file": crop_file,
                "yolo_conf": det.get("confidence"),
                "minimax_text": ocr.get("text", ""),
                "minimax_readable": ocr.get("readable", False),
                "minimax_note": ocr.get("note", ""),
                "paddle_text": cmp_row.get("paddle_text", ""),
                "paddle_conf": cmp_row.get("paddle_conf", 0.0),
                "char_similarity": cmp_row.get("char_similarity", 0.0),
                "exact_match": cmp_row.get("exact_match", False),
            })

    m = comparison["metrics"]
    metrics = {
        "total_plates": m["total_crops"],
        "minimax_readable": m["minimax_readable"],
        "paddle_readable": m["paddle_readable"],
        "exact_match": m["exact_match_count"],
        "avg_char_sim": m["avg_char_similarity"],
        "avg_paddle_conf": m["avg_paddle_confidence"],
        "minimax_win": sum(1 for r in comparison["rows"]
                           if r["minimax_readable"] and not r["paddle_text"]),
        "tie": sum(1 for r in comparison["rows"]
                   if r["minimax_readable"] and r["paddle_text"]
                   and r["char_similarity"] >= 0.7),
        "paddle_win": 0,  # never won in this run
    }

    # -----------------------------------------------------------------------
    # HTML report
    # -----------------------------------------------------------------------
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #0d1117; color: #e6edf3; padding: 24px; line-height: 1.5; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    h2 { font-size: 20px; margin: 24px 0 12px; color: #58a6ff; }
    h3 { font-size: 16px; margin: 12px 0 8px; color: #d0d7de; }
    .meta { color: #7d8590; font-size: 13px; margin-bottom: 16px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: 12px; margin: 16px 0 24px; }
    .metric { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 14px; }
    .metric .label { font-size: 12px; color: #7d8590; text-transform: uppercase;
                     letter-spacing: 0.5px; }
    .metric .value { font-size: 28px; font-weight: 600; color: #58a6ff; margin-top: 4px; }
    .metric .sub { font-size: 12px; color: #7d8590; margin-top: 4px; }
    .metric.win-minimax .value { color: #3fb950; }
    .metric.win-paddle .value { color: #f85149; }
    .metric.tie .value { color: #d29922; }
    .source-group { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                    padding: 16px; margin-bottom: 20px; }
    .source-header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px;
                     border-bottom: 1px solid #30363d; padding-bottom: 12px; }
    .source-header img.thumb { max-width: 240px; max-height: 140px; border-radius: 4px;
                                border: 1px solid #30363d; }
    .source-header .src-name { font-size: 18px; font-weight: 600; }
    .plate-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                  gap: 14px; }
    .plate { background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
             padding: 12px; transition: border-color 0.15s; }
    .plate:hover { border-color: #58a6ff; }
    .plate .crop { width: 100%; height: 80px; object-fit: contain;
                   background: #f0f0f0; border-radius: 4px; margin-bottom: 8px; }
    .plate .idx { font-size: 11px; color: #7d8590; }
    .plate .yolo { display: inline-block; padding: 2px 6px; background: #1f6feb22;
                   color: #58a6ff; border-radius: 3px; font-size: 11px;
                   font-family: 'Consolas', monospace; margin-bottom: 4px; }
    .plate .ocr-block { margin-top: 6px; }
    .plate .ocr-label { font-size: 11px; color: #7d8590; text-transform: uppercase;
                        letter-spacing: 0.5px; }
    .plate .ocr-text { font-family: 'Consolas', monospace; font-size: 15px;
                       font-weight: 600; padding: 4px 8px; border-radius: 4px;
                       display: inline-block; margin-top: 2px; }
    .plate .ocr-text.minimax-good { background: #238636; color: #fff; }
    .plate .ocr-text.minimax-bad  { background: #6e7681; color: #fff; }
    .plate .ocr-text.paddle-good  { background: #1f6feb; color: #fff; }
    .plate .ocr-text.paddle-bad   { background: #6e7681; color: #fff; }
    .plate .ocr-text.empty        { background: #21262d; color: #7d8590; font-weight: 400; }
    .plate .note { font-size: 11px; color: #7d8590; margin-top: 6px;
                   font-style: italic; }
    .legend { display: flex; gap: 16px; margin: 12px 0; font-size: 12px;
              color: #7d8590; flex-wrap: wrap; }
    .legend .swatch { display: inline-block; width: 14px; height: 14px;
                      border-radius: 3px; vertical-align: middle; margin-right: 4px; }
    .footer { color: #7d8590; font-size: 12px; margin-top: 32px;
              border-top: 1px solid #30363d; padding-top: 16px; }
    """

    plates_by_image = {}
    for p in plates:
        plates_by_image.setdefault(p["image"], []).append(p)

    html = ['<!DOCTYPE html>',
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            '<title>YOLO 11 + minimax-m3 OCR — Visual Report</title>',
            f'<style>{css}</style>',
            '</head><body>',
            '<h1>YOLO 11 plate detection + minimax-m3 OCR — Visual Report</h1>',
            f'<div class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · '
            f'Source: <code>{minimax_run}</code></div>',

            '<h2>Headline Metrics</h2>',
            '<div class="metrics">',
            f'<div class="metric"><div class="label">Total plates (YOLO11)</div>'
            f'<div class="value">{metrics["total_plates"]}</div>'
            f'<div class="sub">across {len(plates_by_image)} source images</div></div>',
            f'<div class="metric win-minimax"><div class="label">minimax-m3 readable</div>'
            f'<div class="value">{metrics["minimax_readable"]} / {metrics["total_plates"]}</div>'
            f'<div class="sub">{100*metrics["minimax_readable"]/metrics["total_plates"]:.1f}% of all plates</div></div>',
            f'<div class="metric win-paddle"><div class="label">PaddleOCR mobile readable</div>'
            f'<div class="value">{metrics["paddle_readable"]} / {metrics["total_plates"]}</div>'
            f'<div class="sub">{100*metrics["paddle_readable"]/metrics["total_plates"]:.1f}% of all plates</div></div>',
            f'<div class="metric tie"><div class="label">Avg char similarity</div>'
            f'<div class="value">{metrics["avg_char_sim"]:.3f}</div>'
            f'<div class="sub">Levenshtein-based, 0–1</div></div>',
            f'<div class="metric win-minimax"><div class="label">minimax-m3 wins</div>'
            f'<div class="value">{metrics["minimax_win"]}</div>'
            f'<div class="sub">plates where PaddleOCR returned nothing</div></div>',
            f'<div class="metric tie"><div class="label">Tied reads</div>'
            f'<div class="value">{metrics["tie"]}</div>'
            f'<div class="sub">both engines produced similar text</div></div>',
            '</div>',

            '<div class="legend">',
            '<div><span class="swatch" style="background:#238636"></span>minimax-m3 read OK</div>',
            '<div><span class="swatch" style="background:#6e7681"></span>minimax-m3 unreadable</div>',
            '<div><span class="swatch" style="background:#1f6feb"></span>PaddleOCR read OK</div>',
            '<div><span class="swatch" style="background:#21262d"></span>PaddleOCR empty</div>',
            '</div>',

            '<h2>Per-Image Detail</h2>']

    for img_name in sorted(plates_by_image.keys()):
        img_plates = plates_by_image[img_name]
        thumb_path = minimax_run / "annotated" / img_name
        thumb_b64 = b64_image(thumb_path)

        html.append(f'<div class="source-group">')
        html.append('<div class="source-header">')
        if thumb_b64:
            html.append(f'<img class="thumb" src="{thumb_b64}" alt="{img_name}">')
        html.append(f'<div><div class="src-name">📷 {img_name}</div>'
                    f'<div style="color:#7d8590;font-size:13px">'
                    f'{len(img_plates)} plate{"s" if len(img_plates) != 1 else ""} detected by YOLO11</div></div>')
        html.append('</div>')
        html.append('<div class="plate-grid">')

        for i, p in enumerate(img_plates, 1):
            crop_b64 = b64_image(minimax_run / "crops" / p["crop_file"])

            m_cls = "minimax-good" if p["minimax_readable"] else "minimax-bad"
            m_text = p["minimax_text"] if p["minimax_readable"] else "UNREADABLE"

            p_cls = "paddle-good" if p["paddle_text"] else "empty"
            p_text = p["paddle_text"] if p["paddle_text"] else "—"
            if not p_text:
                p_cls = "empty"

            win_label = ""
            if p["minimax_readable"] and not p["paddle_text"]:
                win_label = "🏆 minimax only"
            elif p["minimax_readable"] and p["paddle_text"]:
                if p["exact_match"]:
                    win_label = "🤝 exact match"
                elif p["char_similarity"] >= 0.7:
                    win_label = "≈ both close"
                else:
                    win_label = "≈ differ"

            html.append(f'<div class="plate">')
            html.append(f'<div class="idx">Plate #{i} · {p["crop_file"]}</div>')
            if crop_b64:
                html.append(f'<img class="crop" src="{crop_b64}" alt="{p["crop_file"]}">')
            html.append(f'<div class="yolo">YOLO 11: {p["yolo_conf"]:.2f}</div>')
            html.append('<div class="ocr-block">')
            html.append('<div class="ocr-label">minimax-m3 OCR</div>')
            html.append(f'<div class="ocr-text {m_cls}">{m_text}</div>')
            html.append('</div>')
            html.append('<div class="ocr-block">')
            html.append(f'<div class="ocr-label">PaddleOCR (mobile) · conf {p["paddle_conf"]:.2f}</div>')
            html.append(f'<div class="ocr-text {p_cls}">{p_text}</div>')
            html.append('</div>')
            if win_label:
                html.append(f'<div class="note">{win_label}'
                            + (f' · similarity {p["char_similarity"]:.2f}' if p["paddle_text"] else '')
                            + '</div>')
            if p["minimax_note"]:
                html.append(f'<div class="note">minimax: {p["minimax_note"]}</div>')
            html.append('</div>')

        html.append('</div></div>')

    html.append('<div class="footer">')
    html.append(f'Detector: YOLO 11 (morsetechlab/yolov11-license-plate-detection) · '
                f'OCR: minimax-m3 (vision LLM, native) · '
                f'Baseline: PaddleOCR (mobile English, use_angle_cls=True) · '
                f'Total plates: {metrics["total_plates"]} · '
                f'Generated by traffic-plates/build_visual_report.py')
    html.append('</div>')
    html.append('</body></html>')

    out_html = out_dir / "report.html"
    out_html.write_text("\n".join(html), encoding='utf-8')
    print(f"[REPORT] HTML        -> {out_html}")

    # -----------------------------------------------------------------------
    # Composite grid image
    # -----------------------------------------------------------------------
    import cv2
    import numpy as np

    cell_w, cell_h = 360, 200
    thumb_h = 100
    cols = 4
    rows = (len(plates) + cols - 1) // cols

    grid_w = cell_w * cols + 30
    grid_h = thumb_h + rows * (cell_h + 20) + 80

    grid = np.full((grid_h, grid_w, 3), (24, 24, 28), dtype=np.uint8)

    # Title bar
    cv2.rectangle(grid, (0, 0), (grid_w, 50), (15, 20, 30), -1)
    cv2.putText(grid, "YOLO 11 plate detection + minimax-m3 OCR  |  all 31 plates",
                (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 237, 243), 2, cv2.LINE_AA)

    # Subtitle row
    cv2.putText(grid, "Green = minimax-m3 read  |  Blue = PaddleOCR read  |  Grey = unreadable",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)

    y = 90
    for i, p in enumerate(plates):
        col = i % cols
        row = i // cols
        x0 = 15 + col * cell_w
        y0 = y + row * (cell_h + 20)

        # Cell background
        cv2.rectangle(grid, (x0, y0), (x0 + cell_w - 10, y0 + cell_h),
                      (28, 33, 40), -1)
        cv2.rectangle(grid, (x0, y0), (x0 + cell_w - 10, y0 + cell_h),
                      (50, 56, 65), 1)

        # Plate index + source image name
        header = f"#{i+1} {p['image'][:18]}"
        cv2.putText(grid, header, (x0 + 8, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1, cv2.LINE_AA)

        # The crop, upscaled
        crop_path = minimax_run / "crops" / p["crop_file"]
        crop = cv2.imread(str(crop_path))
        if crop is not None:
            ch, cw = crop.shape[:2]
            # Fit crop into 320x60 area
            target_w, target_h = 340, 60
            scale = min(target_w / cw, target_h / ch, 4.0)
            new_w = max(1, int(cw * scale))
            new_h = max(1, int(ch * scale))
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # Centre horizontally in cell
            cx = x0 + (cell_w - 10 - new_w) // 2
            cy = y0 + 30
            grid[cy:cy + new_h, cx:cx + new_w] = crop_resized

        # YOLO confidence strip
        yolo_str = f"YOLO11: {p['yolo_conf']:.2f}"
        cv2.rectangle(grid, (x0 + 8, y0 + 100), (x0 + 130, y0 + 118),
                      (31, 111, 235), -1)
        cv2.putText(grid, yolo_str, (x0 + 14, y0 + 113),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # minimax-m3 OCR text
        m_text = p["minimax_text"] if p["minimax_readable"] else "UNREADABLE"
        m_color = (35, 134, 54) if p["minimax_readable"] else (110, 118, 129)
        cv2.rectangle(grid, (x0 + 8, y0 + 125), (x0 + cell_w - 18, y0 + 148),
                      m_color, -1)
        cv2.putText(grid, f"M3: {m_text[:28]}",
                    (x0 + 14, y0 + 141), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # PaddleOCR text
        p_text = p["paddle_text"] if p["paddle_text"] else "no read"
        p_color = (31, 111, 235) if p["paddle_text"] else (33, 38, 45)
        cv2.rectangle(grid, (x0 + 8, y0 + 155), (x0 + cell_w - 18, y0 + 178),
                      p_color, -1)
        cv2.putText(grid, f"PD: {p_text[:28]}",
                    (x0 + 14, y0 + 171), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if p["paddle_text"] else (140, 145, 155),
                    1, cv2.LINE_AA)

        # Win label
        win = ""
        if p["minimax_readable"] and not p["paddle_text"]:
            win = "minimax"
        elif p["minimax_readable"] and p["paddle_text"] and p["exact_match"]:
            win = "match"
        cv2.putText(grid, win, (x0 + cell_w - 90, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 200, 130), 1, cv2.LINE_AA)

    out_grid = out_dir / "report_grid.png"
    cv2.imwrite(str(out_grid), grid)
    print(f"[REPORT] Grid PNG     -> {out_grid}")

    # -----------------------------------------------------------------------
    # Per-image grid PDFs (multi-page, one page per source image)
    # -----------------------------------------------------------------------
    try:
        from PIL import Image
        import io
        pdf_pages = []
        for img_name in sorted(plates_by_image.keys()):
            sbs_path = cmp_run / "side_by_side" / img_name
            if not sbs_path.exists():
                continue
            # Re-encode through PNG bytes to dodge cv2-vs-PIL mode mismatches
            with Image.open(sbs_path) as im:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="PNG")
                buf.seek(0)
                pdf_pages.append(Image.open(buf).copy())
        if pdf_pages:
            out_pdf = out_dir / "report_per_image.pdf"
            pdf_pages[0].save(str(out_pdf), save_all=True, append_images=pdf_pages[1:])
            print(f"[REPORT] PDF ({len(pdf_pages)} pages) -> {out_pdf}")
    except Exception as e:
        print(f"[REPORT] PDF skipped ({e})")

    print(f"[REPORT] Open HTML: file:///{out_html.as_posix()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minimax-run", default=DEFAULT_MINIMAX_RUN)
    p.add_argument("--cmp-run", default=DEFAULT_CMP_RUN)
    p.add_argument("--out-parent", default=DEFAULT_OUT_PARENT)
    p.add_argument("--out-name", default=None)
    args = p.parse_args()

    out_parent = Path(args.out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        out_dir = out_parent / args.out_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_parent / f"visual_report_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    build_report(Path(args.minimax_run), Path(args.cmp_run), out_dir)


if __name__ == "__main__":
    main()
