"""
Build a self-contained HTML report from the YOLO11 + Awiros ANPR-OCR pipeline.

Reads:
  - summary.json
  - annotated/<file>.png  (annotated with bbox + OCR text)
  - crops/<stem>_plate<N>_<conf>.jpg
  - (and the original source images from --src for side-by-side comparison)

Embeds everything as base64 inline so the HTML is fully self-contained
and works offline (no relative paths, no file:// URLs).

Features:
  - Side-by-side: original source vs annotated
  - Plate crops inline under each detection
  - Click any image to open a lightbox modal (zoom + ESC to close)
  - Summary header with detector / OCR model info + totals

Usage:
    python build_report.py
    python build_report.py --run-dir <pipeline_run_dir>
"""

import argparse
import base64
import json
import re
from datetime import datetime
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Image widths (px) for re-encoding. JPEG quality.
SOURCE_W = 900
ANNOT_W = 700
CROP_W = 240
JPEG_Q = 85


def imencode_jpeg(path: Path, target_w: int, quality: int) -> bytes:
    """Read image, resize to target width, re-encode as JPEG, return bytes."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return b""
    h, w = img.shape[:2]
    if w > target_w:
        new_h = int(h * target_w / w)
        img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def b64_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def find_source_image(src_dir: Path, filename: str) -> Path | None:
    """Find the original source image (case-insensitive)."""
    if not src_dir:
        return None
    p = src_dir / filename
    if p.exists():
        return p
    for f in src_dir.iterdir():
        if f.is_file() and f.name.lower() == filename.lower():
            return f
    return None


def crop_label(det: dict, idx: int) -> str:
    """Build a short label like P1 YOLO=0.60."""
    return f"P{idx} YOLO={det['confidence']:.2f}"


def html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def render_image_card(src_url: str, ann_url: str, label: str) -> str:
    """Side-by-side source vs annotated."""
    return f'''
        <div class="pair">
          <figure class="img-card">
            <a href="#" class="zoom" data-src="{src_url}" data-label="{html_escape(label)} | source">
              <img src="{src_url}" alt="{html_escape(label)} source" loading="lazy">
            </a>
            <figcaption>Source</figcaption>
          </figure>
          <figure class="img-card">
            <a href="#" class="zoom" data-src="{ann_url}" data-label="{html_escape(label)} | annotated">
              <img src="{ann_url}" alt="{html_escape(label)} annotated" loading="lazy">
            </a>
            <figcaption>Annotated</figcaption>
          </figure>
        </div>'''


def render_detections_table(detections: list) -> str:
    """Table of detections + OCR readings."""
    if not detections:
        return '<p class="muted">No plates detected.</p>'
    rows = []
    for i, det in enumerate(detections, 1):
        ocr = det.get("ocr", {})
        text = ocr.get("text", "—") or "—"
        conf = ocr.get("confidence", 0.0)
        readable = bool(ocr.get("readable", False))
        cls = det.get("class_name", "?")
        mark = "OK" if readable else "--"
        cls_row = "ok" if readable else "bad"
        rows.append(
            f'<tr class="{cls_row}">'
            f'<td>{i}</td>'
            f'<td>{det["confidence"]:.2f}</td>'
            f'<td>{html_escape(cls)}</td>'
            f'<td class="plate">{html_escape(text)}</td>'
            f'<td>{conf:.2f}</td>'
            f'<td>{mark}</td>'
            f'<td class="bbox">{det["bbox_xyxy"][0]},{det["bbox_xyxy"][1]}'
            f' → {det["bbox_xyxy"][2]},{det["bbox_xyxy"][3]}</td>'
            f'</tr>'
        )
    return f'''
        <table class="det">
          <thead><tr>
            <th>#</th><th>YOLO</th><th>Class</th><th>OCR text</th>
            <th>OCR conf</th><th>Read</th><th>BBox (x1,y1 → x2,y2)</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>'''


def render_crops(crops_dir: Path, detections: list, stem: str) -> str:
    """Inline plate crops for each detection."""
    items = []
    for i, det in enumerate(detections, 1):
        crop_file = det.get("crop_file", "")
        if not crop_file:
            continue
        crop_path = crops_dir / crop_file
        if not crop_path.exists():
            continue
        jpeg = imencode_jpeg(crop_path, CROP_W, JPEG_Q)
        if not jpeg:
            continue
        url = b64_data_url(jpeg)
        ocr = det.get("ocr", {})
        text = ocr.get("text", "—") or "—"
        conf = ocr.get("confidence", 0.0)
        readable = bool(ocr.get("readable", False))
        cls = "ok" if readable else "bad"
        items.append(f'''
          <a href="#" class="crop-card zoom {cls}" data-src="{url}"
             data-label="{html_escape(stem)} plate {i} | {html_escape(text)}">
            <img src="{url}" alt="plate {i}" loading="lazy">
            <div class="crop-lbl">
              <span class="badge">{crop_label(det, i)}</span>
              <span class="plate">{html_escape(text)}</span>
              <span class="muted">({conf:.2f})</span>
            </div>
          </a>''')
    if not items:
        return ""
    return '<div class="crops">' + ''.join(items) + '</div>'


def build_html(run_dir: Path, src_dir: Path | None, out_path: Path) -> None:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    annotated_dir = run_dir / "annotated"
    crops_dir = run_dir / "crops"

    det = summary["detector"]
    ocr = summary["ocr"]
    totals = summary["totals"]

    # --- HTML head + style + lightbox script ---
    # NOTE: not an f-string — CSS uses single { } which would break Python f-strings.
    head_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANPR Report — YOLO11 + Awiros ANPR-OCR — __RUNNAME__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0e1116; color: #e6edf3;
  }
  h1, h2, h3 { margin: 0 0 12px 0; }
  h1 { font-size: 22px; }
  h2 { font-size: 18px; margin-top: 28px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  h3 { font-size: 15px; color: #8b949e; }
  a { color: #58a6ff; }
  .muted { color: #8b949e; }
  .header {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 24px;
  }
  .header dl { display: grid; grid-template-columns: 200px 1fr; gap: 4px 16px; margin: 8px 0 0 0; }
  .header dt { color: #8b949e; }
  .header dd { margin: 0; }
  .stats { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
  .stat {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 10px 14px; min-width: 130px;
  }
  .stat .num { font-size: 22px; font-weight: 600; color: #58a6ff; }
  .stat .lbl { font-size: 12px; color: #8b949e; margin-top: 2px; }
  .img-block {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px; margin: 18px 0;
  }
  .pair { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }
  .img-card { margin: 0; flex: 1 1 360px; }
  .img-card img {
    width: 100%; height: auto; display: block;
    border: 1px solid #30363d; border-radius: 4px; cursor: zoom-in;
    background: #000;
  }
  .img-card figcaption {
    text-align: center; font-size: 12px; color: #8b949e; margin-top: 4px;
  }
  .det {
    width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px;
  }
  .det th, .det td {
    padding: 6px 8px; text-align: left; border-bottom: 1px solid #21262d;
  }
  .det th { background: #0d1117; color: #8b949e; font-weight: 600; }
  .det tr.ok td { background: rgba(46, 160, 67, 0.08); }
  .det tr.bad td { background: rgba(248, 81, 73, 0.08); }
  .det .plate { font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 600; }
  .det .bbox { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; color: #8b949e; }
  .crops {
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px;
  }
  .crop-card {
    display: block; width: 240px; padding: 6px;
    background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
    text-decoration: none; color: inherit;
  }
  .crop-card.ok { border-color: rgba(46, 160, 67, 0.6); }
  .crop-card.bad { border-color: rgba(248, 81, 73, 0.6); }
  .crop-card img { width: 100%; height: auto; display: block; cursor: zoom-in; }
  .crop-lbl {
    margin-top: 6px; display: flex; gap: 6px; align-items: center;
    font-size: 12px; flex-wrap: wrap;
  }
  .crop-lbl .badge {
    background: #30363d; color: #e6edf3; padding: 1px 6px; border-radius: 3px;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .crop-lbl .plate { font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 600; }
  /* Lightbox */
  .lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92);
    z-index: 9999; align-items: center; justify-content: center;
    flex-direction: column; padding: 20px;
  }
  .lightbox.open { display: flex; }
  .lightbox img {
    max-width: 96vw; max-height: 88vh; object-fit: contain;
    background: #000;
  }
  .lightbox .lbl {
    color: #e6edf3; margin-top: 10px; font-size: 14px;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .lightbox .close {
    position: absolute; top: 14px; right: 18px;
    background: #30363d; color: #e6edf3; border: none;
    padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 14px;
  }
  footer { color: #8b949e; font-size: 12px; margin-top: 32px; text-align: center; }
</style>
</head>
<body>

<div class="header">
  <h1>ANPR Report — YOLO11 + Awiros ANPR-OCR</h1>
  <h3>Run: __RUNNAME__ · __START__ → __END__ · __SECS__s</h3>
  <dl>
    <dt>Detector</dt><dd>__DETVER__ · <code>__DETMODEL__</code></dd>
    <dt>OCR engine</dt><dd>__OCRENG__ · __OCRARCH__</dd>
    <dt>OCR weights</dt><dd><code>__OCRWEIGHTS__</code></dd>
    <dt>Device</dt><dd>__OCRDEV__</dd>
    <dt>Source folder</dt><dd><code>__SRCFOLDER__</code></dd>
  </dl>
  <div class="stats">
    <div class="stat"><div class="num">__SIMG__</div><div class="lbl">Images</div></div>
    <div class="stat"><div class="num">__SPLATES__</div><div class="lbl">Plates detected</div></div>
    <div class="stat"><div class="num">__SWPLATES__</div><div class="lbl">Images with plates</div></div>
    <div class="stat"><div class="num">__SREAD__</div><div class="lbl">Plates readable</div></div>
    <div class="stat"><div class="num">__SUNREAD__</div><div class="lbl">Plates unreadable</div></div>
  </div>
</div>
"""
    head = (
        head_template
        .replace("__RUNNAME__", html_escape(run_dir.name))
        .replace("__START__", html_escape(summary.get("run_started", "")))
        .replace("__END__", html_escape(summary.get("run_finished", "")))
        .replace("__SECS__", str(summary.get("total_seconds", 0)))
        .replace("__DETVER__", html_escape(det.get("version", "")))
        .replace("__DETMODEL__", html_escape(det.get("model_path", "")))
        .replace("__OCRENG__", html_escape(ocr.get("engine", "")))
        .replace("__OCRARCH__", html_escape(ocr.get("architecture", "")))
        .replace("__OCRWEIGHTS__", html_escape(ocr.get("weights", "")))
        .replace("__OCRDEV__", html_escape(str(ocr.get("device", ""))))
        .replace("__SRCFOLDER__", html_escape(summary.get("source_folder", "")))
        .replace("__SIMG__", str(totals["images"]))
        .replace("__SPLATES__", str(totals["plates_detected"]))
        .replace("__SWPLATES__", str(totals["images_with_plates"]))
        .replace("__SREAD__", str(totals["plates_ocr_readable"]))
        .replace("__SUNREAD__", str(totals["plates_ocr_unreadable"]))
    )
    html = [head]

    # --- Per-image blocks ---
    for img in summary["images"]:
        name = img.get("file", "")
        if "error" in img:
            html.append(f'<div class="img-block"><h2>{html_escape(name)}</h2>'
                        f'<p class="muted">ERROR: {html_escape(img["error"])}</p></div>')
            continue

        stem = Path(name).stem
        ann_path = annotated_dir / name
        src_path = find_source_image(src_dir, name) if src_dir else None

        ann_jpeg = imencode_jpeg(ann_path, ANNOT_W, JPEG_Q) if ann_path.exists() else b""
        src_jpeg = imencode_jpeg(src_path, SOURCE_W, JPEG_Q) if src_path and src_path.exists() else b""

        ann_url = b64_data_url(ann_jpeg) if ann_jpeg else ""
        src_url = b64_data_url(src_jpeg) if src_jpeg else ""

        plates = img["num_plates"]
        readable = img["num_ocr_readable"]
        size = img.get("size", [0, 0])
        ms_yolo = img.get("inference_ms_yolo", 0)
        ms_ocr = img.get("inference_ms_ocr_total", 0)

        html.append(f'<div class="img-block">')
        html.append(f'<h2>{html_escape(name)} '
                    f'<span class="muted">— {size[0]}×{size[1]} · '
                    f'{plates} plate{"s" if plates != 1 else ""} · '
                    f'{readable} readable · '
                    f'{ms_yolo:.0f}ms YOLO, {ms_ocr:.0f}ms OCR</span></h2>')

        if ann_url or src_url:
            if not src_url:
                html.append(f'<figure class="img-card"><a href="#" class="zoom" '
                            f'data-src="{ann_url}" data-label="{html_escape(name)} | annotated">'
                            f'<img src="{ann_url}" alt="{html_escape(name)}"></a>'
                            f'<figcaption>Annotated (source missing)</figcaption></figure>')
            elif not ann_url:
                html.append(f'<figure class="img-card"><a href="#" class="zoom" '
                            f'data-src="{src_url}" data-label="{html_escape(name)} | source">'
                            f'<img src="{src_url}" alt="{html_escape(name)}"></a>'
                            f'<figcaption>Source (annotated missing)</figcaption></figure>')
            else:
                html.append(render_image_card(src_url, ann_url, name))

        html.append(render_detections_table(img["detections"]))
        html.append(render_crops(crops_dir, img["detections"], stem))
        html.append('</div>')

    # --- Lightbox + JS ---
    html.append("""
<div id="lightbox" class="lightbox">
  <button class="close" id="lb-close">Close (Esc)</button>
  <img id="lb-img" src="" alt="">
  <div class="lbl" id="lb-lbl"></div>
</div>
<script>
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lb-img');
  const lbLbl = document.getElementById('lb-lbl');
  function openLb(src, lbl) {
    lbImg.src = src; lbLbl.textContent = lbl || '';
    lb.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
  function closeLb() {
    lb.classList.remove('open');
    lbImg.src = '';
    document.body.style.overflow = '';
  }
  document.querySelectorAll('a.zoom').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      openLb(a.dataset.src, a.dataset.label);
    });
  });
  document.getElementById('lb-close').addEventListener('click', closeLb);
  lb.addEventListener('click', e => { if (e.target === lb) closeLb(); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && lb.classList.contains('open')) closeLb();
  });
</script>

<footer>
  Generated """ + datetime.now().isoformat(timespec="seconds") + """
  · YOLO 11 detector + Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC) · CPU
</footer>

</body>
</html>
""")

    out_path.write_text("".join(html), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Build HTML report from ANPR pipeline run")
    p.add_argument(
        "--run-dir",
        default=r"C:/Users/gsash/Downloads/test/New folder/images/yolo11_awiros_ocr_run",
        help="Pipeline run folder (containing summary.json, annotated/, crops/)",
    )
    p.add_argument(
        "--src",
        default=r"C:/Users/gsash/Downloads/test/New folder/images",
        help="Original source images folder (for side-by-side comparison)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: <run-dir>/report.html)",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    src_dir = Path(args.src) if args.src else None
    out_path = Path(args.out) if args.out else run_dir / "report.html"

    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")

    print(f"[REPORT] run dir: {run_dir}")
    print(f"[REPORT] src dir: {src_dir}")
    print(f"[REPORT] out    : {out_path}")

    build_html(run_dir, src_dir, out_path)

    size_kb = out_path.stat().st_size / 1024
    print(f"[REPORT] wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()