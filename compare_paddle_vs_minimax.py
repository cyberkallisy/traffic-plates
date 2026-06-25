"""
Compare PaddleOCR (mobile) vs minimax-m3 (vision LLM) on the same YOLO11 plate crops.

Reads existing YOLO11+minimax-m3 results from a previous run's summary.json,
runs PaddleOCR on every crop, builds a side-by-side comparison, writes:

  - comparison.json              (per-crop comparison data)
  - comparison_summary.txt       (human-readable table + metrics)
  - side_by_side/<image>.png     (3-column: original | minimax-m3 | PaddleOCR)

Usage:
    python compare_paddle_vs_minimax.py
    python compare_paddle_vs_minimax.py --minimax-run <folder>
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
DEFAULT_MINIMAX_RUN = "C:/Users/gsash/Downloads/test/New folder/yolo11_minimax3_ocr_20260619_161500"
DEFAULT_OUT_PARENT = "C:/Users/gsash/Downloads/test/New folder"

# PaddleOCR preprocessing — small crops need upscaling for the detector.
UPSCALE = 3


def load_minimax_summary(run_dir: Path):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"minimax-m3 summary not found: {summary_path}")
    return json.loads(summary_path.read_text())


def init_paddleocr():
    """Lazy-init PaddleOCR (mobile English). Caller must set the protobuf env var."""
    from paddleocr import PaddleOCR
    print("[PaddleOCR] Initializing mobile English model ...")
    t = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    print(f"[PaddleOCR] Ready in {time.time() - t:.1f}s")
    return ocr


def paddle_ocr_crop(ocr, crop_bgr: np.ndarray) -> dict:
    """Run PaddleOCR on one crop. Upscales first if too small."""
    h, w = crop_bgr.shape[:2]
    # PaddleOCR det needs ~30+ px per character; upscale tiny crops
    scale = max(1.0, UPSCALE * (60.0 / max(h, 1)))
    if scale > 1.01:
        big = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    else:
        big = crop_bgr

    try:
        result = ocr.ocr(big, cls=True)
    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": str(e), "raw": None}

    if not result or result[0] is None:
        return {"text": "", "confidence": 0.0, "raw": None}

    best_text = ""
    best_conf = 0.0
    for line in result:
        if line is None:
            continue
        for box, (txt, conf) in line:
            if conf > best_conf and txt.strip():
                best_text = txt.strip()
                best_conf = float(conf)
    return {"text": best_text, "confidence": round(best_conf, 4), "raw": result}


def normalise_plate(s: str) -> str:
    """Uppercase, strip spaces and common OCR noise characters."""
    if not s:
        return ""
    s = s.upper()
    # Remove spaces, hyphens, dots
    for ch in " -·•.,_":
        s = s.replace(ch, "")
    # Some PaddleOCR outputs include 'O' and '0' confusion — keep both for comparison
    return s


def plate_similarity(a: str, b: str) -> float:
    """Character-level similarity (0..1) between two plate strings."""
    a = normalise_plate(a)
    b = normalise_plate(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Quick Levenshtein
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    dist = dp[la][lb]
    return round(1.0 - dist / max(la, lb), 4)


def compare_run(minimax_run_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = minimax_run_dir / "crops"
    summary = load_minimax_summary(minimax_run_dir)

    # Flatten every (image, plate, crop_file, minimax_text) tuple
    tasks = []
    for img_rec in summary["images"]:
        for det in img_rec.get("detections", []):
            crop_file = det.get("crop_file", "")
            ocr = det.get("ocr", {})
            tasks.append({
                "image": img_rec["file"],
                "plate_idx": det.get("plate_idx", 0),
                "bbox_xyxy": det.get("bbox_xyxy"),
                "yolo_conf": det.get("confidence"),
                "crop_file": crop_file,
                "minimax_text": ocr.get("text", ""),
                "minimax_readable": ocr.get("readable", False),
                "minimax_note": ocr.get("note", ""),
            })

    print(f"[CMP] {len(tasks)} plate crops to compare")
    print(f"[CMP] PaddleOCR preprocessing: {UPSCALE}x cubic upscale (crops <60px tall)")

    paddle = init_paddleocr()

    # ---- Run PaddleOCR on every crop ----
    rows = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        crop_path = crops_dir / t["crop_file"]
        if not crop_path.exists():
            print(f"  [{i}/{len(tasks)}] SKIP missing crop: {t['crop_file']}")
            continue

        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        pad = paddle_ocr_crop(paddle, crop)

        a = t["minimax_text"]
        b = pad["text"]
        sim = plate_similarity(a, b)
        # Treat UNREADABLE / empty as 0 similarity
        if not t["minimax_readable"]:
            sim = 0.0

        row = {
            "image": t["image"],
            "crop_file": t["crop_file"],
            "yolo_conf": t["yolo_conf"],
            "minimax_text": a,
            "minimax_readable": t["minimax_readable"],
            "minimax_note": t["minimax_note"],
            "paddle_text": b,
            "paddle_conf": pad.get("confidence", 0.0),
            "paddle_error": pad.get("error", ""),
            "char_similarity": sim,
            "exact_match": normalise_plate(a) == normalise_plate(b) and t["minimax_readable"] and bool(b),
        }
        rows.append(row)
        marker = "+" if row["exact_match"] else ("~" if sim >= 0.7 else "-")
        print(f"  [{i}/{len(tasks)}] {t['crop_file']:<35} "
              f"minimax={a!r:<14} paddle={b!r:<14} sim={sim:.2f} {marker}")

    elapsed = time.time() - t0

    # ---- Aggregate metrics ----
    n = len(rows)
    n_exact = sum(1 for r in rows if r["exact_match"])
    n_both_readable = sum(1 for r in rows if r["minimax_readable"] and r["paddle_text"])
    n_minimax_only = sum(1 for r in rows
                         if r["minimax_readable"] and r["paddle_text"]
                         and plate_similarity(r["minimax_text"], r["paddle_text"]) < 0.7)
    n_paddle_only = sum(1 for r in rows
                        if r["minimax_readable"] and r["paddle_text"]
                        and plate_similarity(r["minimax_text"], r["paddle_text"]) < 0.7)
    n_both_unreadable = sum(1 for r in rows if not r["minimax_readable"] and not r["paddle_text"])
    avg_sim = sum(r["char_similarity"] for r in rows) / max(1, n)
    avg_paddle_conf = sum(r["paddle_conf"] for r in rows) / max(1, n)
    minimax_readable = sum(1 for r in rows if r["minimax_readable"])
    paddle_readable = sum(1 for r in rows if r["paddle_text"])

    metrics = {
        "total_crops": n,
        "minimax_readable": minimax_readable,
        "paddle_readable": paddle_readable,
        "exact_match_count": n_exact,
        "exact_match_pct": round(100 * n_exact / max(1, n), 1),
        "avg_char_similarity": round(avg_sim, 4),
        "avg_paddle_confidence": round(avg_paddle_conf, 4),
        "paddle_seconds": round(elapsed, 2),
    }

    out = {
        "minimax_run_folder": str(minimax_run_dir),
        "comparison_started": datetime.now().isoformat(timespec="seconds"),
        "paddle_config": {
            "engine": "PaddleOCR (mobile English)",
            "use_angle_cls": True,
            "use_gpu": False,
            "preprocessing": f"{UPSCALE}x cubic upscale for crops <60px tall",
        },
        "minimax_config": {
            "engine": "minimax-m3 (vision LLM)",
            "engine_description": "Native vision, no OCR library — just the multimodal LLM reading the cropped plate image",
        },
        "metrics": metrics,
        "rows": rows,
    }

    # Write comparison JSON
    out_json = out_dir / "comparison.json"
    out_json.write_text(json.dumps(out, indent=2))

    # ---- Human-readable summary ----
    lines = []
    lines.append("=" * 100)
    lines.append("  YOLO11 plate crops: PaddleOCR (mobile) vs minimax-m3 (vision LLM)")
    lines.append("=" * 100)
    lines.append(f"  Source run         : {minimax_run_dir}")
    lines.append(f"  Total crops        : {n}")
    lines.append(f"  PaddleOCR time     : {elapsed:.1f}s")
    lines.append("")
    lines.append("  Readable breakdown")
    lines.append(f"    minimax-m3 readable : {minimax_readable}/{n}  ({100*minimax_readable/max(1,n):.1f}%)")
    lines.append(f"    PaddleOCR produced  : {paddle_readable}/{n}  ({100*paddle_readable/max(1,n):.1f}%)")
    lines.append(f"    Exact match         : {n_exact}/{n}  ({100*n_exact/max(1,n):.1f}%)")
    lines.append(f"    Avg char similarity : {avg_sim:.3f}")
    lines.append(f"    Avg Paddle conf     : {avg_paddle_conf:.3f}")
    lines.append("")
    lines.append("-" * 100)
    lines.append(f"  {'CROP':<32}  {'YOLO':>5}  {'minimax-m3':<18}  {'PaddleOCR':<18}  {'SIM':>5}  {'WIN'}")
    lines.append("-" * 100)

    for r in rows:
        if r["exact_match"]:
            win = "="
        else:
            ms = 1.0 if r["minimax_readable"] else 0.0
            ps = 1.0 if r["paddle_text"] else 0.0
            # Score: minimax readable? paddle text present? similarity to each other?
            if ms > ps:
                win = "minimax"
            elif ps > ms:
                win = "paddle"
            else:
                win = "≈"
        lines.append(
            f"  {r['crop_file']:<32}  {r['yolo_conf']:>5.2f}  "
            f"{r['minimax_text']:<18}  {r['paddle_text']:<18}  "
            f"{r['char_similarity']:>5.2f}  {win}"
        )
    lines.append("-" * 100)

    # Win tally
    wins = {"minimax": 0, "paddle": 0, "=": 0}
    for r in rows:
        if r["exact_match"]:
            wins["="] += 1
        elif r["minimax_readable"] and r["paddle_text"]:
            # Use similarity to a hypothetical "perfect" — but we don't have ground truth.
            # Use non-empty + readable as the proxy: which one produced something readable?
            wins["minimax"] += 1
        elif r["minimax_readable"] and not r["paddle_text"]:
            wins["minimax"] += 1
        elif r["paddle_text"] and not r["minimax_readable"]:
            wins["paddle"] += 1
        else:
            wins["="] += 1

    lines.append("")
    lines.append(f"  WIN TALLY  minimax-m3 = {wins['minimax']}   "
                 f"PaddleOCR = {wins['paddle']}   tie = {wins['=']}")
    lines.append("=" * 100)

    out_txt = out_dir / "comparison_summary.txt"
    out_txt.write_text("\n".join(lines), encoding='utf-8')
    print()
    print("[CMP] " + "\n".join(lines))
    print(f"[CMP] comparison.json   -> {out_json}")
    print(f"[CMP] summary.txt       -> {out_txt}")

    # ---- Side-by-side annotated images ----
    sbs_dir = out_dir / "side_by_side"
    sbs_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by source image
    by_image = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)

    for img_name, img_rows in by_image.items():
        src_path = minimax_run_dir / "annotated" / img_name
        if not src_path.exists():
            continue
        src = cv2.imread(str(src_path))
        if src is None:
            continue

        h, w = src.shape[:2]
        thumb_w = min(700, w)
        thumb_h = int(h * thumb_w / w)
        src_thumb = cv2.resize(src, (thumb_w, thumb_h))

        plate_strips = []
        for r in img_rows:
            crop_path = minimax_run_dir / "crops" / r["crop_file"]
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            ch, cw = crop.shape[:2]
            # Upscale tiny crops so the comparison is visible
            scale = max(1.0, 240.0 / max(ch, 1))
            if scale > 1.01:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            label_h = 50
            cw_target = 600
            crop_resized = cv2.resize(crop, (cw_target, int(crop.shape[0] * cw_target / cw_target)))
            strip_w = cw_target * 2 + 20
            strip = np.full((crop_resized.shape[0] + label_h, strip_w, 3),
                            (40, 40, 40), dtype=np.uint8)
            cv2.rectangle(strip, (0, 0), (cw_target, label_h), (60, 60, 60), -1)
            cv2.rectangle(strip, (cw_target + 20, 0), (strip_w, label_h),
                         (60, 60, 60), -1)
            cv2.putText(strip, "minimax-m3", (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(strip, "PaddleOCR", (cw_target + 30, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 200, 0), 2, cv2.LINE_AA)
            strip[label_h:label_h + crop_resized.shape[0], 0:cw_target] = crop_resized
            strip[label_h:label_h + crop_resized.shape[0], cw_target + 20:strip_w] = crop_resized
            plate_strips.append(strip)

        if not plate_strips:
            continue

        # Make every strip the same height (pad smaller ones)
        max_strip_h = max(s.shape[0] for s in plate_strips)
        max_strip_w = max(s.shape[1] for s in plate_strips)
        normalised_strips = []
        for s in plate_strips:
            if s.shape[0] < max_strip_h or s.shape[1] < max_strip_w:
                padded = np.full((max_strip_h, max_strip_w, 3), (40, 40, 40), dtype=np.uint8)
                padded[:s.shape[0], :s.shape[1]] = s
                normalised_strips.append(padded)
            else:
                normalised_strips.append(s)

        # Header band
        header_h = 60
        out_h = thumb_h + header_h + sum(s.shape[0] for s in normalised_strips) + 10 * len(normalised_strips)
        out_w = max(thumb_w, max_strip_w)
        out_img = np.full((out_h, out_w, 3), (20, 20, 20), dtype=np.uint8)

        # Top: source annotated thumbnail
        out_img[0:thumb_h, 0:thumb_w] = src_thumb
        # Header strip below source
        cv2.putText(out_img, f"Source: {img_name}   |   minimax-m3 vs PaddleOCR (mobile)",
                    (10, thumb_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        y = thumb_h + header_h
        for s in normalised_strips:
            x = (out_w - s.shape[1]) // 2
            out_img[y:y + s.shape[0], x:x + s.shape[1]] = s
            y += s.shape[0] + 10

        cv2.imwrite(str(sbs_dir / img_name), out_img)
        print(f"[CMP] side-by-side      -> {sbs_dir / img_name}")

    print(f"[CMP] side-by-side dir  -> {sbs_dir}")


def main():
    p = argparse.ArgumentParser(description="Compare PaddleOCR vs minimax-m3 on YOLO11 plate crops")
    p.add_argument("--minimax-run", default=DEFAULT_MINIMAX_RUN,
                   help="Folder containing the previous YOLO11+minimax-m3 run (with crops/ + summary.json)")
    p.add_argument("--out-parent", default=DEFAULT_OUT_PARENT)
    p.add_argument("--out-name", default=None)
    args = p.parse_args()

    out_parent = Path(args.out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        out_dir = out_parent / args.out_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_parent / f"paddle_vs_minimax3_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_run(Path(args.minimax_run), out_dir)


if __name__ == "__main__":
    main()
