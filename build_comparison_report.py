"""Build the final PaddleOCR vs minimax-m3 comparison report.

Reads:
  - paddleocr_full_<ts>.json      (PaddleOCR per-image results)
  - minimax_ocr.json              (LLM OCR results, populated below)

Writes:
  - comparison_report.txt         (human-readable table)
  - comparison_report.json        (full structured data)

Mode: full-image OCR (no YOLO, no crop, no annotated images).
"""

import json
import re
from datetime import datetime
import sys
from pathlib import Path

INPUT_DIR = Path(r"C:/Users/gsash/Downloads/test/New folder")

# minimax-m3 OCR results — populated by the multimodal model after looking at
# each full image directly. List every plate that was readable from the
# full street scene (no YOLO crop, no preprocessing, no annotation).
MINIMAX_OCR = {
    "1.png":       ["HR67B5432", "HR26H0024"],
    "2.png":       ["HR67B5432"],
    "3.png":       ["HR67B5432", "HR26H0034"],
    "4.png":       ["HR134C"],
    "5.png":       [],
    "6.png":       ["MH12LK4115", "DL5CAE1226"],
    "invalid.png": ["HR05BH1839"],
    "no.png":      [],
    "no1.png":     ["DL3C4126"],
    "no2.png":     ["MH12LK4115"],
    "no4.png":     ["HR67B5432", "HR26H0034"],
    "not read.png":["NL61A8934"],
    "yes.png":     ["HR05BH1839"],
    "yes1.png":    ["HR91A2978", "HR65B6500"],
    "yes2.png":    ["HR05LR9761"],
    "yes3.png":    ["HR05BH1839"],
}

INDIAN_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
INDIAN_BH = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


def is_valid(s: str) -> bool:
    return bool(INDIAN_RE.match(s) or INDIAN_BH.match(s))


def char_sim(a: str, b: str) -> float:
    a, b = re.sub(r"[^A-Z0-9]", "", a.upper()), re.sub(r"[^A-Z0-9]", "", b.upper())
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
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
    return round(1.0 - dp[la][lb] / max(la, lb), 3)


def best_paddle_match(paddle_lines, llm_plates):
    """Among PaddleOCR lines, pick the one that best matches any LLM plate.
    Returns (best_text, best_conf, best_sim, valid_flag) or (None,...)."""
    if not paddle_lines or not llm_plates:
        return None, 0.0, 0.0, False
    best = None
    for ln in paddle_lines:
        t = re.sub(r"[^A-Z0-9]", "", ln["text"].upper())
        if not t:
            continue
        for lp in llm_plates:
            sim = char_sim(t, lp)
            score = (1.0 if is_valid(t) else 0.0) + sim + 0.5 * ln["conf"]
            if best is None or score > best["score"]:
                best = {"text": t, "conf": ln["conf"], "sim": sim,
                        "valid": is_valid(t), "score": score,
                        "matched_to": lp}
    if not best:
        return None, 0.0, 0.0, False
    return best["text"], best["conf"], best["sim"], best["valid"]


def main():
    paddle_files = sorted(INPUT_DIR.glob("paddleocr_full_*.json"))
    if not paddle_files:
        raise SystemExit("No paddleocr_full_*.json found in " + str(INPUT_DIR))
    paddle_path = paddle_files[-1]
    paddle_data = json.loads(paddle_path.read_text())
    paddle_by_img = {r["image"]: r for r in paddle_data["per_image"]}

    rows = []
    for image, llm_plates in MINIMAX_OCR.items():
        p = paddle_by_img.get(image, {})
        paddle_lines = p.get("lines", [])
        paddle_seconds = p.get("ocr_seconds", 0.0)
        paddle_best_text, paddle_best_conf, paddle_best_sim, paddle_valid = \
            best_paddle_match(paddle_lines, llm_plates)

        # For each LLM plate, find the closest paddle line
        per_plate = []
        for lp in llm_plates:
            best_line = None
            for ln in paddle_lines:
                t = re.sub(r"[^A-Z0-9]", "", ln["text"].upper())
                sim = char_sim(t, lp) if t else 0.0
                if best_line is None or sim > best_line["sim"]:
                    best_line = {"paddle_text": t, "paddle_conf": ln["conf"],
                                 "sim": sim, "valid": is_valid(t) if t else False}
            if best_line is None:
                best_line = {"paddle_text": "", "paddle_conf": 0.0,
                             "sim": 0.0, "valid": False}
            per_plate.append({
                "minimax_text": lp,
                "minimax_valid": is_valid(lp),
                **best_line,
                "exact_match": best_line["paddle_text"] == lp,
            })

        rows.append({
            "image": image,
            "paddle_lines_seen": len(paddle_lines),
            "paddle_best_match": paddle_best_text,
            "paddle_best_conf": paddle_best_conf,
            "paddle_best_sim": paddle_best_sim,
            "paddle_seconds": paddle_seconds,
            "minimax_plates": llm_plates,
            "minimax_n": len(llm_plates),
            "per_plate": per_plate,
        })

    # ---------- aggregate metrics ----------
    n_img = len(rows)
    n_minimax_with = sum(1 for r in rows if r["minimax_n"] > 0)
    n_paddle_with_any = sum(1 for r in rows if r["paddle_lines_seen"] > 0)
    n_exact = sum(1 for r in rows for pp in r["per_plate"] if pp["exact_match"])
    n_total_pairs = sum(r["minimax_n"] for r in rows)
    avg_sim = round(sum(pp["sim"] for r in rows for pp in r["per_plate"])
                    / max(1, n_total_pairs), 3)
    n_minimax_valid = sum(1 for r in rows for pp in r["per_plate"] if pp["minimax_valid"])
    n_paddle_valid = sum(1 for r in rows for pp in r["per_plate"]
                         if pp["valid"] and pp["paddle_text"])

    # ---------- write JSON ----------
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mode": "FULL IMAGE OCR — NO YOLO, NO CROP, NO ANNOTATION",
        "engines": {
            "paddleocr": "PaddleOCR 2.7 (mobile, en, angle-cls, CPU)",
            "llm":       "minimax-m3 (vision LLM, multimodal)",
        },
        "input_dir": str(INPUT_DIR),
        "paddle_source": str(paddle_path),
        "summary": {
            "images_total":         n_img,
            "llm_readable_images":  n_minimax_with,
            "paddle_lines_any":     n_paddle_with_any,
            "plate_pairs_total":    n_total_pairs,
            "llm_plates_valid_IN":  n_minimax_valid,
            "paddle_lines_valid_IN":n_paddle_valid,
            "exact_matches":        n_exact,
            "avg_char_similarity":  avg_sim,
            "paddle_total_seconds": round(sum(r["paddle_seconds"] for r in rows), 2),
        },
        "per_image": rows,
    }
    json_out = INPUT_DIR / "comparison_report.json"
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ---------- write TXT ----------
    lines = []
    lines.append("=" * 100)
    lines.append("  License-plate OCR comparison — FULL IMAGE mode (NO YOLO, NO CROP)")
    lines.append("=" * 100)
    lines.append("  PaddleOCR : PaddleOCR 2.7 (mobile, en, angle-cls, CPU)")
    lines.append("  LLM       : minimax-m3 (vision LLM, multimodal)")
    lines.append(f"  Source    : {INPUT_DIR}")
    lines.append("")
    sm = report["summary"]
    lines.append("  SUMMARY")
    lines.append("-" * 100)
    lines.append(f"    Images total              : {sm['images_total']}")
    lines.append(f"    Images where LLM read >=1  : {sm['llm_readable_images']}")
    lines.append(f"    Images where Paddle saw >=1: {sm['paddle_lines_any']}")
    lines.append(f"    Plate pairs (LLM plates)  : {sm['plate_pairs_total']}")
    lines.append(f"    LLM plates valid IN format: {sm['llm_plates_valid_IN']}")
    lines.append(f"    Paddle lines valid IN fmt : {sm['paddle_lines_valid_IN']}")
    lines.append(f"    Exact match (engine=engine): {sm['exact_matches']}")
    lines.append(f"    Avg char similarity       : {sm['avg_char_similarity']}")
    lines.append(f"    Paddle total time         : {sm['paddle_total_seconds']}s")
    lines.append("")
    lines.append("-" * 100)
    lines.append(f"  {'IMAGE':<14}  {'PAD_LINES':>8}  "
                 f"{'minimax-m3':<28}  {'PaddleOCR closest':<22}  {'SIM':>5}")
    lines.append("-" * 100)
    for r in rows:
        llm_str = ", ".join(r["minimax_plates"]) if r["minimax_plates"] else "(none)"
        # Show the best paddle line matched against the FIRST LLM plate (if any)
        if r["minimax_plates"]:
            pp = r["per_plate"][0]
            pad_str = f"{pp['paddle_text']} ({pp['paddle_conf']:.2f})" \
                      if pp["paddle_text"] else "(no line)"
            sim = pp["sim"]
        else:
            pad_str = f"{r['paddle_best_match']} ({r['paddle_best_conf']:.2f})" \
                      if r["paddle_best_match"] else "(no line)"
            sim = r["paddle_best_sim"]
        lines.append(f"  {r['image']:<14}  {r['paddle_lines_seen']:>8}  "
                     f"{llm_str:<28}  {pad_str:<22}  {sim:>5.2f}")
    lines.append("-" * 100)

    # Verdict
    if sm["llm_readable_images"] > sm["paddle_lines_any"]:
        lines.append("")
        lines.append("  VERDICT: minimax-m3 wins on full-image OCR (no crop).")
        lines.append(f"           minimax-m3 read {sm['llm_readable_images']}/{sm['images_total']} images,")
        lines.append(f"           PaddleOCR (mobile, full-image) detected any text on only "
                     f"{sm['paddle_lines_any']}/{sm['images_total']}.")
    else:
        lines.append("")
        lines.append("  VERDICT: engines comparable on full-image OCR.")
    lines.append("=" * 100)

    txt_out = INPUT_DIR / "comparison_report.txt"
    txt_out.write_text("\n".join(lines), encoding='utf-8')

    sys.stdout.buffer.write("\n".join(lines).encode('utf-8'))
    print()
    print(f"[REPORT] JSON  -> {json_out}")
    print(f"[REPORT] TXT   -> {txt_out}")


if __name__ == "__main__":
    main()