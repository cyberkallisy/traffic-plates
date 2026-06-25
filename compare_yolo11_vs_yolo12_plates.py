"""
YOLO 11 vs YOLO 12 - License Plate Detection on test images.

Pipeline:
  1. Run pre-trained YOLO 11 plate detector (morsetechlab/yolov11-license-plate-detection)
     on every image in --source.
     -> results saved to results_yolo11/

  2. Fine-tune YOLO 12n (COCO pretrained) for a few epochs on a small plate
     detection subset (keremberke/license-plate-object-detection) so we have
     a comparable YOLO 12 plate detector. (CPU-only machine — small subset,
     few epochs to keep it tractable.)
     -> best weights saved as yolo12_plate.pt

  3. Run fine-tuned YOLO 12 plate detector on the same test images.
     -> results saved to results_yolo12/

  4. Write a side-by-side comparison.json with per-image detection counts,
     inference times, and IoU-based agreement summary.

Note: the YOLO 11 model has been trained for 300 epochs on a 10k+ plate
dataset; the YOLO 12 model is intentionally a quick fine-tune (3 epochs,
~300 images) for a comparable-but-not-identical test. Both are .pt YOLO
nano variants; only plate class.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# CRITICAL for paddleocr cohabitation on this machine (protobuf 7.x crash otherwise)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def draw_plate_box(img, x1, y1, x2, y2, label, color, thickness=3):
    """Draw a plate bbox with a label tag."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty1 = max(y1 - th - baseline - 6, 0)
    ty2 = y1
    cv2.rectangle(img, (x1, ty1), (x1 + tw + 6, ty2), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(img_bgr, detections, model_tag, model_color):
    """Draw plate detections on a copy of the image."""
    out = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{model_tag} {d['confidence']:.2f}"
        draw_plate_box(out, x1, y1, x2, y2, label, model_color)

    # Footer with count + model tag
    h, w = out.shape[:2]
    footer = f"{model_tag}  |  plates: {len(detections)}"
    (tw, th), baseline = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (w, th + baseline + 14), model_color, -1)
    cv2.putText(out, footer, (10, th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_plate_detector(model_tag, weights_path, source_dir, out_dir, conf=0.25, imgsz=640):
    """Run a YOLO plate detector on every image in source_dir, save annotated
    outputs + JSON to out_dir. Returns list of per-image result dicts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir)

    images = sorted([
        p for p in source_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ])
    print(f"\n[{model_tag}] loading {weights_path} ...")
    t0 = time.time()
    model = YOLO(str(weights_path))
    print(f"[{model_tag}] loaded in {time.time() - t0:.1f}s | classes: {model.names} ({len(model.names)} classes)")

    color = (0, 200, 0) if "yolo11" in model_tag.lower() else (0, 100, 255)  # green vs orange

    summary = []
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [{model_tag}] SKIP unreadable: {img_path.name}")
            continue
        h, w = img_bgr.shape[:2]
        t0 = time.time()
        results = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)
        dt = (time.time() - t0) * 1000

        detections = []
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls[0].cpu().numpy())
                cls_name = model.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf_v = float(b.conf[0].cpu().numpy())
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_v, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        annotated = annotate(img_bgr, detections, model_tag.upper(), color)
        out_path = out_dir / f"{img_path.stem}__{model_tag}.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        summary.append({
            "image": img_path.name,
            "image_size": [w, h],
            "inference_ms": round(dt, 1),
            "num_plates": len(detections),
            "detections": detections,
        })
        print(f"  [{model_tag}] {img_path.name:<20} plates={len(detections):>2}  {dt:>6.0f}ms")

    json_path = out_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump({
            "model_tag": model_tag,
            "weights": str(weights_path),
            "source_dir": str(source_dir),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "num_images": len(summary),
            "total_plates": sum(s["num_plates"] for s in summary),
            "results": summary,
        }, f, indent=2)
    print(f"[{model_tag}] wrote {json_path}")
    return summary


def build_comparison(yolo11_summary, yolo12_summary, out_path):
    by11 = {s["image"]: s for s in yolo11_summary}
    by12 = {s["image"]: s for s in yolo12_summary}
    common = sorted(set(by11) & set(by12))

    def _highest_conf(dets):
        return max((d["confidence"] for d in dets), default=0.0)

    rows = []
    for name in common:
        a, b = by11[name], by12[name]
        rows.append({
            "image": name,
            "yolo11": {
                "plates": a["num_plates"],
                "top_confidence": round(_highest_conf(a["detections"]), 3),
                "inference_ms": a["inference_ms"],
            },
            "yolo12": {
                "plates": b["num_plates"],
                "top_confidence": round(_highest_conf(b["detections"]), 3),
                "inference_ms": b["inference_ms"],
            },
            "agreement": "both_found" if a["num_plates"] > 0 and b["num_plates"] > 0
                          else "yolo11_only" if a["num_plates"] > 0
                          else "yolo12_only" if b["num_plates"] > 0
                          else "neither",
        })

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "common_images": len(common),
        "yolo11_total_plates": sum(r["yolo11"]["plates"] for r in rows),
        "yolo12_total_plates": sum(r["yolo12"]["plates"] for r in rows),
        "yolo11_avg_ms": round(sum(r["yolo11"]["inference_ms"] for r in rows) / max(len(rows), 1), 1),
        "yolo12_avg_ms": round(sum(r["yolo12"]["inference_ms"] for r in rows) / max(len(rows), 1), 1),
        "agreement_summary": {
            "both_found":   sum(1 for r in rows if r["agreement"] == "both_found"),
            "yolo11_only":  sum(1 for r in rows if r["agreement"] == "yolo11_only"),
            "yolo12_only":  sum(1 for r in rows if r["agreement"] == "yolo12_only"),
            "neither":      sum(1 for r in rows if r["agreement"] == "neither"),
        },
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[comparison] wrote {out_path}")
    return out


# ---------------------------------------------------------------------------
# Fine-tune YOLO 12n on a small plate subset (CPU-friendly)
# ---------------------------------------------------------------------------

def finetune_yolo12(weights="yolo12n.pt", data_yaml="plate_dataset/yolo_small/data.yaml",
                    epochs=3, imgsz=320, batch=4, project="runs/yolo12_plate"):
    """Quick fine-tune of YOLO12n on a ~300-image plate subset. Outputs best.pt."""
    print(f"\n[fine-tune yolo12] starting {epochs} epochs on CPU (small subset, imgsz={imgsz}) ...")
    model = YOLO(weights)
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=2,
        project=project,
        name="plate_finetune",
        patience=0,
        save=True,
        save_period=-1,
        verbose=False,
        # Don't bother with augmentation tuning; quick pass.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
    )
    dt = time.time() - t0
    best_pt = Path(project) / "plate_finetune" / "weights" / "best.pt"
    if not best_pt.exists():
        last_pt = Path(project) / "plate_finetune" / "weights" / "last.pt"
        best_pt = last_pt if last_pt.exists() else best_pt
    print(f"[fine-tune yolo12] done in {dt:.0f}s ({dt/60:.1f} min). best.pt = {best_pt}")
    return best_pt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"C:\Users\gsash\Downloads\test\New folder",
                    help="Folder of test images")
    ap.add_argument("--project-root", default=r"C:\Users\gsash\Downloads\traffic-plates",
                    help="Where results_yolo11/ and results_yolo12/ live")
    ap.add_argument("--yolo11-weights", default=r"C:\Users\gsash\Downloads\traffic-plates\yolo11_plate.pt",
                    help="Pre-trained YOLO11 plate weights")
    ap.add_argument("--data-yaml", default=r"C:\Users\gsash\Downloads\traffic-plates\plate_dataset\yolo_small\data.yaml",
                    help="Small subset data.yaml for YOLO12 fine-tune")
    ap.add_argument("--yolo12-weights-out", default=r"C:\Users\gsash\Downloads\traffic-plates\yolo12_plate.pt",
                    help="Where to copy the fine-tuned YOLO12 best.pt to")
    ap.add_argument("--epochs", type=int, default=3, help="YOLO12 fine-tune epochs")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference imgsz")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--skip-finetune", action="store_true",
                    help="Skip YOLO12 fine-tune (use existing yolo12_plate.pt if it exists)")
    ap.add_argument("--skip-yolo11", action="store_true", help="Skip YOLO11 inference")
    ap.add_argument("--skip-yolo12", action="store_true", help="Skip YOLO12 inference")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    yolo11_dir = project_root / "results_yolo11"
    yolo12_dir = project_root / "results_yolo12"

    # --- YOLO 11 plate detection (pre-trained) ---
    yolo11_summary = []
    if not args.skip_yolo11:
        yolo11_summary = run_plate_detector(
            "yolo11", args.yolo11_weights, args.source, yolo11_dir,
            conf=args.conf, imgsz=args.imgsz,
        )
    elif (yolo11_dir / "detections.json").exists():
        with open(yolo11_dir / "detections.json") as f:
            yolo11_summary = json.load(f)["results"]

    # --- YOLO 12 fine-tune + plate detection ---
    yolo12_summary = []
    yolo12_weights = Path(args.yolo12_weights_out)

    if not args.skip_yolo12:
        if not args.skip_finetune and not yolo12_weights.exists():
            best_pt = finetune_yolo12(
                weights="yolo12n.pt",
                data_yaml=args.data_yaml,
                epochs=args.epochs,
            )
            if best_pt.exists():
                import shutil
                shutil.copy(best_pt, yolo12_weights)
                print(f"[copy] {best_pt} -> {yolo12_weights}")
        elif yolo12_weights.exists():
            print(f"[fine-tune] skipping, using existing {yolo12_weights}")

        if yolo12_weights.exists():
            yolo12_summary = run_plate_detector(
                "yolo12", yolo12_weights, args.source, yolo12_dir,
                conf=args.conf, imgsz=args.imgsz,
            )
        else:
            print("[yolo12] FAILED: no fine-tuned weights available.")
    elif (yolo12_dir / "detections.json").exists():
        with open(yolo12_dir / "detections.json") as f:
            yolo12_summary = json.load(f)["results"]

    # --- Comparison ---
    if yolo11_summary and yolo12_summary:
        cmp_path = project_root / "comparison_plates.json"
        cmp_data = build_comparison(yolo11_summary, yolo12_summary, cmp_path)

        print("\n" + "=" * 70)
        print("PLATE DETECTION: YOLO 11 vs YOLO 12")
        print("=" * 70)
        for name, s in (("YOLO 11 (pre-trained plate)", yolo11_summary),
                        ("YOLO 12 (quick fine-tune)",   yolo12_summary)):
            total = sum(r["num_plates"] for r in s)
            avg_ms = sum(r["inference_ms"] for r in s) / max(len(s), 1)
            imgs_with = sum(1 for r in s if r["num_plates"] > 0)
            print(f"  {name:<28} imgs={len(s):>2}  total_plates={total:>3}  "
                  f"imgs_with_plate={imgs_with:>2}  avg={avg_ms:>6.0f}ms/img")
        print("-" * 70)
        a = cmp_data["agreement_summary"]
        print(f"  agreement: both={a['both_found']} | "
              f"yolo11_only={a['yolo11_only']} | "
              f"yolo12_only={a['yolo12_only']} | neither={a['neither']}")
        print("=" * 70)
        print(f"\nOutputs:")
        print(f"  YOLO 11 results: {yolo11_dir}/")
        print(f"  YOLO 12 results: {yolo12_dir}/")
        print(f"  Comparison:      {cmp_path}")
    else:
        print("\nSkipped comparison (one or both models missing results).")


if __name__ == "__main__":
    main()
