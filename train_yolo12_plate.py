"""Standalone YOLO 12 plate fine-tune. Runs in background."""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo12n.pt")
    ap.add_argument("--data", default=r"C:\Users\gsash\Downloads\traffic-plates\plate_dataset\yolo_small\data.yaml")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default=r"C:\Users\gsash\Downloads\traffic-plates\runs\yolo12_plate")
    ap.add_argument("--out", default=r"C:\Users\gsash\Downloads\traffic-plates\yolo12_plate.pt")
    args = ap.parse_args()

    print(f"[yolo12 finetune] weights={args.weights} data={args.data}")
    print(f"[yolo12 finetune] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        project=args.project,
        name="plate_finetune",
        patience=0,
        save=True,
        save_period=-1,
        verbose=False,
        cache=False,
        # Disable all augmentation for the quick pass (pure transfer learning)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        plots=False,
        # No validation images during training (saves time on tiny subset)
        val=False,
    )

    import shutil
    src_best = Path(args.project) / "plate_finetune" / "weights" / "best.pt"
    src_last = Path(args.project) / "plate_finetune" / "weights" / "last.pt"
    final = Path(args.out)
    if src_best.exists():
        shutil.copy(src_best, final)
        print(f"[yolo12 finetune] copied best.pt -> {final}")
    elif src_last.exists():
        shutil.copy(src_last, final)
        print(f"[yolo12 finetune] copied last.pt -> {final}")
    else:
        print(f"[yolo12 finetune] ERROR: no weights found in {args.project}")


if __name__ == "__main__":
    main()
