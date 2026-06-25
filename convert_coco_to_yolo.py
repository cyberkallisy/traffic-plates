"""
Convert keremberke/license-plate-object-detection HF dataset (COCO format) to
YOLO format, and create a data.yaml for Ultralytics training.

Source: plate_dataset/extracted/_annotations.coco.json + .jpg images
Target: plate_dataset/yolo/<split>/images/*.jpg
        plate_dataset/yolo/<split>/labels/*.txt
        plate_dataset/yolo/data.yaml
"""

import json
import os
import shutil
from pathlib import Path
import random

SRC = Path(r"C:\Users\gsash\Downloads\traffic-plates\plate_dataset\extracted")
DST = Path(r"C:\Users\gsash\Downloads\traffic-plates\plate_dataset\yolo")
DST.mkdir(parents=True, exist_ok=True)

coco_path = SRC / "_annotations.coco.json"
print(f"Reading {coco_path} ...")
with open(coco_path) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
print(f"  total images in COCO: {len(images)}")
print(f"  total annotations:    {len(coco['annotations'])}")
print(f"  categories:           {coco['categories']}")

# COCO bbox = [x, y, w, h] in pixels
# YOLO bbox = [cx, cy, w, h] normalized [0,1]
def coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h

# Build per-image annotation dict
img_to_anns = {}
for ann in coco["annotations"]:
    img_to_anns.setdefault(ann["image_id"], []).append(ann)

# Single-class plate detection (only category 0 = "license_plate")
# Use class id 0 in YOLO output.
PLATE_CLASS_ID = 0
class_names = ["license_plate"]

# Single split: train. (We won't use a val split since we only need a small
# training subset for the fine-tune.) Ultralytics will split off a fraction
# for val automatically via `model.train(data=..., fraction=0.1)` etc.
split_dir = DST / "train"
img_dir = split_dir / "images"
lbl_dir = split_dir / "labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConverting {len(images)} images to YOLO format ...")
n_imgs = 0
n_labels = 0
for img_id, img in images.items():
    src_path = SRC / img["file_name"]
    if not src_path.exists():
        continue
    # Symlink image to save disk space; fall back to copy if symlink fails
    dst_img = img_dir / img["file_name"]
    try:
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.symlink(src_path, dst_img)
    except OSError:
        shutil.copy(src_path, dst_img)

    # Write YOLO label file
    lbl_path = lbl_dir / (Path(img["file_name"]).stem + ".txt")
    with open(lbl_path, "w") as f:
        for ann in img_to_anns.get(img_id, []):
            # COCO category_id may be anything; force to 0 (license_plate)
            cx, cy, w, h = coco_to_yolo(ann["bbox"], img["width"], img["height"])
            # Clamp to [0,1] for safety
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))
            f.write(f"{PLATE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n_labels += 1
    n_imgs += 1

print(f"  wrote {n_imgs} images and {n_labels} labels to {split_dir}")

# Write data.yaml
data_yaml = DST / "data.yaml"
data_yaml.write_text(
    f"""# Plate detection dataset (keremberke/license-plate-object-detection,
# converted from COCO to YOLO format).

path: {DST.as_posix()}
train: train/images
val: train/images        # use a fraction of train as val (ultralytics auto-split)

names:
  0: license_plate
"""
)
print(f"\nWrote {data_yaml}")
print("\nDone.")
