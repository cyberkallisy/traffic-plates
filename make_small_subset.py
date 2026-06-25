"""Create a small training subset (~300 images) for fast CPU fine-tuning."""
import random
import os
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\gsash\Downloads\traffic-plates\plate_dataset\yolo\train")
DST = Path(r"C:\Users\gsash\Downloads\traffic-plates\plate_dataset\yolo_small\train")
DST.mkdir(parents=True, exist_ok=True)
(DST / "images").mkdir(exist_ok=True)
(DST / "labels").mkdir(exist_ok=True)

N = 300
random.seed(42)

# Collect images that have non-empty labels (skip negatives for now)
candidates = []
for lbl in SRC.glob("labels/*.txt"):
    if lbl.stat().st_size > 0:
        candidates.append(lbl.stem)

print(f"Images with plate labels: {len(candidates)}")
picks = random.sample(candidates, min(N, len(candidates)))
print(f"Picking {len(picks)} images for fine-tune subset")

# Copy
for stem in picks:
    src_img = SRC / "images" / f"{stem}.jpg"
    src_lbl = SRC / "labels" / f"{stem}.txt"
    dst_img = DST / "images" / src_img.name
    dst_lbl = DST / "labels" / src_lbl.name
    if dst_img.exists() or dst_img.is_symlink():
        dst_img.unlink()
    if dst_lbl.exists() or dst_lbl.is_symlink():
        dst_lbl.unlink()
    try:
        os.symlink(src_img, dst_img)
        os.symlink(src_lbl, dst_lbl)
    except OSError:
        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

# data.yaml for small subset
(DST.parent / "data.yaml").write_text(
    f"""path: {DST.parent.as_posix()}
train: train/images
val: train/images
names:
  0: license_plate
"""
)
print(f"Done. Subset at: {DST}")
