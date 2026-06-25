# traffic-plates

Minimal ANPR (Automatic Number Plate Recognition) web app. Upload an image or video, get back detected plates with bounding boxes, track IDs, and OCR text.

## Pipeline

- **Detection**: YOLO11 (`yolo11_plate.pt`) — fine-tuned on Indian number plates (single class: `License_Plate`)
- **Tracking**: ByteTrack (ultralytics built-in, motion-only — no ReID model)
- **OCR**: Awiros ANPR (PP-OCRv5 SVTR_HGNet / CTC)
- **Web**: Flask on port 8766

## Run

```bash
# Install deps (Python 3.11, CUDA optional)
pip install -r requirements.txt

# Drop the two model artifacts into the repo root before first run:
#   yolo11_plate.pt              (YOLO11 plate detector)
#   awiros_anpr/model.safetensors (Awiros ANPR OCR)

python run.py             # default http://127.0.0.1:8766
python run.py --port 9000
python run.py --no-warmup # skip model load at startup
```

## Layout

```
run.py                       # entry point — Flask app + warmup
api/
  __init__.py                # Flask app factory (create_app)
  routes_video.py            # /api/detect_video, /api/track_details, /video-results/*
  routes_image.py            # /api/detect, /api/predict
core/
  engine.py                  # ANPR engine (YOLO + Awiros OCR)
anpr_video_awiros.py         # video pipeline + ByteTracker wrapper
detect_yolo11_awiros_ocr.py  # YOLO11 loader + AwirosANPR class + draw helper
templates/index.html         # upload UI
static/                      # JS + CSS
requirements.txt
```

The model weights (`yolo11_plate.pt`, `awiros_anpr/`), runtime uploads, and per-video result folders are listed in `.gitignore` and are not tracked in git.

## Notes

- `run.py` listens on `127.0.0.1:8766` by default. Use `--port` to change it.
- The video pipeline uses **ByteTrack** via ultralytics' built-in tracker (motion-only Kalman + IoU association). It exposes the same `(bbox, yconf) -> (tid, bbox, yconf)` interface the rest of the pipeline already speaks, so swapping to BoT-SORT / DeepOC-SORT is a one-line change inside `anpr_video_awiros.ByteTracker`.
- Per-character-position voting across frames collapses OCR noise into a final plate text + confidence per track.
