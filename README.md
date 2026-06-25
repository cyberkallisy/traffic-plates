# traffic-plates

ANPR (Automatic Number Plate Recognition) web app. Upload a video or image, get back detected plates with bounding boxes and OCR text.

## Pipeline

- **Detection**: YOLO11 (`yolo11_plate.pt`) — fine-tuned on Indian number plates
- **Tracking (video)**: DeepSORT
- **OCR**: Awiros ANPR
- **Web**: Flask (port 8766)

## Run

```bash
# Install deps (Python 3.11, CUDA optional)
pip install -r requirements.txt

# Drop model weights into the repo root before first run:
#   yolo11_plate.pt             (YOLO11 plate detector)
#   awiros_anpr/model.safetensors (Awiros ANPR)

python run.py            # default http://127.0.0.1:8766
python run.py --port 9000
python run.py --no-warmup   # skip model load at startup
```

## Layout

```
run.py                       # entry point
api/
  __init__.py                # Flask app factory (create_app)
  routes_video.py            # /api/video, /api/video_status, /api/video_results
  routes_image.py            # /api/image
core/
  engine.py                  # ANPR engine (YOLO + OCR + tracker)
templates/index.html         # upload UI
static/                      # JS + CSS
requirements.txt
```

Model weights, sample images, and runtime artifacts (uploads, results, logs) are listed in `.gitignore` and are not tracked in git.

## Notes

- Server logs to `_server.log` (also visible in the embedded terminal during dev).
- `app_backup.py` is the pre-restructured single-file Flask app, kept for reference.
- The `anpr_video_awiros.py` and `detect_yolo11_awiros_ocr.py` scripts are the standalone CLI equivalents of the engine — useful for batch processing outside the web UI.