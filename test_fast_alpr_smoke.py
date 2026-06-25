"""Smoke test: fast-alpr on one image to confirm Windows compatibility."""
import sys
import time
from pathlib import Path

t0 = time.time()
print("[1/3] Importing fast_alpr...", flush=True)
from fast_alpr import ALPR  # type: ignore
print(f"   ok ({time.time()-t0:.1f}s)", flush=True)

print("[2/3] Initialising ALPR (loads YOLO + OCR models)...", flush=True)
t1 = time.time()
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
    detector_conf_thresh=0.3,
    ocr_device="cpu",
)
print(f"   ok ({time.time()-t1:.1f}s)", flush=True)

img = Path("C:/Users/gsash/Downloads/test/New folder/1.png")
if not img.exists():
    sys.exit(f"Image not found: {img}")

print(f"[3/3] Detecting plates in {img.name}...", flush=True)
t2 = time.time()
results = alpr.predict(str(img))
elapsed = time.time() - t2
print(f"   ok ({elapsed:.2f}s)", flush=True)

print(f"\n=== RESULTS ({len(results)} plates) ===")
for i, r in enumerate(results):
    # r is a dataclass-like: detection, recognition, ocr_time
    det = getattr(r, "detection", None)
    rec = getattr(r, "recognition", None)
    if det is not None:
        print(f"  [{i}] bbox={getattr(det,'bbox',None)} score={getattr(det,'score',None):.3f}")
    if rec is not None:
        print(f"      text={getattr(rec,'text',None)!r} conf={getattr(rec,'score',None):.3f}")
    print(f"      ocr_time={getattr(r,'ocr_time',None)}")

print(f"\nTotal: {time.time()-t0:.1f}s")