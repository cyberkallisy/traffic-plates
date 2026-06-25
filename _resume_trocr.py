"""Complete the partial TrOCR model download (siblings left it 35%/62% done)."""
from huggingface_hub import snapshot_download
import os, time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # use plain urllib, faster to resume

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] starting snapshot_download...", flush=True)
p = snapshot_download(
    "microsoft/trocr-base-printed",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.bin"],
    max_workers=2,
)
print(f"[{time.time()-t0:.1f}s] DONE: {p}", flush=True)

import os
for root, _, files in os.walk(p):
    for f in files:
        full = os.path.join(root, f)
        print(f"  {os.path.getsize(full):>12}  {full}", flush=True)