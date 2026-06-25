"""Run only the Qwen2.5-VL part (other steps already done)."""
import os, json, time, subprocess, sys
from pathlib import Path

BASE = Path(r"C:\Users\gsash\Downloads\test\New folder")
OUT = BASE / "comparison_4engines"
CROP_DIR = OUT / "crops" / "yolo11_crops"
QWEN_OUT = OUT / "qwen_results"
QWEN_OUT.mkdir(exist_ok=True)

PYTHON = r"C:\Users\gsash\Downloads\Facial-recognition\venv\Scripts\python.exe"
PROMPT = "Read the license plate text exactly as it appears in the image. Return ONLY the plate text, no other commentary. If unreadable, return UNREADABLE."

# Use float32 + no device_map (simpler on CPU)
qwen_script = OUT / "_qwen_run_v2.py"
qwen_script.write_text(
    "import json, time, os\n"
    "from pathlib import Path\n"
    "from PIL import Image\n"
    "os.environ['HF_HUB_OFFLINE'] = '0'\n"
    "\n"
    "print('Loading Qwen2.5-VL-7B-Instruct (CPU, fp32)...', flush=True)\n"
    "t0 = time.time()\n"
    "import torch\n"
    "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor\n"
    "\n"
    "processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')\n"
    "model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n"
    "    'Qwen/Qwen2.5-VL-7B-Instruct',\n"
    "    torch_dtype=torch.float32,\n"
    "    low_cpu_mem_usage=True,\n"
    ").eval()\n"
    "print(f'  loaded in {time.time()-t0:.1f}s', flush=True)\n"
    "\n"
    f"CROP_DIR = Path(r'{CROP_DIR}')\n"
    f"OUT_FILE = Path(r'{QWEN_OUT}/summary.json')\n"
    "\n"
    f"PROMPT = {PROMPT!r}\n"
    "\n"
    "crops = sorted(CROP_DIR.glob('*.png'))\n"
    "print(f'Processing {len(crops)} crops...', flush=True)\n"
    "results = []\n"
    "for i, c in enumerate(crops, 1):\n"
    "    img = Image.open(c).convert('RGB')\n"
    "    t = time.time()\n"
    "    try:\n"
    "        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]\n"
    "        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
    "        inputs = processor(text=[text_input], images=[img], return_tensors='pt')\n"
    "        with torch.no_grad():\n"
    "            ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)\n"
    "        out_text = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()\n"
    "    except Exception as e:\n"
    "        out_text = f'ERROR: {type(e).__name__}: {str(e)[:100]}'\n"
    "    el = time.time() - t\n"
    "    stem = c.stem.rsplit('_p', 1)[0]\n"
    "    idx = c.stem.rsplit('_p', 1)[1]\n"
    "    results.append({\n"
    "        'crop_file': c.name,\n"
    "        'source_image_stem': stem,\n"
    "        'plate_idx': int(idx),\n"
    "        'qwen_text': out_text,\n"
    "        'elapsed_sec': round(el, 3),\n"
    "    })\n"
    "    if i % 2 == 0 or i == len(crops):\n"
    "        print(f'  {i}/{len(crops)}: {c.name} -> {out_text!r} ({el:.2f}s)', flush=True)\n"
    "    # Save partial results every 5 crops\n"
    "    if i % 5 == 0:\n"
    "        OUT_FILE.write_text(json.dumps(results, indent=2))\n"
    "\n"
    "OUT_FILE.write_text(json.dumps(results, indent=2))\n"
    "print(f'Wrote {OUT_FILE}', flush=True)\n"
, encoding="utf-8")

print("Starting Qwen2.5-VL-7B (CPU fp32)...", flush=True)
print("This is SLOW on CPU. Estimated 30-90s/crop × 31 crops = 15-45 min", flush=True)
print("Will save partial results every 5 crops so we can recover if killed", flush=True)

# Run in background with logging
log = open(OUT / "qwen_run.log", "wb")
proc = subprocess.Popen(
    [PYTHON, "-u", str(qwen_script)],
    cwd=str(OUT),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
)

import threading
def stream():
    for line in iter(proc.stdout.readline, b''):
        log.write(line); log.flush()
        try:
            print(line.decode('utf-8', errors='replace').rstrip())
        except Exception:
            pass

threading.Thread(target=stream, daemon=True).start()

# Wait up to 45 min
for i in range(540):
    time.sleep(5)
    if proc.poll() is not None:
        break
    if i % 12 == 11:
        print(f"  [Qwen still running, {(i+1)*5}s elapsed]", flush=True)

proc.wait()
log.close()
print(f"\n=== Qwen process exited with code {proc.returncode} ===")