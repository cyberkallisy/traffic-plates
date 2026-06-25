"""
Qwen2.5-VL Traffic Plate Detector + OCR
========================================
Uses Qwen2.5-VL (vision-language model) to detect number plates
and read text from traffic photos.

Usage:
    python run_qwen25vl.py
"""

import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_DIR = r"C:\Users\gsash\Downloads\test\New folder"
OUTPUT_DIR = Path(r"C:\Users\gsash\Downloads\traffic-plates\qwen 2.5 vl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID} ...")
t0 = time.time()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

print(f"Model loaded in {time.time() - t0:.1f}s")
print(f"Output dir: {OUTPUT_DIR}")
print()

# ---------------------------------------------------------------------------
# Prompt for plate detection + OCR
# ---------------------------------------------------------------------------
PLATE_PROMPT = (
    "Detect all vehicle number plates in this image. "
    "For each plate, provide: "
    "1) The bounding box coordinates (normalized 0-1000 for x1, y1, x2, y2), "
    "2) The text/characters visible on the plate. "
    "Return the result as a JSON array. "
    "If no plates are detected, return an empty array []. "
    "For each plate object, include: "
    "{'bbox': [x1, y1, x2, y2], 'text': 'plate text here', 'confidence': 0.0-1.0}"
)


def process_image(image_path: str) -> dict:
    """Run Qwen2.5-VL on a single image."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PLATE_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
    )

    generated_ids = [
        output_ids[i][len(input_ids):]
        for i, input_ids in enumerate(inputs["input_ids"])
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text


def extract_json_from_response(response: str) -> list:
    """Try to extract a JSON array from the model's response."""
    import re

    # Try to find JSON in code blocks
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON array directly
    json_match = re.search(r"\[[\s\S]*\]", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find a single JSON object
    obj_match = re.search(r"\{[\s\S]*\}", response)
    if obj_match:
        try:
            return [json.loads(obj_match.group(0))]
        except json.JSONDecodeError:
            pass

    return []


def draw_results(image_path: str, plates: list, output_path: str):
    """Draw plate bounding boxes on the image."""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]

    for plate in plates:
        bbox = plate.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v * w / 1000) if i % 2 == 0 else int(v * h / 1000)
                          for i, v in enumerate(bbox)]

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        text = plate.get("text", "")
        conf = plate.get("confidence", 0.0)

        # Green for detected plates
        color = (0, 200, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        label = f"{text} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    ])

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images to process\n")

    all_results = {}

    for idx, fname in enumerate(image_files, 1):
        image_path = os.path.join(INPUT_DIR, fname)
        print(f"[{idx}/{len(image_files)}] Processing: {fname}")

        t_start = time.time()
        raw_response = process_image(image_path)
        elapsed = round(time.time() - t_start, 2)

        plates = extract_json_from_response(raw_response)

        # Save annotated image
        base_name = Path(fname).stem
        annotated_path = OUTPUT_DIR / f"{base_name}_annotated.jpg"
        draw_results(image_path, plates, annotated_path)

        # Save result JSON
        result = {
            "filename": fname,
            "image_size": [w, h] if (w := cv2.imread(image_path)) else None,
            "num_plates_detected": len(plates),
            "plates": plates,
            "raw_response": raw_response,
            "elapsed_seconds": elapsed,
        }

        result_path = OUTPUT_DIR / f"{base_name}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"  Plates: {len(plates)} ({elapsed}s)")
        for p in plates:
            bbox = p.get("bbox", [])
            text = p.get("text", "N/A")
            conf = p.get("confidence", 0)
            print(f"    [{conf:.2f}] '{text}' bbox={bbox}")

        all_results[fname] = result
        print()

    # Save combined results
    combined_path = OUTPUT_DIR / "all_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_plates = sum(len(r["plates"]) for r in all_results.values())
    print(f"\nDone! Processed {len(image_files)} images, found {total_plates} plate(s) total.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
