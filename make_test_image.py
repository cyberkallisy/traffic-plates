"""Make a synthetic test traffic image with multiple Indian plates."""
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("test_samples", exist_ok=True)

# Try to load a decent font
def get_font(size):
    for path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except: pass
    return ImageFont.load_default()

PLATES = [
    ("HR26DK1234", (40,  60)),
    ("MH12AB9876", (640, 200)),
    ("DL8CAF5032", (200, 480)),
    ("22BH1234AB", (800, 480)),
]

# Big "traffic scene" canvas
W, H = 1280, 720
img = Image.new("RGB", (W, H), (40, 50, 60))  # dark blue-grey bg
draw = ImageDraw.Draw(img)

# Add some "scene" details
# Sky
draw.rectangle([0, 0, W, 200], fill=(70, 90, 110))
# Road
draw.rectangle([0, 480, W, H], fill=(35, 35, 40))
# Lane lines
for x in range(0, W, 80):
    draw.rectangle([x, 580, x+40, 590], fill=(220, 200, 80))

# Draw car bodies (simple rectangles)
cars = [
    (20, 350, 380, 480, (180, 30, 40)),     # red car top-left
    (600, 350, 940, 480, (40, 80, 160)),    # blue car middle
    (180, 580, 600, 720, (60, 130, 60)),    # green car bottom-left
    (760, 580, 1180, 720, (180, 130, 30)),  # orange car bottom-right
]
for x1, y1, x2, y2, color in cars:
    # body
    draw.rounded_rectangle([x1, y1, x2, y2], radius=18, fill=color)
    # windows
    draw.rounded_rectangle([x1+30, y1+15, x1+150, y1+50], radius=6, fill=(160, 200, 220))
    draw.rounded_rectangle([x2-150, y1+15, x2-30, y1+50], radius=6, fill=(160, 200, 220))
    # wheels
    for wx in [x1+40, x2-60]:
        draw.ellipse([wx-22, y2-15, wx+22, y2+25], fill=(20, 20, 20))
        draw.ellipse([wx-12, y2-5, wx+12, y2+15], fill=(80, 80, 80))

# Draw number plates (white bg, black border, black text)
font = get_font(38)
for text, (x, y) in PLATES:
    # Plate dimensions
    pw, ph = 240, 60
    px, py = x, y
    # Drop shadow
    draw.rectangle([px+3, py+3, px+pw+3, py+ph+3], fill=(0, 0, 0))
    # White plate
    draw.rectangle([px, py, px+pw, py+ph], fill=(245, 245, 235), outline=(20, 20, 20), width=2)
    # Text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text((px + (pw - tw)//2, py + (ph - th)//2 - 4), text, fill=(10, 10, 10), font=font)

out = "test_samples/multi_plate_synth.png"
img.save(out, "PNG")
print(f"Created {out} ({W}x{H}) with {len(PLATES)} plates:")
for t, _ in PLATES:
    print(f"  - {t}")
