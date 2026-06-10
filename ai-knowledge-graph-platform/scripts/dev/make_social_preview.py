"""Generate the 1280x640 social-preview banner (GitHub / LinkedIn Featured thumbnail)."""

import math
import random

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG = (13, 17, 28)
NODE = (56, 132, 222)
NODE_DIM = (38, 70, 120)
EDGE = (32, 52, 84)
TITLE_COL = (235, 240, 248)
SUB_COL = (148, 163, 184)
ACCENT = (96, 200, 255)
CHIP_BG = (24, 34, 52)
CHIP_BORDER = (45, 70, 105)

FONT_DIR = "C:/Windows/Fonts/"
title_f = ImageFont.truetype(FONT_DIR + "segoeuib.ttf", 64)
sub_f = ImageFont.truetype(FONT_DIR + "segoeui.ttf", 30)
chip_val_f = ImageFont.truetype(FONT_DIR + "segoeuib.ttf", 34)
chip_lbl_f = ImageFont.truetype(FONT_DIR + "segoeui.ttf", 22)
stack_f = ImageFont.truetype(FONT_DIR + "segoeui.ttf", 24)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# --- graph motif on the right side ---
random.seed(7)
nodes = []
for _ in range(26):
    x = random.uniform(760, 1250)
    y = random.uniform(40, 600)
    r = random.uniform(4, 11)
    nodes.append((x, y, r))

for i, (x1, y1, _) in enumerate(nodes):
    for x2, y2, _ in nodes[i + 1:]:
        if math.hypot(x2 - x1, y2 - y1) < 150:
            d.line([(x1, y1), (x2, y2)], fill=EDGE, width=2)

for x, y, r in nodes:
    col = NODE if r > 7 else NODE_DIM
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)

# fade the motif under the text area
fade = Image.new("L", (W, H), 0)
fd = ImageDraw.Draw(fade)
for i in range(700):
    fd.line([(i, 0), (i, H)], fill=max(0, 230 - i // 2))
img.paste(Image.new("RGB", (W, H), BG), (0, 0), fade)

# --- text block ---
x0 = 70
d.text((x0, 130), "AI Knowledge Graph", font=title_f, fill=TITLE_COL)
d.text((x0, 210), "Platform", font=title_f, fill=TITLE_COL)
d.line([(x0 + 4, 305), (x0 + 124, 305)], fill=ACCENT, width=5)
d.text((x0, 330), "Production GraphRAG for regulatory compliance", font=sub_f, fill=SUB_COL)

# --- metric chips ---
chips = [("0.940", "faithfulness"), ("2.2s", "p95 latency"), ("364", "tests passing")]
cx = x0
cy = 410
for val, lbl in chips:
    vw = d.textlength(val, font=chip_val_f)
    lw = d.textlength(lbl, font=chip_lbl_f)
    w = max(vw, lw) + 48
    d.rounded_rectangle([cx, cy, cx + w, cy + 104], radius=14, fill=CHIP_BG, outline=CHIP_BORDER, width=2)
    d.text((cx + 24, cy + 14), val, font=chip_val_f, fill=ACCENT)
    d.text((cx + 24, cy + 62), lbl, font=chip_lbl_f, fill=SUB_COL)
    cx += w + 22

d.text((x0, 560), "Python  ·  Neo4j  ·  FastAPI  ·  Hybrid retrieval + GNN  ·  RAGAS-evaluated", font=stack_f, fill=SUB_COL)

out = "docs/assets/social-preview.png"
img.save(out)
print(f"saved {out} ({W}x{H})")
