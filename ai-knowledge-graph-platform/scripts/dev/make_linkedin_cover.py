"""Generate the 1584x396 LinkedIn profile cover banner."""

import math
import random

from PIL import Image, ImageDraw, ImageFont

W, H = 1584, 396
BG = (13, 17, 28)
NODE = (56, 132, 222)
NODE_DIM = (38, 70, 120)
EDGE = (32, 52, 84)
TITLE_COL = (235, 240, 248)
SUB_COL = (148, 163, 184)
ACCENT = (96, 200, 255)

FONT_DIR = "C:/Windows/Fonts/"
title_f = ImageFont.truetype(FONT_DIR + "segoeuib.ttf", 52)
sub_f = ImageFont.truetype(FONT_DIR + "segoeui.ttf", 26)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# graph motif across the full width, denser on the right
random.seed(11)
nodes = []
for _ in range(40):
    x = random.uniform(0, W)
    y = random.uniform(20, H - 20)
    r = random.uniform(3, 9)
    nodes.append((x, y, r))

for i, (x1, y1, _) in enumerate(nodes):
    for x2, y2, _ in nodes[i + 1:]:
        if math.hypot(x2 - x1, y2 - y1) < 170:
            d.line([(x1, y1), (x2, y2)], fill=EDGE, width=2)

for x, y, r in nodes:
    col = NODE if r > 6 else NODE_DIM
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)

# dark overlay strip so text stays readable over the motif
overlay = Image.new("L", (W, H), 0)
od = ImageDraw.Draw(overlay)
od.rectangle([0, 0, W, H], fill=140)
img.paste(Image.new("RGB", (W, H), BG), (0, 0), overlay)

# IMPORTANT: LinkedIn overlaps the profile photo on the LEFT ~430px on desktop
# (and bottom-left on mobile), so keep text in the right two-thirds, upper half.
x0 = 500
d.text((x0, 100), "Production LLM & Agentic Systems", font=title_f, fill=TITLE_COL)
d.line([(x0 + 4, 175), (x0 + 124, 175)], fill=ACCENT, width=4)
d.text((x0, 200), "GraphRAG  ·  Knowledge Graphs  ·  Neo4j  ·  RAGAS-evaluated  ·  A2A  ·  MCP", font=sub_f, fill=SUB_COL)

out = "docs/assets/linkedin-cover.png"
img.save(out)
print(f"saved {out} ({W}x{H})")
