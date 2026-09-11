"""Render an implementation-led Energy Asset Intelligence POC movie.

The video presents the repository's implemented data and application paths.
It intentionally distinguishes the current deterministic demo service from
future query-backed RDF-store integration.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "energy_demo_movie_build"
OUT = ROOT / "energy_asset_intelligence_implementation_demo.mp4"

BG = (6, 17, 29)
PANEL = (11, 33, 49)
WHITE = (240, 247, 251)
MUTED = (153, 180, 199)
CYAN = (96, 222, 255)
GOLD = (255, 202, 91)
GREEN = (78, 211, 151)
RED = (255, 111, 118)


@dataclass(frozen=True)
class Scene:
    title: str
    minimum_seconds: int
    voiceover: str


SCENES = [
    Scene(
        "A Reproducible Energy Knowledge Graph POC",
        20,
        "This walkthrough follows the implementation of an Energy Asset and "
        "Maintenance Intelligence proof of concept. It uses synthetic wind-farm "
        "data, so every step can be run locally. The goal is to show how source "
        "contracts, semantic modelling, validation, and an evidence response fit "
        "together in working code.",
    ),
    Scene(
        "Create the SAP-shaped Source Contract",
        20,
        "The workflow starts with a reproducible SQLite export. The create script "
        "makes ten turbine assets and three work orders. Each work order has an "
        "identifier, an asset identifier, and a status. This is a safe stand-in "
        "for an operational SAP extract and gives the mapping layer a concrete "
        "source contract to validate.",
    ),
    Scene(
        "Parse and Validate the R2RML Mapping",
        24,
        "The energy assets mapping is written in R2RML Turtle. The ingestion "
        "command parses that mapping, reads the SQLite rows, and validates the "
        "contract for the energy-demo tenant. The presentation uses validate-only, "
        "which proves the mapping and source fit without writing data. The existing "
        "graph writer is called only when that flag is removed.",
    ),
    Scene(
        "Build the RDF Evidence Graph",
        28,
        "The demonstration service then builds an in-memory RDFLib graph from its "
        "synthetic fixtures. It creates turbines and gearbox components, "
        "Snowflake-shaped telemetry, SAP-shaped work orders, and SharePoint-shaped "
        "manufacturer bulletin revisions. Each source category has an explicit "
        "provenance reference. The graph can be exported as Turtle for inspection "
        "or optional loading into an RDF store.",
    ),
    Scene(
        "Return an Evidence-shaped Assessment",
        28,
        "The maintenance-review path brings the relevant records together. WT-01 "
        "has a gearbox temperature of 96 degrees Celsius. Bulletin MFG-GBX-17-R2 "
        "sets an 85-degree review threshold, and work order WO-9001 is open. The "
        "response returns the advisory assessment together with the source IDs, "
        "fields, timestamps, access scope, mapping version, and query version.",
    ),
    Scene(
        "Handle Revisions and Missing Evidence",
        26,
        "The model also records that bulletin R2 supersedes R1 and lowers the "
        "threshold from 90 to 85 degrees. Supplying a May 2026 date selects R1 "
        "as the authoritative bulletin. Separate fixed questions identify assets "
        "with insufficient evidence. They return no maintenance conclusion when "
        "the required synthetic observations or work-order state are absent.",
    ),
    Scene(
        "Validate Data and Enforce Tenant Boundaries",
        26,
        "SHACL validates a deliberately incomplete observation and reports that it "
        "does not conform. The FastAPI router exposes a small read-only surface: "
        "questions, answers, RDF export, and validation. It requires read scope "
        "and checks the energy-demo tenant. Other tenants receive no evidence. "
        "This keeps the POC's retrieval surface deliberate and reviewable.",
    ),
    Scene(
        "Run the Scenario and Define the Next Milestone",
        28,
        "Five labelled cases verify the current maintenance result, work orders, "
        "revision selection, historical bulletin selection, and insufficient "
        "evidence. The next engineering milestone is to execute the version-"
        "controlled SPARQL query against this graph or a verified RDF store, use "
        "the results to produce the responses, and connect approved enterprise "
        "sources end to end.",
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    face = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(face, size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 20,
         color: tuple[int, int, int] = WHITE, bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color, anchor=anchor)


def wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, width: int,
            size: int = 20, color: tuple[int, int, int] = WHITE, bold: bool = False) -> None:
    draw.multiline_text(xy, textwrap.fill(value, width), font=font(size, bold), fill=color, spacing=8)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
          outline: tuple[int, int, int] = (45, 97, 119), fill: tuple[int, int, int] = PANEL) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline, width=2)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          color: tuple[int, int, int] = CYAN, width: int = 4) -> None:
    draw.line((*start, *end), fill=color, width=width)
    x, y = end
    direction = 1 if x >= start[0] else -1
    draw.polygon([(x, y), (x - 13 * direction, y - 7), (x - 13 * direction, y + 7)], fill=color)


def node(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, sublabel: str,
         color: tuple[int, int, int] = CYAN) -> None:
    x, y = xy
    draw.ellipse((x - 72, y - 52, x + 72, y + 52), fill=(8, 35, 53), outline=color, width=3)
    text(draw, (x, y - 10), label, 15, color, True, "mm")
    text(draw, (x, y + 17), sublabel, 11, MUTED, False, "mm")


def command_box(draw: ImageDraw.ImageDraw, command: str, lines: list[str]) -> None:
    panel(draw, (75, 448, 1205, 644), outline=(46, 104, 130), fill=(5, 22, 35))
    text(draw, (105, 478), "POWERSHELL", 13, GOLD, True)
    text(draw, (105, 515), f"> {command}", 17, CYAN)
    for index, line in enumerate(lines):
        text(draw, (105, 560 + index * 28), line, 15, GREEN if "validated" in line or "passed" in line or "Created" in line else WHITE)


def base(index: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    for x in range(0, W, 64):
        draw.line((x, 0, x, H), fill=(11, 33, 51), width=1)
    for y in range(0, H, 48):
        draw.line((0, y, W, y), fill=(11, 33, 51), width=1)
    text(draw, (54, 34), "ENERGY ASSET INTELLIGENCE / IMPLEMENTATION WALKTHROUGH", 15, CYAN, True)
    text(draw, (1225, 36), "SYNTHETIC DATA · ADVISORY POC", 13, GOLD, True, "ra")
    text(draw, (54, 76), f"{index + 1:02d}", 16, GOLD, True)
    text(draw, (90, 70), SCENES[index].title, 30, WHITE, True)
    for item in range(len(SCENES)):
        x = 54 + item * 143
        draw.line((x, 118, x + 125, 118), fill=GOLD if item <= index else (37, 70, 87), width=4)
    return image, draw


def scene_0(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (70, 160, 1210, 400), outline=GOLD)
    wrapped(draw, (115, 205), "How source contracts, RDF semantics, validation, and evidence responses fit together.", 60, 33, WHITE, True)
    nodes = [
        ((185, 540), "SOURCE", "SQLite export"),
        ((455, 540), "R2RML", "mapping validation"),
        ((725, 540), "RDF", "evidence graph"),
        ((995, 540), "API", "tenant-scoped answer"),
    ]
    for position, label, sublabel in nodes:
        node(draw, position, label, sublabel, GOLD if label == "RDF" else CYAN)
    for start, end in zip([item[0] for item in nodes], [item[0] for item in nodes][1:]):
        arrow(draw, (start[0] + 74, start[1]), (end[0] - 74, end[1]), GOLD)


def scene_1(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (85, 160, 565, 402), outline=CYAN)
    text(draw, (120, 198), "sap_assets", 20, GOLD, True)
    for index, asset in enumerate(("WT-01  |  Wind turbine WT-01", "WT-02  |  Wind turbine WT-02", "…", "WT-10  |  Wind turbine WT-10")):
        text(draw, (120, 250 + index * 34), asset, 17, WHITE)
    panel(draw, (695, 160, 1195, 402), outline=GREEN)
    text(draw, (730, 198), "sap_work_orders", 20, GOLD, True)
    for index, order in enumerate(("WO-9001  |  WT-01  |  open", "WO-9002  |  WT-02  |  open", "WO-9003  |  WT-03  |  closed")):
        text(draw, (730, 250 + index * 45), order, 17, WHITE)
    command_box(draw, "python scripts/create_energy_demo_sqlite.py", ["Created synthetic SAP-shaped SQLite export:", "artifacts/energy-demo-sap.sqlite"])


def scene_2(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (70, 155, 605, 410), outline=CYAN)
    text(draw, (105, 190), "ontology/mappings/energy-assets.r2rml.ttl", 16, GOLD, True)
    mapping_lines = [
        "rr:logicalTable [ rr:tableName \"sap_assets\" ] ;",
        "rr:subjectMap [ rr:template \".../asset/{asset_id}\" ] ;",
        "rr:predicateObjectMap [",
        "  rr:predicate energy:assetId ;",
        "  rr:objectMap [ rr:column \"asset_id\" ]",
        "] .",
    ]
    for index, line in enumerate(mapping_lines):
        text(draw, (105, 235 + index * 26), line, 14, WHITE if index != 1 else CYAN)
    panel(draw, (690, 155, 1210, 410), outline=GREEN)
    text(draw, (730, 190), "Validation boundary", 18, GOLD, True)
    for index, line in enumerate(("1. Parse R2RML Turtle", "2. Read synthetic source rows", "3. Validate entity contracts", "4. Exit before graph write")):
        text(draw, (730, 245 + index * 36), line, 18, WHITE, index == 3)
    command_box(draw, "python scripts/ingest_r2rml.py … --validate-only", ["ingest_r2rml.validated  tenant=energy-demo", "entity_rows=13  relation_rows=0"])


def scene_3(draw: ImageDraw.ImageDraw) -> None:
    items = [
        ((165, 315), "WT-01", "asset"),
        ((420, 220), "GEARBOX", "component"),
        ((420, 430), "96 °C", "telemetry"),
        ((700, 220), "WO-9001", "open work order"),
        ((970, 315), "R2", "85 °C bulletin"),
    ]
    for position, label, sublabel in items:
        node(draw, position, label, sublabel, GOLD if label in ("96 °C", "R2") else CYAN)
    for start, end in (((239, 315), (346, 220)), ((239, 315), (346, 430)), ((494, 220), (626, 220)), ((774, 220), (896, 315))):
        arrow(draw, start, end, GOLD)
    text(draw, (640, 543), "RDFLib graph with explicit types, links, dates, and provenance", 21, GREEN, True, "mm")
    command_box(draw, "python scripts/run_energy_demo.py --export-turtle artifacts/energy-demo.ttl", ["Wrote RDF Turtle: artifacts/energy-demo.ttl"])


def scene_4(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (72, 155, 640, 425), outline=GREEN)
    text(draw, (108, 190), "maintenance_review", 17, GOLD, True)
    wrapped(draw, (108, 235), "Advisory review is required for WT-01. Its gearbox temperature is 96°C, above the 85°C threshold in MFG-GBX-17-R2; WO-9001 is open.", 47, 20, WHITE, True)
    panel(draw, (700, 155, 1208, 425), outline=CYAN)
    text(draw, (735, 190), "Evidence bundle", 17, GOLD, True)
    for index, line in enumerate(("SAP-WO-9001 · work_orders.status · open", "SNOW-OBS-WT-01 · temperature_c=96", "MFG-GBX-17-R2 · threshold: 85 C", "tenant: energy-demo")):
        text(draw, (735, 240 + index * 42), line, 16, WHITE)
    command_box(draw, "python scripts/run_energy_demo.py", ["status: advisory", "mapping_version: energy-r2rml/1.0.0", "query_version: energy-demo/v1"])


def scene_5(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (80, 160, 570, 420), outline=GOLD)
    text(draw, (115, 200), "Manufacturer bulletin revisions", 18, GOLD, True)
    text(draw, (135, 280), "R1", 28, WHITE, True)
    text(draw, (135, 325), "90 °C · valid from 2026-01-01", 17, MUTED)
    arrow(draw, (310, 305), (460, 305), GOLD)
    text(draw, (480, 280), "R2", 28, GREEN, True)
    text(draw, (480, 325), "85 °C · valid from 2026-06-01", 17, MUTED)
    panel(draw, (700, 160, 1200, 420), outline=RED)
    text(draw, (735, 200), "Insufficient evidence", 18, GOLD, True)
    wrapped(draw, (735, 260), "WT-04 through WT-10: missing current gearbox observation or open-work-order state.", 39, 21, WHITE)
    text(draw, (735, 365), "No maintenance conclusion", 20, RED, True)
    command_box(draw, "python scripts/run_energy_demo.py --as-of 2026-05-01T00:00:00Z", ["historical_state  authoritative_bulletin=MFG-GBX-17-R1", "review threshold: 90 °C"])


def scene_6(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (75, 160, 590, 418), outline=RED)
    text(draw, (110, 197), "SHACL validation", 18, GOLD, True)
    text(draw, (110, 250), "Candidate observation", 16, MUTED)
    for index, line in enumerate(("type: Observation", "observedAsset: WT-10", "observedAt: 2026-08-28T08:00:00Z", "unit: missing", "value: missing")):
        text(draw, (110, 285 + index * 25), line, 16, WHITE if index < 3 else RED)
    text(draw, (110, 390), "conforms: false", 22, RED, True)
    panel(draw, (690, 160, 1210, 418), outline=CYAN)
    text(draw, (725, 197), "FastAPI boundary", 18, GOLD, True)
    for index, line in enumerate(("GET /energy-demo/questions", "GET /energy-demo/answer/{question_id}", "GET /energy-demo/rdf", "GET /energy-demo/validation")):
        text(draw, (725, 250 + index * 32), line, 17, WHITE)
    text(draw, (725, 385), "read scope + energy-demo tenant", 18, GREEN, True)
    command_box(draw, "python scripts/run_energy_demo.py", ["Invalid-batch validation: { conforms: false, … }"])


def scene_7(draw: ImageDraw.ImageDraw) -> None:
    panel(draw, (80, 160, 555, 420), outline=GREEN)
    text(draw, (115, 198), "Labelled evaluation cases", 18, GOLD, True)
    for index, item in enumerate(("maintenance_review", "open_work_orders", "revision_change", "historical_state", "insufficient_evidence")):
        text(draw, (120, 248 + index * 30), f"✓  {item}", 17, GREEN)
    panel(draw, (690, 160, 1205, 420), outline=CYAN)
    text(draw, (725, 198), "Next query-backed milestone", 18, GOLD, True)
    for index, item in enumerate(("Execute version-controlled SPARQL", "Use results to build evidence responses", "Verify GraphDB repository path", "Connect approved enterprise sources")):
        text(draw, (725, 248 + index * 35), item, 17, WHITE)
    command_box(draw, "python scripts/evaluate_energy_demo.py", ["dataset: energy-demo/v1", "passed: 5", "total: 5"])


DRAWERS = [scene_0, scene_1, scene_2, scene_3, scene_4, scene_5, scene_6, scene_7]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def audio_duration(ffprobe: str, path: Path) -> float:
    output = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    )
    return float(output.strip())


def audio_for_scene(scene: Scene, mp3: Path, stamp: Path) -> None:
    signature = hashlib.sha256(scene.voiceover.encode("utf-8")).hexdigest()
    if mp3.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == signature:
        return
    gTTS(scene.voiceover, lang="en", tld="co.uk", slow=False).save(mp3)
    stamp.write_text(signature, encoding="utf-8")


def main() -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")
    BUILD.mkdir(exist_ok=True)
    segments: list[Path] = []
    for index, scene in enumerate(SCENES):
        image, draw = base(index)
        DRAWERS[index](draw)
        png = BUILD / f"scene_{index + 1:02d}.png"
        mp3 = BUILD / f"scene_{index + 1:02d}.mp3"
        stamp = BUILD / f"scene_{index + 1:02d}.voice.sha256"
        segment = BUILD / f"scene_{index + 1:02d}.mp4"
        image.save(png)
        audio_for_scene(scene, mp3, stamp)
        if (
            segment.exists()
            and segment.stat().st_size > 0
            and segment.stat().st_mtime >= max(png.stat().st_mtime, mp3.stat().st_mtime)
        ):
            segments.append(segment)
            continue
        duration = max(scene.minimum_seconds, math.ceil(audio_duration(ffprobe, mp3) + 1))
        run([
            ffmpeg, "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png), "-i", str(mp3),
            "-filter_complex", "[0:v]format=yuv420p[v];[1:a]adelay=400|400,apad[a]",
            "-map", "[v]", "-map", "[a]", "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", str(segment),
        ])
        segments.append(segment)
    concat = BUILD / "segments.txt"
    concat.write_text("".join(f"file '{path.as_posix()}'\n" for path in segments), encoding="utf-8")
    run([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", str(OUT),
    ])
    print(OUT)


if __name__ == "__main__":
    main()
