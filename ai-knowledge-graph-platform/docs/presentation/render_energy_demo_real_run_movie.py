"""Render a narrated Energy POC movie from an actual recorded workflow trace."""

from __future__ import annotations

import hashlib
import json
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
TRACE_PATH = ROOT / "energy_demo_real_run.json"
BUILD = ROOT / "energy_demo_real_run_movie_build"
OUT = ROOT / "energy_asset_intelligence_real_run_demo.mp4"
UI_CAPTURE = ROOT / "energy_demo_ui_capture"

BG, PANEL = (6, 17, 29), (10, 31, 47)
WHITE, MUTED, CYAN, GOLD, GREEN, RED = (
    (240, 247, 251), (151, 179, 198), (95, 221, 255),
    (255, 201, 90), (77, 211, 150), (255, 110, 117),
)


@dataclass(frozen=True)
class Scene:
    key: str
    title: str
    minimum_seconds: int
    voiceover: str


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 20,
         color: tuple[int, int, int] = WHITE, bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color, anchor=anchor)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
          outline: tuple[int, int, int] = (46, 98, 121), fill: tuple[int, int, int] = PANEL) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline, width=2)


def clean(value: str) -> str:
    return value.replace("\\u00b0", "°").replace("\r\n", "\n").replace("\r", "\n")


def lines_containing(output: str, patterns: tuple[str, ...], max_lines: int = 10) -> list[str]:
    selected = [line.strip() for line in clean(output).splitlines() if any(pattern in line for pattern in patterns)]
    return selected[:max_lines] or ["(expected output was not found in the captured transcript)"]


def excerpt(output: str, start: str, count: int) -> list[str]:
    values = clean(output).splitlines()
    index = next((i for i, line in enumerate(values) if start in line), 0)
    return [line.rstrip() for line in values[index:index + count] if line.strip()]


TRACE = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
if not TRACE["all_succeeded"]:
    raise RuntimeError("The workflow trace contains a failing command; inspect it before rendering.")
COMMANDS: dict[str, dict[str, object]] = TRACE["commands"]

RUN = str(COMMANDS["run_demo"]["stdout"])
HISTORICAL = str(COMMANDS["run_historical"]["stdout"])
EVALUATION = str(COMMANDS["evaluate"]["stdout"])
TESTS = str(COMMANDS["unit_tests"]["stdout"])

SCENES = [
    Scene(
        "run_summary", "Real workflow capture", 18,
        "This is a recorded run of the Energy Asset Intelligence proof of concept. "
        "The film is rendered from actual command stdout, exit codes, and timestamps "
        "captured in this repository. All six commands in this trace completed with "
        "exit code zero.",
    ),
    Scene(
        "create_source", "Create the source export", 18,
        "The run starts by generating the synthetic SAP-shaped SQLite export. The "
        "captured stdout confirms that the file was created under artifacts. This "
        "is the reproducible relational source used by the mapping validation.",
    ),
    Scene(
        "validate_r2rml", "Validate R2RML against the source", 22,
        "Next, the repository parses the energy R2RML mapping and validates it "
        "against the generated SQLite source. The captured result reports the "
        "energy-demo tenant, thirteen entity rows, and zero relation rows. The "
        "command uses validate-only, so it proves the source and mapping contract "
        "without issuing a graph write.",
    ),
    Scene(
        "run_demo", "Run the RDF evidence demo", 28,
        "The deterministic service then creates and exports an RDF Turtle graph. "
        "The captured maintenance assessment identifies WT-01, its 96 degree "
        "temperature, the 85 degree threshold from bulletin R2, and open work "
        "order WO-9001. The returned evidence includes the SAP, Snowflake, and "
        "SharePoint-shaped source identifiers.",
    ),
    Scene(
        "run_historical", "Run the historical bulletin scenario", 24,
        "A second real run supplies the first of May 2026. The captured historical "
        "answer selects bulletin R1 and its 90 degree threshold. This demonstrates "
        "the implemented bulletin effective-date selection.",
    ),
    Scene(
        "shacl", "Reject an invalid observation", 20,
        "The same demo run invokes SHACL validation against a deliberately "
        "incomplete observation. The captured result says conforms false and lists "
        "the missing value and unit requirements. This proves that the semantic "
        "validation code executed during the run.",
    ),
    Scene(
        "verification", "Evaluate and test the implementation", 25,
        "The recorded evaluator passes all five labelled scenarios. The targeted "
        "unit suite also passes all four tests, covering answers, tenant isolation, "
        "RDF serialization, SHACL validation, and the R2RML mapping contract. "
        "This is the real, repeatable workflow demonstrated by the film.",
    ),
    Scene(
        "ui_current", "Present the operational answer", 20,
        "The same implementation is presented through a client-facing workspace. "
        "An operations user sees the affected asset, the review priority, and the "
        "evidence-backed recommendation without needing to read RDF or SPARQL.",
    ),
    Scene(
        "ui_evidence", "Show evidence coverage", 18,
        "The interface also makes uncertainty visible. Assets without enough evidence "
        "are not given a confident maintenance conclusion. This is the safety boundary "
        "between a useful advisory system and an ungrounded answer.",
    ),
    Scene(
        "ui_technical", "Reveal the technical trace", 20,
        "For technical stakeholders, the same screen expands to show the RDF evidence, "
        "the version-controlled SPARQL query result, the authoritative bulletin, and "
        "the validation status behind the business answer.",
    ),
]


def base(index: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    for x in range(0, W, 64):
        draw.line((x, 0, x, H), fill=(11, 32, 50), width=1)
    for y in range(0, H, 48):
        draw.line((0, y, W, y), fill=(11, 32, 50), width=1)
    text(draw, (52, 33), "ENERGY ASSET INTELLIGENCE / REAL WORKFLOW CAPTURE", 15, CYAN, True)
    text(draw, (1225, 35), f"CAPTURED {TRACE['captured_at']}", 12, GOLD, True, "ra")
    text(draw, (52, 76), f"{index + 1:02d}", 16, GOLD, True)
    text(draw, (88, 70), SCENES[index].title, 30, WHITE, True)
    for item in range(len(SCENES)):
        x = 52 + item * 161
        draw.line((x, 118, x + 141, 118), fill=GOLD if item <= index else (36, 68, 86), width=4)
    return image, draw


def console(draw: ImageDraw.ImageDraw, command: str, output: list[str], status: str = "EXIT CODE 0") -> None:
    panel(draw, (65, 152, 1215, 652), outline=GREEN)
    text(draw, (98, 185), "CAPTURED COMMAND", 13, GOLD, True)
    text(draw, (1180, 185), status, 13, GREEN, True, "ra")
    wrapped = textwrap.wrap(command, 108)
    for index, line in enumerate(wrapped[:2]):
        text(draw, (98, 220 + index * 27), f"> {line}" if index == 0 else f"  {line}", 15, CYAN)
    start_y = 293
    for index, line in enumerate(output[:13]):
        color = GREEN if any(mark in line.lower() for mark in ("validated", "passed", "created", '"conforms": false')) else WHITE
        text(draw, (98, start_y + index * 26), line[:122], 15, color)


def draw_scene(index: int, image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    scene = SCENES[index]
    if scene.key == "run_summary":
        panel(draw, (85, 160, 1195, 385), outline=GOLD)
        text(draw, (640, 215), "RECORDED COMMAND TRACE", 20, GOLD, True, "mm")
        for item, y in (("create synthetic source", 270), ("validate R2RML", 310), ("export and run RDF demo", 350), ("run historical scenario", 390), ("evaluate labelled cases", 430), ("run targeted unit tests", 470)):
            text(draw, (260, y), "✓", 23, GREEN, True, "mm")
            text(draw, (300, y), item, 20, WHITE)
        panel(draw, (220, 525, 1060, 615), outline=GREEN)
        text(draw, (640, 560), "6 commands completed · 6 exit codes were zero", 24, GREEN, True, "mm")
    elif scene.key == "create_source":
        console(draw, str(COMMANDS["create_source"]["command"]), clean(str(COMMANDS["create_source"]["stdout"])).splitlines())
    elif scene.key == "validate_r2rml":
        console(draw, str(COMMANDS["validate_r2rml"]["command"]), clean(str(COMMANDS["validate_r2rml"]["stdout"])).splitlines())
    elif scene.key == "run_demo":
        output = ["Wrote RDF Turtle: artifacts\\energy-demo.ttl", ""]
        output += lines_containing(RUN, ('"answer":', '"authoritative_bulletin"', '"source_id"', '"mapping_version"', '"query_version"'), 9)
        console(draw, str(COMMANDS["run_demo"]["command"]), output)
    elif scene.key == "run_historical":
        output = excerpt(HISTORICAL, "What would the answer have been", 9)
        output += lines_containing(HISTORICAL, ('"authoritative_bulletin": "MFG-GBX-17-R1"',), 1)
        console(draw, str(COMMANDS["run_historical"]["command"]), output)
    elif scene.key == "shacl":
        console(draw, str(COMMANDS["run_demo"]["command"]), lines_containing(RUN, ("Invalid-batch validation:", "conforms", "energy:value", "energy:unit"), 6))
    elif scene.key == "verification":
        output = clean(EVALUATION).splitlines()[-7:] + [""] + clean(TESTS).splitlines()[-3:]
        console(draw, str(COMMANDS["evaluate"]["command"]), output)
    else:
        screenshot_name = {
            "ui_current": "dashboard_current.png",
            "ui_evidence": "dashboard_insufficient_evidence.png",
            "ui_technical": "dashboard_technical_trace.png",
        }[scene.key]
        screenshot_path = UI_CAPTURE / screenshot_name
        if not screenshot_path.exists():
            raise RuntimeError(f"Missing UI capture: {screenshot_path}. Run scripts/capture_energy_demo_ui.py first.")
        screenshot = Image.open(screenshot_path).convert("RGB")
        screenshot.thumbnail((1060, 515), Image.Resampling.LANCZOS)
        x = (W - screenshot.width) // 2
        y = 150 + (515 - screenshot.height) // 2
        panel(draw, (85, 145, 1195, 665), outline=CYAN)
        draw.rectangle((x - 2, y - 2, x + screenshot.width + 2, y + screenshot.height + 2), outline=WHITE, width=2)
        image.paste(screenshot, (x, y))


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def audio_duration(ffprobe: str, path: Path) -> float:
    output = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    )
    return float(output.strip())


def audio(scene: Scene, mp3: Path) -> None:
    signature = hashlib.sha256(scene.voiceover.encode("utf-8")).hexdigest()
    stamp = mp3.with_suffix(".sha256")
    if mp3.exists() and mp3.stat().st_size and stamp.exists() and stamp.read_text(encoding="utf-8") == signature:
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
        draw_scene(index, image, draw)
        png = BUILD / f"scene_{index + 1:02d}.png"
        mp3 = BUILD / f"scene_{index + 1:02d}.mp3"
        segment = BUILD / f"scene_{index + 1:02d}.mp4"
        image.save(png)
        audio(scene, mp3)
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
