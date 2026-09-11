"""Render a short, client-facing teaser from live Energy demo UI captures."""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "energy_demo_ui_capture"
BUILD = ROOT / "energy_demo_client_teaser_build"
OUT = ROOT / "energy_asset_intelligence_client_teaser.mp4"

NAVY, PANEL, WHITE, MUTED, CYAN, GOLD, GREEN, RED = (
    (6, 17, 29), (10, 31, 47), (240, 247, 251), (151, 179, 198),
    (95, 221, 255), (255, 201, 90), (77, 211, 150), (255, 110, 117),
)


@dataclass(frozen=True)
class Scene:
    title: str
    minimum_seconds: int
    voiceover: str
    capture: str | None = None


SCENES = [
    Scene(
        "Offshore wind maintenance, explained", 18,
        "This is an explainable maintenance-intelligence workspace for offshore wind "
        "turbines. It helps operations teams identify which turbine needs review, and why, "
        "without relying on another disconnected dashboard.",
    ),
    Scene(
        "One clear, explainable priority", 25,
        "Here, WT-zero-one is prioritised for maintenance review. The workspace brings "
        "together the temperature observation, the current manufacturer threshold, and "
        "the open work order, so the recommendation is concrete and actionable.",
        "dashboard_current.png",
    ),
    Scene(
        "Confidence where evidence exists", 18,
        "The system also makes uncertainty visible. Where evidence is incomplete, it "
        "does not invent a maintenance conclusion. That is how an advisory tool earns "
        "the confidence of operations teams.",
        "dashboard_insufficient_evidence.png",
    ),
    Scene(
        "Built for trust and scale", 22,
        "Behind the business view, enterprise data is mapped into RDF, evaluated by "
        "version-controlled SPARQL, and checked by semantic validation. The technical "
        "trace remains available whenever an engineer or auditor needs it.",
        "dashboard_technical_trace.png",
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(path, size)


def write(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int,
          color: tuple[int, int, int] = WHITE, bold: bool = False,
          anchor: str | None = None) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color, anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, value: str, x: int, y: int, width: int,
         size: int, color: tuple[int, int, int] = WHITE) -> None:
    words, line, lines = value.split(), "", []
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font(size)) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, line in enumerate(lines):
        write(draw, (x, y + index * (size + 12)), line, size, color)


def base(scene: Scene, index: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), NAVY)
    draw = ImageDraw.Draw(image)
    for x in range(0, W, 64):
        draw.line((x, 0, x, H), fill=(11, 32, 50), width=1)
    for y in range(0, H, 48):
        draw.line((0, y, W, y), fill=(11, 32, 50), width=1)
    write(draw, (52, 35), "OFFSHORE WIND ASSET INTELLIGENCE", 15, CYAN, True)
    write(draw, (1228, 35), "CLIENT OVERVIEW", 13, GOLD, True, "ra")
    write(draw, (52, 85), f"0{index + 1}", 16, GOLD, True)
    write(draw, (90, 78), scene.title, 31, WHITE, True)
    return image, draw


def add_capture(image: Image.Image, draw: ImageDraw.ImageDraw, filename: str) -> None:
    source = CAPTURES / filename
    if not source.exists():
        raise RuntimeError(f"Missing capture {source}; run scripts/capture_energy_demo_ui.py")
    capture = Image.open(source).convert("RGB")
    capture.thumbnail((1080, 510), Image.Resampling.LANCZOS)
    x, y = (W - capture.width) // 2, 164 + (500 - capture.height) // 2
    draw.rounded_rectangle((x - 10, y - 10, x + capture.width + 10, y + capture.height + 10),
                           radius=12, fill=PANEL, outline=CYAN, width=2)
    image.paste(capture, (x, y))


def draw_scene(image: Image.Image, draw: ImageDraw.ImageDraw, scene: Scene, index: int) -> None:
    if scene.capture:
        add_capture(image, draw, scene.capture)
        if index == 2:
            write(draw, (640, 680), "Grounded recommendation · evidence in view · advisory only", 16, MUTED, anchor="mm")
        return
    draw.rounded_rectangle((90, 165, 1190, 610), radius=18, fill=PANEL, outline=(46, 98, 121), width=2)
    write(draw, (640, 235), "OFFSHORE WIND ASSET INTELLIGENCE", 22, GOLD, True, "mm")
    write(draw, (640, 315), "Which turbine needs maintenance review?", 38, WHITE, True, "mm")
    wrap(draw, "From SAP, telemetry and engineering bulletins to explainable turbine-maintenance decisions.",
         210, 405, 860, 23, MUTED)
    for x, label in ((280, "Operations"), (640, "Evidence"), (1000, "Action")):
        draw.ellipse((x - 20, 550, x + 20, 590), fill=GREEN)
        write(draw, (x, 620), label, 17, WHITE, True, "mm")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration(ffprobe: str, path: Path) -> float:
    value = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)], text=True,
    )
    return float(value.strip())


def audio(scene: Scene, path: Path) -> None:
    signature = hashlib.sha256(scene.voiceover.encode("utf-8")).hexdigest()
    stamp = path.with_suffix(".sha256")
    if path.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == signature:
        return
    gTTS(scene.voiceover, lang="en", tld="co.uk", slow=False).save(path)
    stamp.write_text(signature, encoding="utf-8")


def main() -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")
    BUILD.mkdir(exist_ok=True)
    segments: list[Path] = []
    for index, scene in enumerate(SCENES):
        image, draw = base(scene, index)
        draw_scene(image, draw, scene, index)
        png, mp3, mp4 = (BUILD / f"scene_{index + 1:02d}.png",
                         BUILD / f"scene_{index + 1:02d}.mp3",
                         BUILD / f"scene_{index + 1:02d}.mp4")
        image.save(png)
        audio(scene, mp3)
        scene_duration = max(scene.minimum_seconds, math.ceil(duration(ffprobe, mp3) + 1))
        run([ffmpeg, "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png), "-i", str(mp3),
             "-filter_complex", "[0:v]format=yuv420p[v];[1:a]adelay=400|400,apad[a]",
             "-map", "[v]", "-map", "[a]", "-t", str(scene_duration), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", str(mp4)])
        segments.append(mp4)
    concat = BUILD / "segments.txt"
    concat.write_text("".join(f"file '{segment.as_posix()}'\n" for segment in segments), encoding="utf-8")
    run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-fflags", "+genpts",
         "-avoid_negative_ts", "make_zero", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", str(OUT)])
    print(OUT)


if __name__ == "__main__":
    main()
