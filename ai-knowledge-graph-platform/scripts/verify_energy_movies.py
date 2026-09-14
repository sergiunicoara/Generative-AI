"""Verify every static scene in the final MP4s; optionally transcribe final audio.

Requires the presentation tools (Pillow, gTTS, ffmpeg/ffprobe). --transcribe
additionally uses faster-whisper locally; it downloads the base.en model on
first use. No video or audio is uploaded. ASR is review evidence, not proof
of word-perfect speech or a substitute for human listening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PRESENTATION = Path(__file__).resolve().parents[1] / "docs" / "presentation"
RENDERERS = (
    "render_energy_demo_client_teaser.py",
    "render_energy_demo_movie.py",
    "render_energy_demo_real_run_movie.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path),
    ], text=True))


def checked_ffmpeg(*args: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-y", *args],
        capture_output=True, text=True, check=True,
    )
    if result.stderr.strip():
        raise RuntimeError(result.stderr)


def render_current_frame(namespace: dict, index: int):
    """Re-render in memory so a stale intermediate PNG cannot certify old slides."""
    if "DRAWERS" in namespace:
        image, draw = namespace["base"](index)
        namespace["DRAWERS"][index](draw)
    elif "TRACE" in namespace:
        image, draw = namespace["base"](index)
        namespace["draw_scene"](index, image, draw)
    else:
        scene = namespace["SCENES"][index]
        image, draw = namespace["base"](scene, index)
        namespace["draw_scene"](image, draw, scene, index)
    return image


def verify_movie(renderer: Path, build: Path, model=None) -> dict:
    from PIL import Image, ImageChops, ImageStat

    namespace = runpy.run_path(str(renderer))
    movie = namespace["OUT"]
    streams = probe(movie)
    video = next(item for item in streams["streams"] if item["codec_type"] == "video")
    audio = next(item for item in streams["streams"] if item["codec_type"] == "audio")
    if (video["codec_name"], video["width"], video["height"], audio["codec_name"]) != (
        "h264", 1280, 720, "aac",
    ):
        raise ValueError(f"Unexpected streams: {movie}")
    checked_ffmpeg("-i", str(movie), "-f", "null", "-")
    elapsed, scenes = 0.0, []
    for index, scene in enumerate(namespace["SCENES"], 1):
        stem = f"scene_{index:02d}"
        intermediate = namespace["BUILD"]
        duration = float(probe(intermediate / f"{stem}.mp4")["format"]["duration"])
        narration_duration = float(probe(intermediate / f"{stem}.mp3")["format"]["duration"])
        stamp_suffix = ".voice.sha256" if renderer.name == "render_energy_demo_movie.py" else ".sha256"
        expected_hash = hashlib.sha256(scene.voiceover.encode()).hexdigest()
        if (intermediate / f"{stem}{stamp_suffix}").read_text() != expected_hash:
            raise ValueError(f"Stale narration: {renderer.name} {index}")
        if duration < narration_duration + 0.4:
            raise ValueError(f"Clipped narration: {renderer.name} {index}")
        frame = build / f"{movie.stem}_{stem}.png"
        checked_ffmpeg("-ss", str(elapsed + duration / 2), "-i", str(movie),
                       "-frames:v", "1", "-update", "1", str(frame))
        with Image.open(frame) as actual, render_current_frame(namespace, index - 1) as expected:
            rms = max(ImageStat.Stat(ImageChops.difference(actual.convert("RGB"), expected.convert("RGB"))).rms)
        if rms > 12:
            raise ValueError(f"Final frame differs from scene source: {frame} RMS={rms}")
        entry = {
            "scene": index, "title": scene.title, "start_seconds": round(elapsed, 3),
            "duration_seconds": duration, "narration_seconds": narration_duration,
            "expected_voiceover": scene.voiceover, "frame_rms_error": round(rms, 3),
            "frame": str(frame.relative_to(PRESENTATION)), "narration_stamp_matches": True,
        }
        if model is not None:
            wav = build / f"{movie.stem}_{stem}.wav"
            checked_ffmpeg("-ss", str(elapsed), "-i", str(movie), "-t", str(duration),
                           "-vn", "-ac", "1", "-ar", "16000", str(wav))
            segments, _ = model.transcribe(str(wav), language="en", beam_size=5, vad_filter=True)
            entry["heard_by_local_asr"] = " ".join(segment.text.strip() for segment in segments)
        scenes.append(entry)
        elapsed += duration
        print(f"Verified {movie.name} scene {index}/{len(namespace['SCENES'])}", flush=True)
    final_duration = float(streams["format"]["duration"])
    if abs(final_duration - elapsed) > 0.25:
        raise ValueError(f"Concatenation duration mismatch: {movie}")
    return {
        "movie": movie.name, "sha256": sha256(movie), "renderer_sha256": sha256(renderer),
        "duration_seconds": final_duration, "full_decode_passed": True,
        "streams": "1280x720 H.264 / AAC", "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(PRESENTATION))
    build = PRESENTATION / "energy_movie_qa_build"
    build.mkdir(exist_ok=True)
    model = None
    if args.transcribe:
        from faster_whisper import WhisperModel

        model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4,
                             download_root=str(build / "models"))
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Full decode; every static scene/frame; narration cache and timing; optional local ASR.",
        "human_listening_completed": False,
        "asr_completed": args.transcribe,
        "trace_sha256": sha256(PRESENTATION / "energy_demo_real_run.json"),
        "shared_evidence_code_sha256": sha256(PRESENTATION / "energy_movie_evidence.py"),
        "ui_capture_sha256": {path.name: sha256(path) for path in sorted((PRESENTATION / "energy_demo_ui_capture").glob("*.png"))},
        "movies": [verify_movie(PRESENTATION / name, build, model) for name in RENDERERS],
    }
    output = PRESENTATION / "energy_movie_qa.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
