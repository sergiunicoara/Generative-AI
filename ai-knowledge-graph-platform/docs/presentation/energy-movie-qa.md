# Energy movie QA and handoff

## What is ready

The three local MP4s in this folder are rebuilt from the recorded synthetic
workflow and actual UI captures. All 24 static scenes have been reviewed.
`energy_movie_qa.json` binds automated QA evidence to the exact movie hashes
and includes every scene's intended narration and local ASR transcript.

Corrections made during final review:

- Read R2RML counts from the successful trace: **13 entity rows / 3 relation rows**.
  Missing or failed capture evidence now fails closed rather than inventing counts.
- Keep **91.5°C**, **85°C**, revision R2, and the synthetic/advisory boundary consistent.
- Show **read scope plus write scope** for the governed POST operations.
- Remove unsupported scale wording, repair overlapping labels and unsupported glyphs,
  wrap long evidence lines, and show the complete `--validate-only` option.
- Show actual scorecard results with their tiny-fixture qualification.
- Enlarge and explicitly label the technical screenshot excerpt; preserve the
  unmodified full capture alongside it.
- Display **Synthetic data · Advisory POC · No equipment control** in every scene.

## Rebuild

From the project root, use an isolated authoring environment and install
`requirements-presentation.txt`. Install `ffmpeg` and `ffprobe` separately
and put both executables on PATH. These renderers currently use Windows Arial
font paths. gTTS generates speech through an external service; do not put
customer/private content into narration without approval.

```powershell
python -m pip install -r requirements-presentation.txt
python docs/presentation/render_energy_demo_client_teaser.py
python docs/presentation/render_energy_demo_movie.py
python docs/presentation/render_energy_demo_real_run_movie.py
python scripts/verify_energy_movies.py
```

The last command decodes each complete MP4, checks streams, verifies that scene
and narration durations fit, checks narration cache signatures, and compares
every static scene's midpoint against a fresh render (allowing encoding loss).
It fails on missing files or failed assertions. Intermediate frames and audio
live under ignored `*_build/` directories, not in the release.

For local speech recognition, install the optional tool in an isolated QA
environment, without changing the application lockfile:

```powershell
python -m pip install faster-whisper==1.2.1
python scripts/verify_energy_movies.py --transcribe
```

This downloads the `base.en` model on first use into the ignored QA build
directory; transcription runs locally. Review `heard_by_local_asr` against
`expected_voiceover`. Acronym/homophone errors in ASR are not automatically
evidence of a narration error. Running without `--transcribe` produces a new
report explicitly marked `asr_completed: false`; it does not carry forward
old speech checks. Rerun QA after any movie, renderer or input change.

## Verified scope and remaining human sign-off

The follow-up local regression run passed **106 Energy/semantic-model tests**,
including six new mapping-evidence tests. Canonical model compilation verified
four generated artifacts. Runtime code and its lockfile were not changed by
this presentation follow-up; the earlier clean-snapshot and live Neo4j evidence
remain earlier checks, not newly repeated full-platform acceptance.

Automated QA and transcript review do not establish a pleasant voice or
word-perfect pronunciation. Before sending externally, listen to each final
movie once on the intended device, check technical acronyms and pacing, and
confirm sound and readability at the intended playback size. Human listening
has **not** been marked complete. The slides do not contain word-timed subtitles.

This package demonstrates a production-oriented synthetic POC. It does not
certify customer connectors, customer identity/network deployment, enterprise
capacity, operational SLAs, or equipment control.
