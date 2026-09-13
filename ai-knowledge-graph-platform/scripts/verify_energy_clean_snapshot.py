#!/usr/bin/env python
"""Verify Energy from the exact Git snapshot that would be delivered.

With staged changes present, ``git write-tree`` is archived; otherwise HEAD is
used. The check never reads the developer's working tree after extraction.
CI uses the default fresh-install path. ``--no-install`` is useful for a fast
local smoke run when dependencies are already available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _git_tree() -> str:
    safe = str(REPO).replace("\\", "/")
    git = ["git", "-c", f"safe.directory={safe}"]
    staged = subprocess.run(git + ["diff", "--cached", "--quiet"], cwd=REPO).returncode != 0
    if staged:
        tree = subprocess.run(git + ["write-tree"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        return tree
    return subprocess.run(git + ["rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract a trusted local Git archive on Python 3.11+ safely."""
    destination_root = destination.resolve()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"Archive member escapes destination: {member.name}")
            stream.extract(member, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-install", action="store_true", help="Use the current interpreter instead of creating/installing a venv")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="energy-clean-snapshot-") as temporary:
        workspace = Path(temporary) / "repo"
        workspace.mkdir()
        archive = Path(temporary) / "snapshot.tar"
        safe = str(REPO).replace("\\", "/")
        tree = _git_tree()
        subprocess.run(
            ["git", "-c", f"safe.directory={safe}", "archive", "--format=tar", "--output", str(archive), tree, "ai-knowledge-graph-platform"],
            cwd=REPO, check=True,
        )
        _extract_archive(archive, workspace)
        project = workspace / "ai-knowledge-graph-platform"

        if args.no_install:
            python = sys.executable
        else:
            venv_dir = Path(temporary) / "venv"
            venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
            python = str(venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
            _run([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-dev.txt"], cwd=project)

        env = {**os.environ, "ENV": "test", "PYTHONDONTWRITEBYTECODE": "1"}
        _run([python, "-m", "ruff", "check", "graphrag/", "api/", "scripts/", "tests/"], cwd=project, env=env)
        _run([python, "-m", "pytest", "tests/unit/test_energy_demo_e2e.py", "tests/unit/test_energy_lpg_projection.py", "-q"], cwd=project, env=env)
        print(f"Clean snapshot verification passed: {tree}")


if __name__ == "__main__":
    main()
