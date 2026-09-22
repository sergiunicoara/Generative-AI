#!/usr/bin/env python
"""Verify Energy from the exact Git snapshot that would be delivered.

With staged changes present, ``git write-tree`` is archived; otherwise HEAD is
used. The check never reads the developer's working tree after extraction.
CI uses the default fresh-install path. It installs the checked-in runtime
lockfile plus only the focused verification tools, avoiding an unbounded
resolver pass over the broad development requirements, then runs ``pip
check`` to confirm those installs are actually mutually compatible --
``--no-deps`` and passing a couple of focused tests prove neither of those on
their own. ``--no-install`` is useful for a fast local smoke run when
dependencies are already available.
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
            if sys.platform in {"linux", "win32"}:
                # The default PyPI torch wheel now declares CUDA runtime
                # packages even when these CPU-only verification jobs cannot
                # use them. Install the matching official CPU wheel first;
                # the lockfile's `torch==X.Y.Z` pin accepts its `+cpu` local
                # version, and pip check can then validate the actual runtime
                # dependency set without pulling a CUDA toolkit into CI.
                locked_torch = next(
                    (line.split("==", 1)[1].strip() for line in (project / "requirements.lock").read_text().splitlines()
                     if line.startswith("torch==")),
                    None,
                )
                if locked_torch is None:
                    raise RuntimeError("requirements.lock does not pin torch")
                _run(
                    [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
                     "--index-url", "https://download.pytorch.org/whl/cpu", f"torch=={locked_torch}+cpu"],
                    cwd=project,
                )
            # setuptools isn't in requirements.lock -- pip-compile excludes it
            # by policy (pinning your own build toolchain via a normal
            # requirements file is its own hazard), so it's whatever version
            # `venv`'s own bootstrap happens to ship. torch's wheel declares a
            # floor pip check will not silently ignore; satisfy it explicitly
            # rather than let a bootstrap-version accident fail the gate.
            _run([python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "setuptools>=77.0.3"], cwd=project)
            _run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "-r", "requirements.lock"], cwd=project)
            _run([python, "-m", "pip", "install", "--disable-pip-version-check", "pytest", "pytest-asyncio", "ruff"], cwd=project)
            # Explicit, visible evidence of the version actually left standing
            # after every later install -- not merely "pip check raised
            # nothing about it," which proves the same thing but leaves no
            # trace of what was actually checked.
            _run([python, "-c", "import setuptools; print('setuptools', setuptools.__version__)"], cwd=project)
            # --no-deps above deliberately skips resolution against the
            # lockfile's own pins (that's the whole point -- no unbounded
            # resolver pass). But it also means nothing has yet confirmed
            # those pins are mutually compatible, or that installing the
            # three focused tools afterward (which DOES resolve their own
            # deps) didn't quietly upgrade something the lockfile pinned.
            # `pip check` inspects installed package metadata for exactly
            # that: unmet or conflicting requirements. Passing tests is not
            # the same claim as "this environment's dependencies are
            # consistent" -- a real but latent conflict can sit in a code
            # path the two focused test files don't exercise.
            _run([python, "-m", "pip", "check"], cwd=project)

        env = {**os.environ, "ENV": "test", "PYTHONDONTWRITEBYTECODE": "1"}
        _run([python, "-m", "ruff", "check", "graphrag/", "api/", "scripts/", "tests/"], cwd=project, env=env)
        _run([python, "-m", "pytest", "tests/unit/test_energy_demo_e2e.py", "tests/unit/test_energy_lpg_projection.py", "-q"], cwd=project, env=env)
        print(f"Clean snapshot verification passed: {tree}")


if __name__ == "__main__":
    main()
