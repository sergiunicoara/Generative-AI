"""Guards against a few structural mistakes that have already happened once
and are otherwise invisible until someone notices CI "isn't running" or a
generated/scratch artifact shows up in a diff.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent  # this project lives inside a monorepo


class TestNoDeadInProjectCiWorkflow:
    """GitHub Actions only reads workflows from the REPOSITORY root's
    .github/workflows/, never from a subdirectory. This project's real CI
    lives at <repo_root>/.github/workflows/ai-knowledge-graph-platform-ci.yml
    (its own header comment explains why). A copy at
    <project_root>/.github/workflows/ci.yml looks like it should run but
    never does -- GitHub silently ignores it -- so a green "CI" checkmark
    could mean the file simply wasn't executed, not that anything passed.

    This has already happened twice: added in f4ff7ea, correctly relocated
    to the repo root in 328f8fc ("fix dead in-project CI workflow
    location"), then silently reintroduced in f5dbf28 with no comment
    acknowledging the earlier fix -- almost certainly a stale branch merge
    or an agent recreating a "missing" file without checking why it wasn't
    there. Nothing short of an explicit, checked assertion catches this,
    since the file's mere presence looks correct to a human skimming the
    tree and the workflow itself never fails (it never runs).
    """

    def test_no_workflow_directory_inside_the_project(self) -> None:
        in_project_workflows = PROJECT_ROOT / ".github" / "workflows"
        assert not in_project_workflows.exists(), (
            f"{in_project_workflows} exists but GitHub Actions cannot see it "
            "-- workflows must live at the repository root "
            f"({REPO_ROOT / '.github' / 'workflows'}), not inside this "
            "project directory. See test class docstring for the history."
        )

    def test_the_real_workflow_exists_at_the_repo_root(self) -> None:
        real_workflow = REPO_ROOT / ".github" / "workflows" / "ai-knowledge-graph-platform-ci.yml"
        assert real_workflow.is_file(), (
            f"expected the real CI workflow at {real_workflow} -- if it was "
            "renamed or moved, update this test alongside it rather than "
            "deleting the check."
        )
