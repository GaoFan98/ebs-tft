"""Prepare bounded experiment artifact directories safely."""

from __future__ import annotations

import shutil
from pathlib import Path


class CompletedArtifactExistsError(Exception):
    """Indicate that a completed run would be overwritten without permission."""


def prepare_run_directory(
    *, path: Path, replace: bool, replacement_parent: Path | None = None
) -> None:
    """
    Prepare one explicit output directory while preserving resumable partial runs.

    :raises CompletedArtifactExistsError: if a completed output exists
    """
    resolved = path.resolve()
    if replace and resolved.exists():
        if not _replacement_is_bounded(
            path=resolved, replacement_parent=replacement_parent
        ):
            raise ValueError("replace is restricted to a bounded output directory")
        shutil.rmtree(resolved)
    if (resolved / "run_summary.json").exists():
        raise CompletedArtifactExistsError(
            f"completed output already exists: {resolved}; use --replace-output"
        )
    resolved.mkdir(parents=True, exist_ok=True)


def _replacement_is_bounded(*, path: Path, replacement_parent: Path | None) -> bool:
    if path == path.parent:
        return False
    if replacement_parent is None:
        return path.name.endswith("_outputs")
    parent = replacement_parent.resolve()
    return path.parent == parent
