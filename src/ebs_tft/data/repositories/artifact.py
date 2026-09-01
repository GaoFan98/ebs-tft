"""Prepare bounded experiment artifact directories safely."""

from __future__ import annotations

import shutil
from pathlib import Path


class CompletedArtifactExistsError(Exception):
    """Indicate that a completed run would be overwritten without permission."""


def prepare_run_directory(*, path: Path, replace: bool) -> None:
    """
    Prepare one explicit output directory while preserving resumable partial runs.

    :raises CompletedArtifactExistsError: if a completed output exists
    """
    resolved = path.resolve()
    if replace and resolved.exists():
        if resolved == resolved.parent or not resolved.name.endswith("_outputs"):
            raise ValueError("replace is restricted to named *_outputs directories")
        shutil.rmtree(resolved)
    if (resolved / "run_summary.json").exists():
        raise CompletedArtifactExistsError(
            f"completed output already exists: {resolved}; use --replace-output"
        )
    resolved.mkdir(parents=True, exist_ok=True)
