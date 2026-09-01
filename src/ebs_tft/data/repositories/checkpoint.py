"""Persist PyTorch checkpoint payloads atomically."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch


class InvalidCheckpointError(Exception):
    """Indicate that a checkpoint is missing or has an unexpected payload."""


def write(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write one checkpoint atomically so readers never observe partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read(*, path: Path) -> dict[str, object]:
    """
    Read one weights-only checkpoint payload.

    :raises InvalidCheckpointError: if the checkpoint is missing or not a mapping
    """
    if not path.is_file():
        raise InvalidCheckpointError(f"checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise InvalidCheckpointError("checkpoint payload must be a string-key mapping")
    return cast(dict[str, object], loaded)
