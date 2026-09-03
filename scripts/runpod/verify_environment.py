"""Verify a remote Runpod environment without reading EBS file contents."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch

from ebs_tft.application.usecases import research_protocol
from ebs_tft.data.repositories import raw_file as raw_file_repository

EXPECTED_PYTHON = (3, 13, 15)


def main() -> None:
    """Print environment evidence and fail when remote prerequisites are absent."""
    repository_dir = Path(__file__).resolve().parents[2]
    protocol = research_protocol.load_protocol(
        path=repository_dir / "notebooks" / "research_protocol.yaml"
    )
    discovered = tuple(
        raw_file_repository.find_raw_files(
            data_dir=protocol.data_dir,
            instruments=tuple(item.value for item in protocol.instruments),
            years=protocol.years,
        )
    )
    instrument_counts = Counter(item.instrument for item in discovered)
    workspace_usage = shutil.disk_usage(repository_dir)
    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_memory_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if torch.cuda.is_available()
            else None
        ),
        "raw_sessions": len(discovered),
        "raw_sessions_by_instrument": dict(sorted(instrument_counts.items())),
        "workspace_free_gib": round(workspace_usage.free / (1024**3), 2),
        "protocol": str(repository_dir / "notebooks" / "research_protocol.yaml"),
    }
    print(json.dumps(report, indent=2))
    failures: list[str] = []
    if sys.version_info[:3] != EXPECTED_PYTHON:
        failures.append("Python must be exactly 3.13.15")
    if platform.system() != "Linux":
        failures.append("remote execution must run on Linux")
    if not torch.cuda.is_available():
        failures.append("PyTorch cannot access an NVIDIA CUDA device")
    if not discovered:
        failures.append("no configured EBS raw sessions were discovered")
    if set(instrument_counts) != {item.value for item in protocol.instruments}:
        failures.append("raw data does not contain every configured instrument")
    if failures:
        raise SystemExit("Remote verification failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
