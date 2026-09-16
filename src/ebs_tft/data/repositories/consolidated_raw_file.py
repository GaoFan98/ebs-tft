"""Discover consolidated legacy EBS Level 2 source files."""

from __future__ import annotations

import datetime
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import attrs

_FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<trading_date>\d{8})-EBS_Level2_0_0_0\.csv\.gz$"
)


class UnableToScanConsolidatedDirectoryError(Exception):
    """Indicate that a consolidated source directory could not be inspected."""


class InvalidConsolidatedRawFileError(Exception):
    """Indicate invalid identity or metadata for one consolidated source file."""


class DuplicateConsolidatedRawFileError(Exception):
    """Indicate more than one consolidated source for a trading date."""


@attrs.frozen
class ConsolidatedRawDataFile:
    """Reference one consolidated multi-instrument EBS gzip source."""

    path: Path
    trading_date: datetime.date
    size_bytes: int


def find_consolidated_files(
    *, source_dir: Path, expected_year: int
) -> Iterator[ConsolidatedRawDataFile]:
    """
    Yield consolidated sources in deterministic trading-date order.

    Canonical per-instrument gzip files in the same directory are ignored. Any
    other gzip filename is rejected rather than silently omitted.

    :raises UnableToScanConsolidatedDirectoryError: if source_dir is unavailable
    :raises InvalidConsolidatedRawFileError: if a gzip source is malformed
    :raises DuplicateConsolidatedRawFileError: if a date occurs more than once
    """
    if isinstance(expected_year, bool) or expected_year < 1:
        raise ValueError("expected_year must be a positive integer")
    if not source_dir.is_dir():
        raise UnableToScanConsolidatedDirectoryError(
            f"Consolidated source directory not found: {source_dir}"
        )
    try:
        paths = tuple(sorted(source_dir.iterdir(), key=lambda item: item.name))
    except OSError as exc:
        raise UnableToScanConsolidatedDirectoryError(
            f"Unable to list consolidated source directory: {source_dir}"
        ) from exc
    discovered: list[ConsolidatedRawDataFile] = []
    seen: set[datetime.date] = set()
    for path in paths:
        if not path.name.endswith(".csv.gz"):
            continue
        matched = _FILENAME_PATTERN.fullmatch(path.name)
        if matched is None:
            if re.fullmatch(r"\d{8}-EBS_LVL2_[A-Z]+_[A-Z]+_0\.csv\.gz", path.name):
                continue
            raise InvalidConsolidatedRawFileError(
                f"Malformed consolidated EBS filename: {path}"
            )
        try:
            trading_date = datetime.datetime.strptime(
                matched.group("trading_date"), "%Y%m%d"
            ).date()
        except ValueError as exc:
            raise InvalidConsolidatedRawFileError(
                f"Invalid date in consolidated EBS filename: {path}"
            ) from exc
        if trading_date.year != expected_year:
            raise InvalidConsolidatedRawFileError(
                f"Consolidated file year does not match expected year: {path}"
            )
        if not path.is_file():
            raise InvalidConsolidatedRawFileError(
                f"Consolidated EBS candidate is not a regular file: {path}"
            )
        if trading_date in seen:
            raise DuplicateConsolidatedRawFileError(
                f"Duplicate consolidated file for {trading_date.isoformat()}"
            )
        seen.add(trading_date)
        discovered.append(
            ConsolidatedRawDataFile(
                path=path,
                trading_date=trading_date,
                size_bytes=path.stat().st_size,
            )
        )
    yield from discovered


def get_content_fingerprint(*, source: ConsolidatedRawDataFile) -> str:
    """Return a SHA-256 fingerprint of the exact compressed source bytes."""
    digest = hashlib.sha256()
    try:
        with source.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InvalidConsolidatedRawFileError(
            f"Unable to fingerprint consolidated EBS file: {source.path}"
        ) from exc
    return digest.hexdigest()
