"""
Discover and validate raw EBS data file references.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import attrs

logger = logging.getLogger(__name__)

_FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<trading_date>\d{8})-EBS_LVL2_"
    r"(?P<instrument>[A-Z]+_[A-Z]+)_0\.csv\.gz$"
)


class UnableToScanDirectoryError(Exception):
    """
    Indicate that a raw-data directory could not be inspected.
    """


class InvalidRawDataFileError(Exception):
    """
    Indicate that a candidate raw-data file has invalid identity or metadata.
    """


class DuplicateRawDataFileError(Exception):
    """
    Indicate duplicate files for one instrument and trading date.
    """


class InvalidRawFileQueryError(Exception):
    """
    Indicate invalid instrument or year filters supplied by a caller.
    """


@attrs.frozen
class RawDataFile:
    """
    Reference one validated EBS Level 2 compressed CSV file.
    """

    path: Path
    instrument: str
    trading_date: datetime.date
    size_bytes: int
    modified_time_ns: int

    @property
    def year(self) -> int:
        """
        Return the year encoded by trading_date.
        """
        return self.trading_date.year

    @property
    def trading_date_label(self) -> str:
        """
        Return trading_date in the compact filename representation.
        """
        return self.trading_date.strftime("%Y%m%d")

    @property
    def fingerprint(self) -> str:
        """
        Return a stable metadata fingerprint for discovery-stage comparisons.

        This is not a content checksum. Content hashing belongs to the later manifest
        phase because reading every multi-gigabyte input is outside file discovery.
        """
        return f"{self.size_bytes}:{self.modified_time_ns}"


def get_content_fingerprint(*, raw_data_file: RawDataFile) -> str:
    """Return a strong fingerprint of the exact compressed source bytes."""
    digest = hashlib.sha256()
    try:
        with raw_data_file.path.open(mode="rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InvalidRawDataFileError(
            f"Unable to fingerprint raw EBS file: {raw_data_file.path}"
        ) from exc
    return digest.hexdigest()


def find_raw_files(
    *,
    data_dir: Path,
    instruments: Sequence[str],
    years: Sequence[int],
) -> Iterator[RawDataFile]:
    """
    Yield validated raw files in deterministic instrument/date/path order.

    Missing year directories are treated as absent optional data and logged.

    :raises InvalidRawFileQueryError: if filters are empty, duplicated, or invalid
    :raises UnableToScanDirectoryError: if an existing year path cannot be scanned
    :raises InvalidRawDataFileError: if a candidate file has invalid metadata
    :raises DuplicateRawDataFileError: if a pair/date has more than one file
    """
    instrument_values = _validate_instruments(instruments=instruments)
    requested_years = _validate_years(years=years)
    discovered: list[RawDataFile] = []

    for year in requested_years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            logger.info(
                "Year directory not found",
                extra={"year_dir": str(year_dir)},
            )
            continue
        if not year_dir.is_dir():
            raise UnableToScanDirectoryError(
                f"Expected raw-data directory but found another file type: {year_dir}"
            )
        discovered.extend(
            _scan_year_dir(
                year_dir=year_dir,
                expected_year=year,
                instrument_values=instrument_values,
            )
        )

    ordered = sorted(
        discovered,
        key=lambda item: (
            item.instrument,
            item.trading_date,
            item.path.name,
        ),
    )
    _validate_no_duplicates(files=ordered)
    yield from ordered


def _validate_instruments(*, instruments: Sequence[str]) -> frozenset[str]:
    if not instruments:
        raise InvalidRawFileQueryError("instruments must not be empty")
    if any(not isinstance(item, str) for item in instruments):
        raise InvalidRawFileQueryError("instruments must contain only strings")
    if any(re.fullmatch(r"[A-Z]+_[A-Z]+", item) is None for item in instruments):
        raise InvalidRawFileQueryError(
            "instruments must use uppercase underscore-delimited symbols"
        )
    if len(set(instruments)) != len(instruments):
        raise InvalidRawFileQueryError("instruments must not contain duplicates")
    return frozenset(instruments)


def _validate_years(*, years: Sequence[int]) -> tuple[int, ...]:
    if not years:
        raise InvalidRawFileQueryError("years must not be empty")
    if any(isinstance(year, bool) or not isinstance(year, int) for year in years):
        raise InvalidRawFileQueryError("years must contain only integers")
    if any(year < 1 for year in years):
        raise InvalidRawFileQueryError("years must contain positive values")
    if len(set(years)) != len(years):
        raise InvalidRawFileQueryError("years must not contain duplicates")
    return tuple(sorted(years))


def _scan_year_dir(
    *,
    year_dir: Path,
    expected_year: int,
    instrument_values: frozenset[str],
) -> list[RawDataFile]:
    try:
        entries = sorted(year_dir.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        raise UnableToScanDirectoryError(
            f"Unable to list raw-data directory: {year_dir}"
        ) from exc

    files: list[RawDataFile] = []
    for path in entries:
        if not path.name.endswith(".csv.gz"):
            continue
        file = _parse_candidate(
            path=path,
            expected_year=expected_year,
            instrument_values=instrument_values,
        )
        if file is not None:
            files.append(file)
    return files


def _parse_candidate(
    *, path: Path, expected_year: int, instrument_values: frozenset[str]
) -> RawDataFile | None:
    matched = _FILENAME_PATTERN.fullmatch(path.name)
    if matched is None:
        raise InvalidRawDataFileError(f"Malformed raw EBS filename: {path}")

    try:
        trading_date = datetime.datetime.strptime(
            matched.group("trading_date"), "%Y%m%d"
        ).date()
    except ValueError as exc:
        raise InvalidRawDataFileError(
            f"Invalid date in raw EBS filename: {path}"
        ) from exc
    if trading_date.year != expected_year:
        raise InvalidRawDataFileError(
            f"Raw file year does not match its directory: {path}"
        )

    instrument_value = matched.group("instrument")
    if instrument_value not in instrument_values:
        return None

    if not path.is_file():
        raise InvalidRawDataFileError(
            f"Raw EBS candidate is not a regular file: {path}"
        )

    try:
        stat = path.stat()
    except OSError as exc:
        raise InvalidRawDataFileError(
            f"Unable to read raw EBS file metadata: {path}"
        ) from exc

    return RawDataFile(
        path=path,
        instrument=instrument_value,
        trading_date=trading_date,
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
    )


def _validate_no_duplicates(*, files: Sequence[RawDataFile]) -> None:
    seen: dict[tuple[str, datetime.date], Path] = {}
    for file in files:
        key = (file.instrument, file.trading_date)
        previous = seen.get(key)
        if previous is not None:
            raise DuplicateRawDataFileError(
                f"Duplicate raw files for {file.instrument} "
                f"on {file.trading_date.isoformat()}: {previous}, {file.path}"
            )
        seen[key] = file.path
