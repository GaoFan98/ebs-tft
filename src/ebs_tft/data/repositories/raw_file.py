"""
Scan data/raw/{year}/ directories and return paths to EBS .csv.gz files.

"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import attrs

logger = logging.getLogger(__name__)


# Constants
KNOWN_INSTRUMENTS: tuple[str, ...] = ("EUR_USD", "EUR_JPY", "USD_JPY")

_FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<trading_date>\d{8})-EBS_LVL2_(?P<instrument>[A-Z]{3}_[A-Z]{3})_0\.csv\.gz$"
)


# Exceptions
class UnableToScanDirectoryError(Exception):
    """
    Raised when a year directory cannot be scanned.
    """


# Value objects
@attrs.frozen
class RawDataFile:
    """
    A reference to a single EBS Level 2 .csv.gz file on disk.
    """

    path: Path
    # e.g. "EUR_USD", "EUR_JPY", "USD_JPY"
    instrument: str
    # Calendar year this file belongs to, derived from trading_date for
    # convenient grouping and filtering without string slicing at call sites.
    year: int
    # e.g. "20240102"
    trading_date: str


def find_raw_files(
    *, data_dir: Path, instruments: Sequence[str], years: Sequence[int]
) -> Iterator[RawDataFile]:
    """
    Yield RawDataFile references for all matching .csv.gz files found under
    data_dir/{year}/ for the given instruments and years.

    :param data_dir: root directory that contains per-year subdirectories
    :param instruments: e.g. ["EUR_USD", "EUR_JPY", "USD_JPY"]
    :param years: e.g. [2024] or [2023, 2024]
    :raises UnableToScanDirectory: if a year directory exists but cannot be read
    """
    instrument_filter = set(instruments)

    for year in sorted(years):
        year_dir = data_dir / str(year)

        if not year_dir.exists():
            logger.info(
                "Year directory not found, skipping",
                extra={"year_dir": str(year_dir)},
            )
            continue

        if not year_dir.is_dir():
            raise UnableToScanDirectoryError(
                f"Expected a directory at {year_dir} but found a file"
            )

        yield from _scan_year_dir(
            year_dir=year_dir, year=year, instrument_filter=instrument_filter
        )


def _scan_year_dir(
    *, year_dir: Path, year: int, instrument_filter: set[str]
) -> Iterator[RawDataFile]:
    """
    Scan single year directory and yield matched objects, sorting output by filename

    :raises UnableToScanDirectory: if the directory cannot be listed
    """
    try:
        entries = sorted(year_dir.glob("*.csv.gz"), key=lambda entry: entry.name)
    except OSError as exc:
        raise UnableToScanDirectoryError(
            f"Cannot list directory {year_dir}: {exc}"
        ) from exc

    for file_path in entries:
        raw_data_file = _parse_filename(
            file_path=file_path, year=year, instrument_filter=instrument_filter
        )
        if raw_data_file is None:
            # Filename did not match pattern or instrument not in filter
            logger.debug(
                "Skipping file",
                extra={"path": str(file_path)},
            )
            continue
        yield raw_data_file


def _parse_filename(
    *, file_path: Path, year: int, instrument_filter: set[str]
) -> RawDataFile | None:
    """
    Parse filename and return RawDataFile if matches, else None
    """
    matched = _FILENAME_PATTERN.match(file_path.name)
    if matched is None:
        return None

    instrument = matched.group("instrument")
    if instrument not in instrument_filter:
        return None

    trading_date = matched.group("trading_date")

    # Year in the filename should match the directory year.
    if not trading_date.startswith(str(year)):
        logger.warning(
            "File year mismatch: file is in wrong year directory",
            extra={
                "path": str(file_path),
                "directory_year": year,
                "filename_date": trading_date,
            },
        )

    return RawDataFile(
        path=file_path, instrument=instrument, year=year, trading_date=trading_date
    )
