"""
Scan data/raw/{year}/ directories and return paths to EBS .csv.gz files.

"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

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
    *, data_dir: Path, instruments: Sequence[str], years: Sequence[str]
) -> Iterator[RawDataFile]:
    """
    Yield RawDataFile references for all matching .csv.gz files found under
    data_dir/{year}/ for the given instruments and years.

    :param data_dir: root directory that contains per-year subdirectories
    :param instruments: e.g. ["EUR_USD", "EUR_JPY", "USD_JPY"]
    :param years: e.g. [2024] or [2023, 2024]
    :raises UnableToScanDirectory: if a year directory exists but cannot be read
    """

    pass
