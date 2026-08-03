"""
Read and write processed 1-minute bar data as Parquet files.

Directory structure as following:
    data/processed/{level_group}/{instrument}/{trading_date}.parquet
    e.g. data/processed/l1/EUR_USD/20240102.parquet
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import attrs

logger = logging.getLogger(__name__)


# Constants
LEVEL_GROUPS: tuple[str, ...] = ("l1", "l1_l5", "l1_l10")


# Exceptions
class UnableToWriteBarsError(Exception):
    """
    Raised when a processed bar file cannot be written to disk.
    """


class UnableToReadBarsError(Exception):
    """
    Raised when a processed bar file cannot be read from disk.
    """


# Value objects
@attrs.frozen
class ProcessedDataFile:
    """
    Reference to a single processed Parquet file on disk.

    Path structure: processed_dir/{level_group}/{instrument}/{trading_date}.parquet
    """

    path: Path
    # "l1", "l1_l5", or "l1_l10"
    level_group: str
    # "EUR_USD", "EUR_JPY", "USD_JPY"
    instrument: str
    trading_date: str


def get_file_path(
    *, processed_dir: Path, level_group: str, instrument=str, trading_date=str
) -> Path:
    return processed_dir / level_group / instrument / f"{trading_date}.parquet"


def is_processed(
    *,
    processed_dir: Path,
    level_group: str,
    instrument: str,
    trading_date: str,
) -> bool:
    """
    Return True if a processed Parquet file already exists for this combination.
    """
    path = get_file_path(
        processed_dir=processed_dir,
        level_group=level_group,
        instrument=instrument,
        trading_date=trading_date,
    )
    return path.exists()


def write_bars() -> None:
    pass


def read_bars() -> None:
    pass


def find_processed_files() -> Iterator[ProcessedDataFile]:
    pass
