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
import polars as pl

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


def write_bars(path: Path, data: pl.DataFrame) -> None:
    """
    Write a polars DataFrame to a Parquet file at the given path.

    :param path: destination path (use get_file_path() to construct it)
    :param data: the processed bar DataFrame to persist
    :raises UnableToWriteBarsError: if the file cannot be written
    """
    try:
        # same as mkdir -p command
        path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(path)
    except Exception as exc:
        raise UnableToWriteBarsError(
            f"Cannot write processed bars to {path}: {exc}"
        ) from exc

    logger.debug(
        "Wrote processed bars",
        extra={"path": str(path), "rows": len(data), "columns": len(data.columns)},
    )


def read_bars(path: Path) -> pl.DataFrame:
    """
    Read a processed Parquet file and return it as a polars DataFrame.

    :param path: path to the .parquet file
    :raises UnableToReadBarsError: if the file does not exist or cannot be read
    """
    if not path.exists():
        raise UnableToReadBarsError(f"File not found: {path}")

    try:
        data = pl.read_parquet(path)
    except Exception as exc:
        raise UnableToReadBarsError(
            f"Cannot read processed bars from {path}: {exc}"
        ) from exc

    logger.debug(
        "Read processed bars",
        extra={"path": str(path), "rows": len(data), "columns": len(data.columns)},
    )
    return data


def find_processed_files() -> Iterator[ProcessedDataFile]:
    pass
