"""
Read and write processed 1-minute bar data as Parquet files.

Directory structure as following:
    data/processed/{level_group}/{instrument}/{trading_date}.parquet
    e.g. data/processed/l1/EUR_USD/20240102.parquet
"""

from __future__ import annotations

import logging

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

    pass
