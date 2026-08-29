"""
Define value objects and enums for the dataset subdomain.
"""

from __future__ import annotations

import datetime
import enum
from pathlib import Path

import attrs

from ebs_tft.domain.orderbook import models as orderbook_models


class LevelGroup(enum.StrEnum):
    """
    Identify the legacy depth groups used by the current processed-data layout.

    This enum remains until the canonical full-depth storage work replaces physical
    per-depth files. New research depth generation is driven by configuration.
    """

    L1 = "l1"
    L1_L5 = "l1_l5"
    L1_L10 = "l1_l10"

    def to_dir_name(self) -> str:
        """
        Return the processed-data directory name for this legacy group.
        """
        return self.value

    def max_level(self) -> int:
        """
        Return the deepest order-book level included in this legacy group.
        """
        if self is LevelGroup.L1:
            return 1
        if self is LevelGroup.L1_L5:
            return 5
        return orderbook_models.MAX_LEVELS

    def feature_columns(self) -> list[str]:
        """
        Return the ordered feature columns for this legacy level group.
        """
        max_level = self.max_level()
        return [
            orderbook_models.COL_TIMESTAMP,
            orderbook_models.COL_INSTRUMENT,
            orderbook_models.COL_MID_PRICE,
            orderbook_models.COL_SPREAD,
            *orderbook_models.all_bid_price_cols(max_level=max_level),
            *orderbook_models.all_bid_size_cols(max_level=max_level),
            *orderbook_models.all_ask_price_cols(max_level=max_level),
            *orderbook_models.all_ask_size_cols(max_level=max_level),
            orderbook_models.COL_QUOTE_IMBALANCE,
            orderbook_models.COL_BUY_VOLUME,
            orderbook_models.COL_SELL_VOLUME,
            orderbook_models.COL_TRADE_COUNT,
            orderbook_models.COL_DEAL_FLOW_IMBALANCE,
            orderbook_models.COL_VWAP,
        ]


class Split(enum.StrEnum):
    """
    Identify chronological model-development partitions.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@attrs.frozen
class SplitSpec:
    """
    Define temporal boundaries for train, validation, and test partitions.

    Boundary validation will be added with the leakage-safe dataset repair phase.
    """

    train_date_from: datetime.date
    train_date_to: datetime.date
    val_date_from: datetime.date
    val_date_to: datetime.date
    test_date_from: datetime.date
    test_date_to: datetime.date

    @property
    def full_date_from(self) -> datetime.date:
        """
        Return the earliest configured date.
        """
        return self.train_date_from

    @property
    def full_date_to(self) -> datetime.date:
        """
        Return the latest configured date.
        """
        return self.test_date_to

    def date_from_for(self, *, split: Split) -> datetime.date:
        """
        Return the inclusive start date for split.
        """
        if split is Split.TRAIN:
            return self.train_date_from
        if split is Split.VALIDATION:
            return self.val_date_from
        return self.test_date_from

    def date_to_for(self, *, split: Split) -> datetime.date:
        """
        Return the inclusive end date for split.
        """
        if split is Split.TRAIN:
            return self.train_date_to
        if split is Split.VALIDATION:
            return self.val_date_to
        return self.test_date_to


@attrs.frozen
class DatasetSpec:
    """
    Define the inputs required by the current dataset-building implementation.

    The future dataset repair replaces the legacy level group and row-based horizon
    with a cumulative depth specification and an elapsed-time horizon.
    """

    level_group: LevelGroup
    instruments: tuple[orderbook_models.Instrument, ...]
    split_spec: SplitSpec
    forecast_horizon_bars: int
    context_length_bars: int
    processed_dir: str

    @property
    def processed_path(self) -> Path:
        """
        Return processed_dir as a filesystem path.
        """
        return Path(self.processed_dir)
