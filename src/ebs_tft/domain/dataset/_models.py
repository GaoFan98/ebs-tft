# Value objects and enums for the dataset sub-domain.

from __future__ import annotations

import datetime
import enum
from pathlib import Path

import attrs

from ebs_tft.domain.orderbook import _models as ob

# Enums


class LevelGroup(enum.StrEnum):
    """
    The three depth-level groupings used in the research experiment.

    Each member corresponds to a processed Parquet directory and a distinct
    feature column set. The ordering L1 < L1_L5 < L1_L10
    """

    L1 = "l1"
    L1_L5 = "l1_l5"
    L1_L10 = "l1_l10"

    def to_dir_name(self) -> str:
        """
        Return the Parquet directory name for this group.

        Matches the directory structure written by the exporter:
        data/processed/{dir_name}/{instrument}/{date}.parquet
        """
        return self.value

    def max_level(self) -> int:
        """
        Return the deepest order book level included in this group.
        """
        if self is LevelGroup.L1:
            return 1
        if self is LevelGroup.L1_L5:
            return 5

        return ob.MAX_LEVELS

    def feature_columns(self) -> list[str]:
        """
        Return the full ordered list of feature columns for this level group.

        Column order:
            timestamp, instrument, mid_price, spread,
            bid_price_l1..lN, bid_size_l1..lN,
            ask_price_l1..lN, ask_size_l1..lN,
            quote_imbalance, buy_volume, sell_volume,
            trade_count, deal_flow_imbalance, vwap
        """
        max_lvl = self.max_level()
        return [
            ob.COL_TIMESTAMP,
            ob.COL_INSTRUMENT,
            ob.COL_MID_PRICE,
            ob.COL_SPREAD,
            *ob.all_bid_price_cols(max_level=max_lvl),
            *ob.all_bid_size_cols(max_level=max_lvl),
            *ob.all_ask_price_cols(max_level=max_lvl),
            *ob.all_ask_size_cols(max_level=max_lvl),
            ob.COL_QUOTE_IMBALANCE,
            ob.COL_BUY_VOLUME,
            ob.COL_SELL_VOLUME,
            ob.COL_TRADE_COUNT,
            ob.COL_DEAL_FLOW_IMBALANCE,
            ob.COL_VWAP,
        ]


class Split(enum.StrEnum):
    """
    The three temporal splits used for model training and evaluation.

    Always chronological — TRAIN precedes VALIDATION which precedes TEST.
    Random splits are forbidden for time series (they leak future into training).
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# Value objects


@attrs.frozen
class SplitSpec:
    """
    Temporal boundaries for the three-way train/validation/test split.

    Default 2024 splits (from configs/training.yaml):
        train:  2024-01-01 -> 2024-09-30  (~75%)
        val:    2024-10-01 -> 2024-11-30  (~15%)
        test:   2024-12-01 -> 2024-12-31  (~10%)
    """

    train_date_from: datetime.date
    train_date_to: datetime.date
    val_date_from: datetime.date
    val_date_to: datetime.date
    test_date_from: datetime.date
    test_date_to: datetime.date

    @property
    def full_date_from(self) -> datetime.date:
        """Earliest date across all splits."""
        return self.train_date_from

    @property
    def full_date_to(self) -> datetime.date:
        """Latest date across all splits."""
        return self.test_date_to

    def date_from_for(self, *, split: Split) -> datetime.date:
        """Return the start date for the given split."""
        if split is Split.TRAIN:
            return self.train_date_from
        if split is Split.VALIDATION:
            return self.val_date_from
        return self.test_date_from

    def date_to_for(self, *, split: Split) -> datetime.date:
        """Return the end date (inclusive) for the given split."""
        if split is Split.TRAIN:
            return self.train_date_to
        if split is Split.VALIDATION:
            return self.val_date_to
        return self.test_date_to


@attrs.frozen
class DatasetSpec:
    """
    Complete specification for building a model-ready dataset.

    Attributes:
        level_group:
            Which depth group to use (L1, L1_L5, or L1_L10). Determines
            both the Parquet directory to read from and the feature columns
            to include.
        instruments:
            Which currency pairs to include. Typically all three:
            (EUR_USD, EUR_JPY, USD_JPY). Stored as a tuple (not list)
            so DatasetSpec is hashable.
        split_spec:
            Temporal split boundaries. Loaded from configs/training.yaml.
        forecast_horizon_bars:
            H — how many 1-minute bars ahead to predict direction for.
            H=5 means "will mid-price be higher or lower in 5 minutes?"
        context_length_bars:
            How many past bars the TFT model sees as input context.
            neuralforecast requires context_length >= forecast_horizon.
            Typical: 2-4x forecast_horizon_bars.
        processed_dir:
            Root of the processed Parquet tree as a string path.
            String (not Path) for trivial JSON/YAML serialisability.
    """

    level_group: LevelGroup
    instruments: tuple[ob.Instrument, ...]
    split_spec: SplitSpec
    forecast_horizon_bars: int
    context_length_bars: int
    processed_dir: str

    @property
    def processed_path(self) -> Path:
        """Return processed_dir as a Path object for filesystem operations."""
        return Path(self.processed_dir)
