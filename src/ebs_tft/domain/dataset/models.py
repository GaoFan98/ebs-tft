"""Define immutable, model-neutral dataset specifications."""

from __future__ import annotations

import datetime
import enum
from pathlib import Path

import attrs

from ebs_tft.domain.orderbook import models as orderbook_models


class Split(enum.StrEnum):
    """Identify chronological model-development partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SampleRole(enum.StrEnum):
    """Distinguish model development from later external validation samples."""

    DEVELOPMENT = "development"
    EXTERNAL_VALIDATION = "external_validation"


class FlatTargetPolicy(enum.StrEnum):
    """Define an analysis choice made before evaluation results are inspected."""

    THREE_CLASS = "three_class"
    EXCLUDE_EXACT_FLAT = "exclude_exact_flat"
    EXCLUDE_NEUTRAL_BAND = "exclude_neutral_band"


@attrs.frozen
class DepthSpec:
    """Represent inclusive cumulative levels 1 through maximum_level."""

    maximum_level: int

    def __attrs_post_init__(self) -> None:
        orderbook_models.all_level_cols(max_level=self.maximum_level)

    def feature_columns(self) -> tuple[str, ...]:
        """Return base features plus exactly the configured cumulative depth."""
        return tuple(
            orderbook_models.canonical_bar_columns(max_level=self.maximum_level)
        )


@attrs.frozen
class DateRange:
    """Represent one inclusive trading-date range."""

    date_from: datetime.date
    date_to: datetime.date

    def __attrs_post_init__(self) -> None:
        if self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")

    def contains(self, *, date: datetime.date) -> bool:
        """Return whether date lies in this inclusive range."""
        return self.date_from <= date <= self.date_to


@attrs.frozen
class SplitSpec:
    """Define ordered, non-overlapping trading-date development partitions."""

    train: DateRange
    validation: DateRange
    test: DateRange

    def __attrs_post_init__(self) -> None:
        if not self.train.date_to < self.validation.date_from:
            raise ValueError("train and validation ranges must be ordered and disjoint")
        if not self.validation.date_to < self.test.date_from:
            raise ValueError("validation and test ranges must be ordered and disjoint")

    @property
    def full_date_from(self) -> datetime.date:
        return self.train.date_from

    @property
    def full_date_to(self) -> datetime.date:
        return self.test.date_to

    def range_for(self, *, split: Split) -> DateRange:
        """Return the date range assigned to split."""
        if split is Split.TRAIN:
            return self.train
        if split is Split.VALIDATION:
            return self.validation
        return self.test


@attrs.frozen
class DatasetSpec:
    """Define one leakage-safe cumulative-depth dataset build."""

    depth: DepthSpec
    instruments: tuple[orderbook_models.Instrument, ...]
    split_spec: SplitSpec
    forecast_horizon: datetime.timedelta
    context_length: datetime.timedelta
    state_interval: datetime.timedelta
    flat_target_policy: FlatTargetPolicy
    neutral_threshold: float
    processed_dir: Path
    sample_role: SampleRole = SampleRole.DEVELOPMENT

    def __attrs_post_init__(self) -> None:
        if not self.instruments or len(set(self.instruments)) != len(self.instruments):
            raise ValueError("instruments must be non-empty and unique")
        if self.forecast_horizon <= datetime.timedelta(0):
            raise ValueError("forecast_horizon must be positive")
        if self.context_length <= datetime.timedelta(0):
            raise ValueError("context_length must be positive")
        if self.state_interval <= datetime.timedelta(0):
            raise ValueError("state_interval must be positive")
        if self.forecast_horizon % self.state_interval:
            raise ValueError("forecast_horizon must align to state_interval")
        if self.context_length % self.state_interval:
            raise ValueError("context_length must align to state_interval")
        if self.neutral_threshold < 0:
            raise ValueError("neutral_threshold must be non-negative")
        if (
            self.flat_target_policy is not FlatTargetPolicy.EXCLUDE_NEUTRAL_BAND
            and self.neutral_threshold != 0
        ):
            raise ValueError(
                "neutral_threshold is only valid for a neutral-band policy"
            )
