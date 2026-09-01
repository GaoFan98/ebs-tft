"""Test immutable cumulative-depth dataset specifications."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from ebs_tft.domain.dataset import models
from ebs_tft.domain.orderbook import models as orderbook_models


class TestDepthSpec:
    def test_feature_columns_include_exactly_levels_one_through_k(self) -> None:
        spec = models.DepthSpec(maximum_level=2)

        actual = spec.feature_columns()

        assert orderbook_models.bid_price_col(level=1) in actual
        assert orderbook_models.bid_price_col(level=2) in actual
        assert "bid_price_l3" not in actual

    def test_depth_spec_rejects_an_unsupported_depth(self) -> None:
        with pytest.raises(ValueError, match="max_level"):
            models.DepthSpec(maximum_level=0)


class TestSplitSpec:
    def test_split_spec_rejects_overlapping_partitions(self) -> None:
        shared = datetime.date(2024, 1, 2)

        with pytest.raises(ValueError, match="disjoint"):
            models.SplitSpec(
                train=models.DateRange(datetime.date(2024, 1, 1), shared),
                validation=models.DateRange(shared, datetime.date(2024, 1, 3)),
                test=models.DateRange(
                    datetime.date(2024, 1, 4), datetime.date(2024, 1, 5)
                ),
            )


class TestDatasetSpec:
    def test_dataset_spec_rejects_a_row_like_non_multiple_horizon(self) -> None:
        split_spec = _split_spec()

        with pytest.raises(ValueError, match="align"):
            models.DatasetSpec(
                depth=models.DepthSpec(maximum_level=1),
                instruments=(orderbook_models.Instrument.EUR_USD,),
                split_spec=split_spec,
                forecast_horizon=datetime.timedelta(milliseconds=150),
                context_length=datetime.timedelta(minutes=5),
                state_interval=datetime.timedelta(milliseconds=100),
                flat_target_policy=models.FlatTargetPolicy.THREE_CLASS,
                neutral_threshold=0,
                processed_dir=Path("processed"),
            )


def _split_spec() -> models.SplitSpec:
    return models.SplitSpec(
        train=models.DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)),
        validation=models.DateRange(
            datetime.date(2024, 1, 2), datetime.date(2024, 1, 2)
        ),
        test=models.DateRange(datetime.date(2024, 1, 3), datetime.date(2024, 1, 3)),
    )
