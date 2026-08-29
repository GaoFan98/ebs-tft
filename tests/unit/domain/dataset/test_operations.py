"""Test elapsed targets, partition boundaries, depth, and training-only transforms."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest import mock

import polars as pl

from ebs_tft.domain.dataset import models, operations
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.orderbook import queries as orderbook_queries

_INSTRUMENT = orderbook_models.Instrument.EUR_USD


class TestBuildDataset:
    def test_build_dataset_joins_exact_elapsed_time_inside_each_partition(self) -> None:
        bars = _bars()
        spec = _spec()

        with mock.patch.object(orderbook_queries, "load_bars", return_value=bars):
            actual = operations.build_dataset(spec=spec)

        assert [len(actual[split]) for split in models.Split] == [1, 1, 1]
        assert actual[models.Split.TRAIN][orderbook_models.COL_DIRECTION_TARGET][0] == 1
        assert (
            actual[models.Split.VALIDATION][orderbook_models.COL_DIRECTION_TARGET][0]
            == -1
        )
        assert "bid_price_l2" not in actual[models.Split.TRAIN].columns

    def test_build_dataset_excludes_flat_labels_only_under_preselected_policy(
        self,
    ) -> None:
        bars = _bars().with_columns(
            pl.when(
                pl.col(orderbook_models.COL_TRADING_DATE) == datetime.date(2024, 1, 1)
            )
            .then(pl.lit(1.0))
            .otherwise(pl.col(orderbook_models.COL_MID_PRICE))
            .alias(orderbook_models.COL_MID_PRICE)
        )
        spec = attrs_evolve_policy(
            spec=_spec(), policy=models.FlatTargetPolicy.EXCLUDE_EXACT_FLAT
        )

        with mock.patch.object(orderbook_queries, "load_bars", return_value=bars):
            actual = operations.build_dataset(spec=spec)

        assert actual[models.Split.TRAIN].is_empty()


class TestFitStandardization:
    def test_fit_standardization_ignores_evaluation_values(self) -> None:
        training = pl.DataFrame({"feature": [1.0, 3.0]})
        changed_evaluation = pl.DataFrame({"feature": [1000.0]})

        parameters = operations.fit_standardization(
            training_data=training, columns=("feature",)
        )
        actual = operations.apply_standardization(
            data=changed_evaluation, standardization=parameters
        )

        assert parameters.parameters[0].mean == 2.0
        assert actual["feature"][0] > 100


def attrs_evolve_policy(
    *, spec: models.DatasetSpec, policy: models.FlatTargetPolicy
) -> models.DatasetSpec:
    return models.DatasetSpec(
        depth=spec.depth,
        instruments=spec.instruments,
        split_spec=spec.split_spec,
        forecast_horizon=spec.forecast_horizon,
        context_length=spec.context_length,
        bar_frequency=spec.bar_frequency,
        flat_target_policy=policy,
        neutral_threshold=0,
        processed_dir=spec.processed_dir,
    )


def _spec() -> models.DatasetSpec:
    return models.DatasetSpec(
        depth=models.DepthSpec(maximum_level=1),
        instruments=(_INSTRUMENT,),
        split_spec=models.SplitSpec(
            train=models.DateRange(
                datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            ),
            validation=models.DateRange(
                datetime.date(2024, 1, 2), datetime.date(2024, 1, 2)
            ),
            test=models.DateRange(datetime.date(2024, 1, 3), datetime.date(2024, 1, 3)),
        ),
        forecast_horizon=datetime.timedelta(minutes=1),
        context_length=datetime.timedelta(minutes=2),
        bar_frequency=datetime.timedelta(minutes=1),
        flat_target_policy=models.FlatTargetPolicy.THREE_CLASS,
        neutral_threshold=0,
        processed_dir=Path("unused"),
    )


def _bars() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    prices = {
        datetime.date(2024, 1, 1): (1.0, 1.1, 1.2),
        datetime.date(2024, 1, 2): (1.1, 1.0, 0.9),
        datetime.date(2024, 1, 3): (1.0, 1.2, 1.3),
    }
    for trading_date, mids in prices.items():
        for minute, mid in zip((0, 1, 3), mids):
            rows.append(_bar(trading_date=trading_date, minute=minute, mid=mid))
    return pl.DataFrame(rows).sort(
        [orderbook_models.COL_INSTRUMENT, orderbook_models.COL_TIMESTAMP]
    )


def _bar(*, trading_date: datetime.date, minute: int, mid: float) -> dict[str, object]:
    timestamp = datetime.datetime.combine(
        trading_date, datetime.time(hour=12, minute=minute), tzinfo=datetime.UTC
    )
    return {
        orderbook_models.COL_TIMESTAMP: timestamp,
        orderbook_models.COL_INSTRUMENT: _INSTRUMENT.value,
        orderbook_models.COL_TRADING_DATE: trading_date,
        orderbook_models.bid_price_col(level=1): mid - 0.01,
        orderbook_models.ask_price_col(level=1): mid + 0.01,
        orderbook_models.bid_size_col(level=1): 1_000_000,
        orderbook_models.ask_size_col(level=1): 1_000_000,
        orderbook_models.bid_order_count_col(level=1): 1,
        orderbook_models.ask_order_count_col(level=1): 1,
        orderbook_models.COL_MID_PRICE: mid,
        orderbook_models.COL_SPREAD: 0.02,
        orderbook_models.COL_QUOTE_IMBALANCE: 0.0,
        orderbook_models.COL_BUY_VOLUME: 0,
        orderbook_models.COL_SELL_VOLUME: 0,
        orderbook_models.COL_TRADE_COUNT: 0,
        orderbook_models.COL_DEAL_FLOW_IMBALANCE: 0.0,
        orderbook_models.COL_EXTREMAL_PRICE_MEAN: None,
        orderbook_models.COL_QUOTE_AGE_SECONDS: 10.0,
        orderbook_models.COL_QUOTE_UPDATE_COUNT: 2,
        orderbook_models.COL_BOOK_OBSERVED: True,
        orderbook_models.COL_DEALS_OBSERVED: True,
    }
