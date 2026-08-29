"""Build leakage-safe, model-neutral cumulative-depth datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import attrs
import polars as pl

from ebs_tft.domain.dataset import models as dataset_models
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.orderbook import queries as orderbook_queries

_COL_SPLIT = "split"
_COL_FUTURE_MID = "future_mid_price"


class InvalidDatasetError(Exception):
    """Indicate invalid canonical input or a leakage-prone dataset request."""


@attrs.frozen
class ScaleParameter:
    """Store one training-only standardization parameter pair."""

    column: str
    mean: float
    standard_deviation: float


@attrs.frozen
class Standardization:
    """Store immutable transforms fitted only on the training partition."""

    parameters: tuple[ScaleParameter, ...]


def build_dataset(
    *, spec: dataset_models.DatasetSpec
) -> dict[dataset_models.Split, pl.DataFrame]:
    """Build exact-elapsed-horizon partitions without crossing any boundary."""
    bars = orderbook_queries.load_bars(
        processed_dir=spec.processed_dir,
        instruments=spec.instruments,
        maximum_depth=spec.depth.maximum_level,
        date_from=spec.split_spec.full_date_from,
        date_to=spec.split_spec.full_date_to,
    )
    missing = sorted(set(spec.depth.feature_columns()) - set(bars.columns))
    if missing:
        raise InvalidDatasetError(f"canonical bars are missing columns: {missing}")
    bars = bars.select(spec.depth.feature_columns())
    bars = _assign_splits(data=bars, split_spec=spec.split_spec)
    result: dict[dataset_models.Split, pl.DataFrame] = {}
    for split in dataset_models.Split:
        partition = bars.filter(pl.col(_COL_SPLIT) == split.value)
        partition = _add_target_with_exact_join(data=partition, spec=spec)
        result[split] = partition.drop(_COL_SPLIT).sort(
            [orderbook_models.COL_INSTRUMENT, orderbook_models.COL_TIMESTAMP]
        )
    return result


def fit_standardization(
    *, training_data: pl.DataFrame, columns: tuple[str, ...]
) -> Standardization:
    """Fit immutable scaling values from training data only."""
    missing = sorted(set(columns) - set(training_data.columns))
    if missing:
        raise InvalidDatasetError(f"training data is missing scale columns: {missing}")
    parameters: list[ScaleParameter] = []
    for column in columns:
        raw_mean = training_data[column].mean()
        raw_standard_deviation = training_data[column].std()
        if raw_mean is None or raw_standard_deviation is None:
            raise InvalidDatasetError(f"cannot standardize column {column!r}")
        mean = cast(float, raw_mean)
        standard_deviation = cast(float, raw_standard_deviation)
        if standard_deviation <= 0:
            raise InvalidDatasetError(f"cannot standardize column {column!r}")
        parameters.append(
            ScaleParameter(
                column=column,
                mean=mean,
                standard_deviation=standard_deviation,
            )
        )
    return Standardization(parameters=tuple(parameters))


def apply_standardization(
    *, data: pl.DataFrame, standardization: Standardization
) -> pl.DataFrame:
    """Apply previously fitted parameters without inspecting evaluation values."""
    return data.with_columns(
        [
            ((pl.col(item.column) - item.mean) / item.standard_deviation).alias(
                item.column
            )
            for item in standardization.parameters
        ]
    )


def _assign_splits(
    *, data: pl.DataFrame, split_spec: dataset_models.SplitSpec
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for split in dataset_models.Split:
        date_range = split_spec.range_for(split=split)
        expressions.append(
            pl.when(
                pl.col(orderbook_models.COL_TRADING_DATE).is_between(
                    date_range.date_from, date_range.date_to
                )
            ).then(pl.lit(split.value))
        )
    split_expression = expressions[0]
    for expression in expressions[1:]:
        split_expression = split_expression.fill_null(expression)
    return data.with_columns(split_expression.alias(_COL_SPLIT)).filter(
        pl.col(_COL_SPLIT).is_not_null()
    )


def _add_target_with_exact_join(
    *, data: pl.DataFrame, spec: dataset_models.DatasetSpec
) -> pl.DataFrame:
    duration = f"{int(spec.forecast_horizon.total_seconds())}s"
    join_keys = [
        orderbook_models.COL_INSTRUMENT,
        orderbook_models.COL_TRADING_DATE,
        _COL_SPLIT,
        orderbook_models.COL_TIMESTAMP,
    ]
    future = (
        data.select(
            [
                orderbook_models.COL_INSTRUMENT,
                orderbook_models.COL_TRADING_DATE,
                _COL_SPLIT,
                orderbook_models.COL_TIMESTAMP,
                orderbook_models.COL_MID_PRICE,
                orderbook_models.COL_BOOK_OBSERVED,
            ]
        )
        .with_columns(
            pl.col(orderbook_models.COL_TIMESTAMP).dt.offset_by(f"-{duration}"),
            pl.col(orderbook_models.COL_MID_PRICE).alias(_COL_FUTURE_MID),
            pl.col(orderbook_models.COL_BOOK_OBSERVED).alias("_future_book_observed"),
        )
        .drop([orderbook_models.COL_MID_PRICE, orderbook_models.COL_BOOK_OBSERVED])
    )
    joined = data.join(future, on=join_keys, how="left", validate="1:1")
    valid = (
        pl.col(orderbook_models.COL_BOOK_OBSERVED)
        & pl.col("_future_book_observed").fill_null(False)
        & pl.col(orderbook_models.COL_MID_PRICE).is_not_null()
        & pl.col(_COL_FUTURE_MID).is_not_null()
    )
    delta = pl.col(_COL_FUTURE_MID) - pl.col(orderbook_models.COL_MID_PRICE)
    joined = joined.filter(valid)
    if spec.flat_target_policy is dataset_models.FlatTargetPolicy.EXCLUDE_EXACT_FLAT:
        joined = joined.filter(delta != 0)
    elif (
        spec.flat_target_policy is dataset_models.FlatTargetPolicy.EXCLUDE_NEUTRAL_BAND
    ):
        joined = joined.filter(delta.abs() > spec.neutral_threshold)
    return joined.with_columns(
        delta.sign().cast(pl.Int8).alias(orderbook_models.COL_DIRECTION_TARGET)
    ).drop([_COL_FUTURE_MID, "_future_book_observed"])


def standardization_columns(*, data: pl.DataFrame) -> tuple[str, ...]:
    """Return numeric feature columns while excluding labels and identifiers."""
    excluded = {
        orderbook_models.COL_TIMESTAMP,
        orderbook_models.COL_TRADING_DATE,
        orderbook_models.COL_INSTRUMENT,
        orderbook_models.COL_DIRECTION_TARGET,
        orderbook_models.COL_BOOK_OBSERVED,
        orderbook_models.COL_DEALS_OBSERVED,
    }
    numeric = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    schema: Mapping[str, pl.DataType] = data.schema
    return tuple(
        column
        for column in data.columns
        if column not in excluded and schema[column] in numeric
    )
