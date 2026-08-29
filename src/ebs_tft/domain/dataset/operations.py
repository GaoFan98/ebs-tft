# Build model-ready train/val/test datasets from processed Parquet bars.

from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from ebs_tft.domain.dataset import models as dataset_models
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.orderbook import queries as ob_queries

logger = logging.getLogger(__name__)


def build_dataset(
    *, spec: dataset_models.DatasetSpec
) -> dict[dataset_models.Split, pd.DataFrame]:
    """
    Build train/validation/test DataFrames ready for neuralforecast.

    Runs the full dataset pipeline for the given DatasetSpec:
      1. Load processed bars for all instruments over the full split range
      2. Select only the feature columns for the requested level group
      3. Fill null deal columns (no-trade minutes)
      4. Forward-fill session-open book nulls, drop any unfillable leading rows
      5. Add direction_target: sign(mid_price[t+H] - mid_price[t])
      6. Drop the final H rows per instrument (no future price to target)
      7. Split by timestamp into train/val/test
      8. Rename to neuralforecast format (unique_id, ds, y)
      9. Convert each split to pandas

    :param spec: complete dataset specification
    :returns: dict mapping Split -> pandas DataFrame in neuralforecast format
    :raises ob_queries.NoBarsFoundError: if no processed Parquet files exist
    """
    logger.info(
        "Building dataset",
        extra={
            "level_group": spec.level_group.value,
            "instruments": [i.value for i in spec.instruments],
            "date_from": str(spec.split_spec.full_date_from),
            "date_to": str(spec.split_spec.full_date_to),
            "forecast_horizon_bars": spec.forecast_horizon_bars,
        },
    )

    bars = _load_bars(spec=spec)
    bars = _select_feature_columns(df=bars, level_group=spec.level_group)
    bars = _fill_deal_nulls(df=bars)
    bars = _fill_book_nulls(df=bars)
    bars = _add_direction_target(df=bars, horizon=spec.forecast_horizon_bars)
    bars = bars.drop_nulls(subset=[orderbook_models.COL_DIRECTION_TARGET])

    logger.info(
        "Dataset built",
        extra={
            "total_rows": len(bars),
            "instruments": bars[orderbook_models.COL_INSTRUMENT].unique().to_list(),
        },
    )

    return _split_and_convert(df=bars, spec=spec)


# Helper functions


def _load_bars(*, spec: dataset_models.DatasetSpec) -> pl.DataFrame:
    """
    Loads the full date range across all splits in one call so that
    forward-filling and direction target computation work correctly at
    split boundaries.
    """
    return ob_queries.load_multi_instrument_bars(
        processed_dir=spec.processed_path,
        level_group=spec.level_group.to_dir_name(),
        instruments=list(spec.instruments),
        date_from=spec.split_spec.full_date_from,
        date_to=spec.split_spec.full_date_to,
    )


def _select_feature_columns(
    *, df: pl.DataFrame, level_group: dataset_models.LevelGroup
) -> pl.DataFrame:
    """
    Select only the feature columns defined for this level group.
    """
    expected = level_group.feature_columns()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Processed bars are missing expected columns for {level_group.value!r}: "
            f"{missing}. Re-run the exporter to regenerate the Parquet files."
        )

    return df.select(expected)


def _fill_deal_nulls(*, df: pl.DataFrame) -> pl.DataFrame:
    """
    Fill null deal columns for minutes where no trades executed.

    When no deals occurred in a minute, the LEFT join leaves deal columns as
    null. This is correct for storage (null != 0) but must be resolved before
    model training:

    - buy_volume, sell_volume, trade_count:
        Filled with 0 — no trades means zero volume, zero count.
    - deal_flow_imbalance:
        Filled with 0 — no imbalance can be computed without trades.
    - vwap:
        Forward-filled per instrument — the last valid VWAP price persists
        as the "last known execution price". Filling with 0 would be wrong
        (VWAP is a price level, not a count). Any remaining leading nulls
        (start of data before first trade) are filled with mid_price.
    """
    fill_with_zero = [
        c
        for c in [
            orderbook_models.COL_BUY_VOLUME,
            orderbook_models.COL_SELL_VOLUME,
            orderbook_models.COL_TRADE_COUNT,
            orderbook_models.COL_DEAL_FLOW_IMBALANCE,
        ]
        if c in df.columns
    ]

    if fill_with_zero:
        df = df.with_columns([pl.col(c).fill_null(0) for c in fill_with_zero])

    if orderbook_models.COL_VWAP in df.columns:
        df = df.with_columns(
            pl.col(orderbook_models.COL_VWAP)
            .forward_fill()
            .over(orderbook_models.COL_INSTRUMENT)
            .alias(orderbook_models.COL_VWAP)
        )
        if orderbook_models.COL_MID_PRICE in df.columns:
            df = df.with_columns(
                pl.col(orderbook_models.COL_VWAP)
                .fill_null(pl.col(orderbook_models.COL_MID_PRICE))
                .alias(orderbook_models.COL_VWAP)
            )

    return df


def _fill_book_nulls(*, df: pl.DataFrame) -> pl.DataFrame:
    """
    Forward-fill session-open book nulls, then drop unfillable leading rows.

    At the start of each trading session (~22:00 Tokyo time) the order book
    initialises over several minutes. During this period bid/ask prices and
    sizes, mid_price, spread, and quote_imbalance are null.

    Strategy:
      1. Forward-fill each book column per instrument — later valid data
         fills backward into the leading nulls... wait, that is backward fill.
         Actually: we sort ascending (already done by queries), so forward-fill
         will NOT fill leading nulls (there is nothing before them to fill from).
      2. Drop rows that still have null mid_price after the fill — these are
         the true session-open initialisation minutes that cannot be imputed.
         Dropping ~8–15 rows per instrument per day is negligible (<1% of data).

    This is safer than backward-filling: backward-fill would impute the
    session-open period with data from the future of that session, which is
    a form of look-ahead bias.
    """
    book_cols = [
        c
        for c in df.columns
        if c not in (orderbook_models.COL_TIMESTAMP, orderbook_models.COL_INSTRUMENT)
        and c.startswith(("bid_", "ask_", "mid_", "spread", "quote_"))
    ]

    if book_cols:
        df = df.with_columns(
            [
                pl.col(c).forward_fill().over(orderbook_models.COL_INSTRUMENT)
                for c in book_cols
            ]
        )

    if orderbook_models.COL_MID_PRICE in df.columns:
        before = len(df)
        df = df.drop_nulls(subset=[orderbook_models.COL_MID_PRICE])
        dropped = before - len(df)
        if dropped > 0:
            logger.debug(
                "Dropped leading session-open null rows",
                extra={"rows_dropped": dropped},
            )

    return df


def _add_direction_target(*, df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Add the direction_target column: sign(mid_price[t+H] - mid_price[t]).

    Values:
        +1.0 — price is higher H minutes later (upward move)
        -1.0 — price is lower H minutes later (downward move)
         0.0 — price is identical (flat, rare in FX)
        null — the last H rows per instrument (no future price available)

    :param df: bars DataFrame, must contain COL_MID_PRICE and COL_INSTRUMENT
    :param horizon: H — how many bars ahead to look for the future price
    """
    future_mid = (
        pl.col(orderbook_models.COL_MID_PRICE)
        .shift(-horizon)
        .over(orderbook_models.COL_INSTRUMENT)
    )
    raw_direction = future_mid - pl.col(orderbook_models.COL_MID_PRICE)

    return df.with_columns(
        raw_direction.sign().alias(orderbook_models.COL_DIRECTION_TARGET)
    )


def _split_and_convert(
    *, df: pl.DataFrame, spec: dataset_models.DatasetSpec
) -> dict[dataset_models.Split, pd.DataFrame]:
    """
    Filter the full DataFrame into train/val/test splits and convert to pandas.
    """
    result: dict[dataset_models.Split, pd.DataFrame] = {}

    for split in dataset_models.Split:
        date_from = spec.split_spec.date_from_for(split=split)
        date_to = spec.split_spec.date_to_for(split=split)

        split_df = df.filter(
            pl.col(orderbook_models.COL_TIMESTAMP)
            .dt.date()
            .is_between(date_from, date_to)
        )

        if split_df.is_empty():
            logger.warning(
                "Split produced empty DataFrame",
                extra={
                    "split": split.value,
                    "date_from": str(date_from),
                    "date_to": str(date_to),
                },
            )

        split_df = split_df.rename(
            {
                orderbook_models.COL_INSTRUMENT: "unique_id",
                orderbook_models.COL_TIMESTAMP: "ds",
                orderbook_models.COL_DIRECTION_TARGET: "y",
            }
        )

        result[split] = split_df.to_pandas()

        logger.debug(
            "Split ready",
            extra={
                "split": split.value,
                "rows": len(result[split]),
                "date_from": str(date_from),
                "date_to": str(date_to),
            },
        )

    return result
