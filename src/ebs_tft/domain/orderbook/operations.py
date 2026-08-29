# Transform raw EBS quote and deal rows into 1-minute bar DataFrames.

from __future__ import annotations

import logging
from collections.abc import Iterator

import polars as pl

from ebs_tft.data.parsers.ebs_csv import RawEBSDealRow, RawEBSQuoteRow
from ebs_tft.domain.orderbook import models

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMAT: str = "%Y/%m/%d %H:%M:%S%.3f"


def build_bars(
    *,
    quotes: Iterator[RawEBSQuoteRow],
    deals: Iterator[RawEBSDealRow],
    instrument: models.Instrument,
) -> pl.DataFrame:
    """
    Transform raw EBS rows for one trading day into 1-minute bar Dataframe.

    Each row in the returned Dataframe is one 1-minute bar and contains:
         - Timestamp (e.g. 2024-01-02 22:01:00)
         - Instrument identifier
         - Mid-price, spread
         - Per-level bid/ask prices and sizes (up to MAX_LEVELS = 10)
         - Quote-side order imbalance
         - Deal-side order flow features (buy/sell volume,flow imbalance)

       Deal feature columns are null for minutes with no trades

       :param quotes: iterator from ebs_csv.parse_quotes()
       :param deals:  iterator from ebs_csv.parse_deals()
       :param instrument: validated Instrument enum value for this file
       :returns: polars DataFrame sorted by timestamp ascending
    """
    logger.debug("Building bars", extra={"instrument": instrument.value})

    quotes_df = _quotes_to_polars(quotes=quotes)
    deals_df = _deals_to_polars(deals=deals)

    if quotes_df.is_empty():
        logger.warning(
            "No valid quote rows — returning empty DataFrame",
            extra={"instrument": instrument.value},
        )
        return pl.DataFrame()

    quote_bars = _resample_quotes(df=quotes_df)
    quote_bars = _compute_quote_features(df=quote_bars)
    deal_bars = _compute_deal_features(deals_df=deals_df)
    bars = _join_features(quote_bars=quote_bars, deal_bars=deal_bars)

    # Add instrument as a column: needed when concatenating multiple instruments
    # for multi-series training.
    bars = bars.with_columns(pl.lit(instrument.value).alias(models.COL_INSTRUMENT))

    logger.debug(
        "Bars built",
        extra={"instrument": instrument.value, "n_bars": len(bars)},
    )
    return bars


# Helper functions


def _quotes_to_polars(*, quotes: Iterator[RawEBSQuoteRow]) -> pl.DataFrame:
    """
    Consume a quote row iterator and return a polars DataFrame.

    Drops:
      - Initialization rows: price is None (session open, no price set yet)
      - Zero-size rows: size == 0 carries no liquidity information
    """
    timestamp_strs: list[str] = []
    sides: list[int] = []
    levels: list[int] = []
    prices: list[float] = []
    sizes: list[int] = []

    for row in quotes:
        if row.price is None or row.size == 0:
            continue
        timestamp_strs.append(f"{row.date} {row.time}")
        sides.append(row.side)
        levels.append(row.level)
        prices.append(row.price)
        sizes.append(row.size)

    if not timestamp_strs:
        return pl.DataFrame()

    df = pl.DataFrame(
        {
            "timestamp_str": timestamp_strs,
            "side": pl.Series(sides, dtype=pl.Int8),
            "level": pl.Series(levels, dtype=pl.Int8),
            "price": pl.Series(prices, dtype=pl.Float64),
            "size": pl.Series(sizes, dtype=pl.Int64),
        }
    )

    # Parse datetime and truncate to minute for grouping.
    return (
        df.with_columns(
            pl.col("timestamp_str")
            .str.to_datetime(format=_TIMESTAMP_FORMAT)
            .alias(models.COL_TIMESTAMP)
        )
        .with_columns(
            pl.col(models.COL_TIMESTAMP).dt.truncate("1m").alias("minute_bucket")
        )
        .drop("timestamp_str")
    )


def _deals_to_polars(*, deals: Iterator[RawEBSDealRow]) -> pl.DataFrame:
    """
    Consume a deal row iterator and return a polars DataFrame.
    Drops rows where deal_price is None (no trade occurred in that 100ms slice —
     EBS format emits D records even in quiet periods with an empty price field).
    """
    timestamp_strs: list[str] = []
    sides: list[int] = []
    deal_prices: list[float] = []
    deal_sizes: list[int] = []
    deal_counts: list[int] = []

    for row in deals:
        if row.deal_price is None:
            continue
        timestamp_strs.append(f"{row.date} {row.time}")
        sides.append(row.side)
        deal_prices.append(row.deal_price)
        deal_sizes.append(row.deal_size)
        deal_counts.append(row.deal_count)

    if not timestamp_strs:
        return pl.DataFrame()

    df = pl.DataFrame(
        {
            "timestamp_str": timestamp_strs,
            "side": pl.Series(sides, dtype=pl.Int8),
            "deal_price": pl.Series(deal_prices, dtype=pl.Float64),
            "deal_size": pl.Series(deal_sizes, dtype=pl.Int64),
            "deal_count": pl.Series(deal_counts, dtype=pl.Int32),
        }
    )

    return (
        df.with_columns(
            pl.col("timestamp_str")
            .str.to_datetime(format=_TIMESTAMP_FORMAT)
            .alias(models.COL_TIMESTAMP)
        )
        .with_columns(
            pl.col(models.COL_TIMESTAMP).dt.truncate("1m").alias("minute_bucket")
        )
        .drop("timestamp_str")
    )


def _resample_quotes(*, df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate 100ms quote snapshots into 1-minute wide-format bars.

    Pipeline:
      1. Split into bid (side=0) and ask (side=1)
      2. Group by (minute_bucket, level): mean price, mean size
      3. Pivot: spread level values
      4. Rename to convention: bid_price_l1, ask_size_l3, etc.
      5. Full-outer join bid columns + ask columns on minute_bucket

    Pivot creates column names like "price_1", "size_10".
    """
    # Separate bid and ask sides
    bid_df = df.filter(pl.col("side") == 0)
    ask_df = df.filter(pl.col("side") == 1)
    # Aggregate per (minute_bucket, level)
    bid_agg = bid_df.group_by(["minute_bucket", "level"]).agg(
        [
            pl.mean("price").alias("price"),
            pl.mean("size").alias("size"),
        ]
    )
    ask_agg = ask_df.group_by(["minute_bucket", "level"]).agg(
        [
            pl.mean("price").alias("price"),
            pl.mean("size").alias("size"),
        ]
    )
    # Result columns: minute_bucket | price_1 | price_2 | ... | size_1 | ...
    bid_wide = bid_agg.pivot(
        on="level",
        index="minute_bucket",
        values=["price", "size"],
        aggregate_function="mean",
        sort_columns=True,
    )
    ask_wide = ask_agg.pivot(
        on="level",
        index="minute_bucket",
        values=["price", "size"],
        aggregate_function="mean",
        sort_columns=True,
    )
    # Rename columns to match naming convention
    bid_rename = {
        f"price_{n}": models.bid_price_col(level=n)
        for n in range(1, models.MAX_LEVELS + 1)
    } | {
        f"size_{n}": models.bid_size_col(level=n)
        for n in range(1, models.MAX_LEVELS + 1)
    }
    ask_rename = {
        f"price_{n}": models.ask_price_col(level=n)
        for n in range(1, models.MAX_LEVELS + 1)
    } | {
        f"size_{n}": models.ask_size_col(level=n)
        for n in range(1, models.MAX_LEVELS + 1)
    }
    bid_wide = bid_wide.rename(bid_rename)
    ask_wide = ask_wide.rename(ask_rename)

    # join bid + ask into one row per minute
    bars = bid_wide.join(ask_wide, on="minute_bucket", how="full", coalesce=True)
    return bars.sort("minute_bucket")


def _compute_quote_features(*, df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived features from L1 bid/ask prices and sizes.

    Adds: mid_price, spread, quote_imbalance.
    All three are computed from Level 1 only — best bid and best ask.
    """
    b1 = models.bid_price_col(level=1)
    a1 = models.ask_price_col(level=1)
    bs1 = models.bid_size_col(level=1)
    as1 = models.ask_size_col(level=1)

    return df.with_columns(
        [
            # Mid-price: reference price for direction target labelling
            ((pl.col(b1) + pl.col(a1)) / 2.0).alias(models.COL_MID_PRICE),
            # Spread: market transaction cost. Wider spread = worse liquidity.
            (pl.col(a1) - pl.col(b1)).alias(models.COL_SPREAD),
            # Quote imbalance from L1 sizes.
            # > 0: more bid liquidity -> passive sellers dominate -> bullish lean
            # < 0: more ask liquidity -> passive buyers dominate -> bearish lean
            (
                (pl.col(bs1) - pl.col(as1)).cast(pl.Float64)
                / (pl.col(bs1) + pl.col(as1)).cast(pl.Float64)
            ).alias(models.COL_QUOTE_IMBALANCE),
        ]
    )


def _compute_deal_features(*, deals_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate deal (D record) rows into per-minute order flow features.
    Returns a DataFrame with columns:
        minute_bucket, buy_volume, sell_volume, deal_flow_imbalance,
        trade_count, vwap
    If no deals were provided (holiday or very quiet day), returns an empty
    DataFrame.
    """
    if deals_df.is_empty():
        return pl.DataFrame()
    # Aggregate buy-initiated deals (side=1) per minute
    buy_agg = (
        deals_df.filter(pl.col("side") == 1)
        .group_by("minute_bucket")
        .agg(
            [
                pl.sum("deal_size").alias(models.COL_BUY_VOLUME),
                pl.sum("deal_count").alias("_buy_count"),
                (pl.col("deal_price") * pl.col("deal_size")).sum().alias("_buy_pxs"),
            ]
        )
    )
    # Aggregate sell-initiated deals (side=0) per minute
    sell_agg = (
        deals_df.filter(pl.col("side") == 0)
        .group_by("minute_bucket")
        .agg(
            [
                pl.sum("deal_size").alias(models.COL_SELL_VOLUME),
                pl.sum("deal_count").alias("_sell_count"),
                (pl.col("deal_price") * pl.col("deal_size")).sum().alias("_sell_pxs"),
            ]
        )
    )
    # Full join: minutes with only buys or only sells are preserved
    deal_bars = buy_agg.join(sell_agg, on="minute_bucket", how="full", coalesce=True)
    # Fill missing side with 0 (if a minute had only buys, sell_volume = 0)
    deal_bars = deal_bars.with_columns(
        [
            pl.col(models.COL_BUY_VOLUME).fill_null(0),
            pl.col(models.COL_SELL_VOLUME).fill_null(0),
            pl.col("_buy_count").fill_null(0),
            pl.col("_sell_count").fill_null(0),
            pl.col("_buy_pxs").fill_null(0.0),
            pl.col("_sell_pxs").fill_null(0.0),
        ]
    )
    deal_bars = deal_bars.with_columns(
        [
            (pl.col("_buy_count") + pl.col("_sell_count")).alias(
                models.COL_TRADE_COUNT
            ),
            # Flow imbalance: (+1) = pure aggressive buying,
            #  (-1) = pure aggressive selling
            # null when no deals at all (denominator = 0 → polars returns null)
            (
                (pl.col(models.COL_BUY_VOLUME) - pl.col(models.COL_SELL_VOLUME)).cast(
                    pl.Float64
                )
                / (pl.col(models.COL_BUY_VOLUME) + pl.col(models.COL_SELL_VOLUME)).cast(
                    pl.Float64
                )
            ).alias(models.COL_DEAL_FLOW_IMBALANCE),
            # VWAP across both sides: total(price × size) / total(size)
            (
                (pl.col("_buy_pxs") + pl.col("_sell_pxs"))
                / (pl.col(models.COL_BUY_VOLUME) + pl.col(models.COL_SELL_VOLUME)).cast(
                    pl.Float64
                )
            ).alias(models.COL_VWAP),
        ]
    ).drop(["_buy_count", "_sell_count", "_buy_pxs", "_sell_pxs"])
    return deal_bars.sort("minute_bucket")


def _join_features(
    *,
    quote_bars: pl.DataFrame,
    deal_bars: pl.DataFrame,
) -> pl.DataFrame:
    """
    Left-join quote bars with deal features on minute_bucket.
    """
    if deal_bars.is_empty():
        # Quiet day: add null deal columns so schema is consistent
        return quote_bars.with_columns(
            [
                pl.lit(None).cast(pl.Int64).alias(models.COL_BUY_VOLUME),
                pl.lit(None).cast(pl.Int64).alias(models.COL_SELL_VOLUME),
                pl.lit(None).cast(pl.Float64).alias(models.COL_DEAL_FLOW_IMBALANCE),
                pl.lit(None).cast(pl.Int32).alias(models.COL_TRADE_COUNT),
                pl.lit(None).cast(pl.Float64).alias(models.COL_VWAP),
            ]
        ).rename({"minute_bucket": models.COL_TIMESTAMP})
    return quote_bars.join(deal_bars, on="minute_bucket", how="left").rename(
        {"minute_bucket": models.COL_TIMESTAMP}
    )
