# Transform raw EBS quote and deal rows into 1-minute bar DataFrames.

from __future__ import annotations

import logging
from collections.abc import Iterator

import polars as pl

from ebs_tft.data.parsers.ebs_csv import RawEBSDealRow, RawEBSQuoteRow
from ebs_tft.domain.orderbook import _models as models

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
    bars = bars.with_columns(pl.lit(instrument.value).alias(m.COL_INSTRUMENT))

    logger.debug(
        "Bars built",
        extra={"instrument": instrument.value, "n_bars": len(bars)},
    )
    return bars


# Helper functions


def _quotes_to_polars(*, quotes: Iterator[RawEBSQuoteRow]) -> pl.DataFrame:
    """ """


def _deals_to_polars(*, deals: Iterator[RawEBSDealRow]) -> pl.DataFrame:
    """ """


def _resample_quotes(*, df: pl.DataFrame) -> pl.DataFrame:
    """ """


def _compute_quote_features(*, df: pl.DataFrame) -> pl.DataFrame:
    """ """


def _compute_deal_features(*, deals_df: pl.DataFrame) -> pl.DataFrame:
    """ """


def _join_features(
    *,
    quote_bars: pl.DataFrame,
    deal_bars: pl.DataFrame,
) -> pl.DataFrame:
    """ """
