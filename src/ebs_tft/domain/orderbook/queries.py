"""Query canonical order-book partitions with strict batched scans."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ebs_tft.data.repositories import processed
from ebs_tft.domain.orderbook import models


class NoBarsFoundError(Exception):
    """Indicate that a required canonical range has no published partitions."""


class InvalidBarsError(Exception):
    """Indicate schema drift, corruption, duplicates, or invalid ordering."""


def load_bars(
    *,
    processed_dir: Path,
    instruments: Sequence[models.Instrument],
    maximum_depth: int,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    required: bool = True,
    required_trading_dates: Sequence[datetime.date] | None = None,
) -> pl.DataFrame:
    """Discover once and scan all selected canonical Parquet files as one query."""
    selected = list(
        processed.find_partitions(
            processed_dir=processed_dir,
            instruments=instruments,
            date_from=date_from,
            date_to=date_to,
        )
    )
    if not selected:
        if required:
            raise NoBarsFoundError(
                "No canonical bars found for the requested instruments and dates"
            )
        return pl.DataFrame()
    if required_trading_dates is not None:
        expected = {
            (instrument, trading_date)
            for instrument in instruments
            for trading_date in required_trading_dates
        }
        actual = {
            (item.metadata.instrument, item.metadata.trading_date) for item in selected
        }
        missing_partitions = sorted(
            expected - actual, key=lambda item: (item[0].value, item[1])
        )
        if missing_partitions:
            missing_labels = [
                (item[0].value, item[1].isoformat()) for item in missing_partitions
            ]
            raise NoBarsFoundError(
                f"Missing required canonical partitions: {missing_labels}"
            )
    schema_versions = {item.metadata.schema_version for item in selected}
    stored_depths = {item.metadata.maximum_depth for item in selected}
    schemas = {item.metadata.schema for item in selected}
    if (
        schema_versions != {processed.SCHEMA_VERSION}
        or len(stored_depths) != 1
        or len(schemas) != 1
    ):
        raise InvalidBarsError("selected manifests have incompatible schemas")
    stored_depth = next(iter(stored_depths))
    if not 1 <= maximum_depth <= stored_depth:
        raise InvalidBarsError(
            f"requested depth {maximum_depth} is outside stored depth {stored_depth}"
        )
    try:
        for partition in selected:
            processed.verify_partition(partition=partition)
    except processed.UnableToReadBarsError as exc:
        raise InvalidBarsError(
            "Canonical partition checksum validation failed"
        ) from exc

    columns = _columns(maximum_depth=maximum_depth)
    try:
        data = (
            pl.scan_parquet([item.data_path for item in selected])
            .select(columns)
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise InvalidBarsError("Unable to scan canonical bar partitions") from exc
    keys = [models.COL_INSTRUMENT, models.COL_TIMESTAMP]
    if data.select(keys).n_unique() != len(data):
        raise InvalidBarsError("canonical query contains duplicate keys")
    ordered = data.sort(keys)
    if not data.equals(ordered):
        data = ordered
    return data


def available_date_range(
    *, processed_dir: Path, instrument: models.Instrument
) -> tuple[datetime.date, datetime.date] | None:
    """Return the inclusive trading-date coverage from one discovery pass."""
    partitions = list(
        processed.find_partitions(
            processed_dir=processed_dir,
            instruments=(instrument,),
        )
    )
    if not partitions:
        return None
    dates = [item.metadata.trading_date for item in partitions]
    return min(dates), max(dates)


def _columns(*, maximum_depth: int) -> list[str]:
    return models.canonical_bar_columns(max_level=maximum_depth)
