"""Test strict batched queries over canonical manifests."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

from ebs_tft.data.repositories import processed
from ebs_tft.domain.orderbook import models, operations, queries

_INSTRUMENT = models.Instrument.EUR_USD


class TestLoadBars:
    def test_load_bars_discovers_once_and_returns_strict_sorted_depth_view(
        self, tmp_path: Path
    ) -> None:
        later = datetime.date(2024, 1, 3)
        earlier = datetime.date(2024, 1, 2)
        _publish(processed_dir=tmp_path, trading_date=later)
        _publish(processed_dir=tmp_path, trading_date=earlier)

        actual = queries.load_bars(
            processed_dir=tmp_path,
            instruments=(_INSTRUMENT,),
            maximum_depth=1,
        )

        assert actual.height == 2
        assert actual[models.COL_TRADING_DATE].to_list() == [earlier, later]
        assert "bid_price_l2" not in actual.columns

    def test_load_bars_uses_one_discovery_and_one_batch_scan(
        self, tmp_path: Path
    ) -> None:
        _publish(processed_dir=tmp_path, trading_date=datetime.date(2024, 1, 2))
        _publish(processed_dir=tmp_path, trading_date=datetime.date(2024, 1, 3))

        with (
            mock.patch.object(
                processed, "find_partitions", wraps=processed.find_partitions
            ) as discover,
            mock.patch.object(pl, "scan_parquet", wraps=pl.scan_parquet) as scan,
        ):
            queries.load_bars(
                processed_dir=tmp_path,
                instruments=(_INSTRUMENT,),
                maximum_depth=1,
            )

        discover.assert_called_once()
        scan.assert_called_once()

    def test_load_bars_rejects_schema_drift_between_manifests(
        self, tmp_path: Path
    ) -> None:
        earlier = datetime.date(2024, 1, 2)
        later = datetime.date(2024, 1, 3)
        _publish(processed_dir=tmp_path, trading_date=earlier)
        _publish(processed_dir=tmp_path, trading_date=later, maximum_depth=2)

        with pytest.raises(queries.InvalidBarsError, match="incompatible"):
            queries.load_bars(
                processed_dir=tmp_path,
                instruments=(_INSTRUMENT,),
                maximum_depth=1,
            )

    def test_load_bars_distinguishes_optional_empty_range(self, tmp_path: Path) -> None:
        optional = queries.load_bars(
            processed_dir=tmp_path,
            instruments=(_INSTRUMENT,),
            maximum_depth=1,
            required=False,
        )

        assert optional.is_empty()
        with pytest.raises(queries.NoBarsFoundError):
            queries.load_bars(
                processed_dir=tmp_path,
                instruments=(_INSTRUMENT,),
                maximum_depth=1,
            )

    def test_load_bars_rejects_content_corruption(self, tmp_path: Path) -> None:
        trading_date = datetime.date(2024, 1, 2)
        _publish(processed_dir=tmp_path, trading_date=trading_date)
        partition = next(
            processed.find_partitions(
                processed_dir=tmp_path, instruments=(_INSTRUMENT,)
            )
        )
        _bar(trading_date=trading_date).with_columns(
            pl.lit(9.9).alias(models.COL_MID_PRICE)
        ).write_parquet(partition.data_path)

        with pytest.raises(queries.InvalidBarsError, match="checksum"):
            queries.load_bars(
                processed_dir=tmp_path,
                instruments=(_INSTRUMENT,),
                maximum_depth=1,
            )

    def test_load_bars_rejects_a_missing_explicitly_required_partition(
        self, tmp_path: Path
    ) -> None:
        available = datetime.date(2024, 1, 2)
        missing = datetime.date(2024, 1, 3)
        _publish(processed_dir=tmp_path, trading_date=available)

        with pytest.raises(queries.NoBarsFoundError, match="Missing required"):
            queries.load_bars(
                processed_dir=tmp_path,
                instruments=(_INSTRUMENT,),
                maximum_depth=1,
                required_trading_dates=(available, missing),
            )


def _publish(
    *,
    processed_dir: Path,
    trading_date: datetime.date,
    data: pl.DataFrame | None = None,
    maximum_depth: int = 1,
) -> None:
    processed.write_bars(
        processed_dir=processed_dir,
        instrument=_INSTRUMENT,
        trading_date=trading_date,
        data=(
            data
            if data is not None
            else _bar(trading_date=trading_date, maximum_depth=maximum_depth)
        ),
        maximum_depth=maximum_depth,
        source_fingerprint=trading_date.isoformat(),
        config_fingerprint="config",
        quality={},
    )


def _bar(*, trading_date: datetime.date, maximum_depth: int = 1) -> pl.DataFrame:
    timestamp = datetime.datetime.combine(
        trading_date, datetime.time(hour=12), tzinfo=datetime.UTC
    )
    records: list[models.RawRecord] = []
    for side, prices in (
        (models.QuoteSide.BID, (1.1, 1.09)),
        (models.QuoteSide.OFFER, (1.2, 1.21)),
    ):
        records.extend(
            models.RawQuote(
                timestamp=timestamp,
                instrument=_INSTRUMENT,
                side=side,
                level=level,
                price=prices[level - 1],
                size=1_000_000,
                order_count=1,
                source_line=level,
            )
            for level in range(1, maximum_depth + 1)
        )
    return operations.build_bars(
        records=records,
        instrument=_INSTRUMENT,
        trading_date=trading_date,
        maximum_depth=maximum_depth,
    )
