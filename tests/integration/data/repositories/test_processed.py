"""Test atomic, manifest-backed canonical processed storage."""

from __future__ import annotations

import datetime
from concurrent import futures
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

from ebs_tft.data.repositories import processed
from ebs_tft.domain.orderbook import models

_INSTRUMENT = models.Instrument.EUR_USD
_TRADING_DATE = datetime.date(2024, 1, 2)


class TestWriteBars:
    def test_write_bars_publishes_valid_metadata_and_content(
        self, tmp_path: Path
    ) -> None:
        data = _bars()

        partition = _write(processed_dir=tmp_path, data=data)
        actual = processed.read_bars(partition=partition)

        assert actual.equals(data)
        assert partition.metadata.row_count == 2
        assert partition.metadata.maximum_depth == 1
        assert partition.metadata.quality == (("error_rows", 0),)
        assert processed.is_current(
            processed_dir=tmp_path,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            source_fingerprint="source-a",
            config_fingerprint="config-a",
        )
        assert not processed.is_current(
            processed_dir=tmp_path,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            source_fingerprint="changed-source",
            config_fingerprint="config-a",
        )

    def test_write_bars_leaves_no_success_manifest_after_write_failure(
        self, tmp_path: Path
    ) -> None:
        data = _bars()

        with mock.patch.object(
            pl.DataFrame, "write_parquet", side_effect=OSError("simulated")
        ):
            with pytest.raises(processed.UnableToWriteBarsError):
                _write(processed_dir=tmp_path, data=data)

        partition_dir = processed.get_partition_dir(
            processed_dir=tmp_path,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
        )
        assert not (partition_dir / "manifest.json").exists()
        assert list((partition_dir / "generations").glob("*")) == []

    def test_write_bars_serializes_concurrent_publication_to_one_final_generation(
        self, tmp_path: Path
    ) -> None:
        data = _bars()

        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda _: _write(processed_dir=tmp_path, data=data), range(2)
                )
            )

        final_partition = processed.load_partition(
            manifest_path=results[-1].manifest_path
        )
        assert processed.read_bars(partition=final_partition).equals(data)
        assert (
            len(
                list(
                    (final_partition.manifest_path.parent / "generations").glob(
                        "*.parquet"
                    )
                )
            )
            == 1
        )

    def test_write_bars_does_not_globally_lock_different_partitions(
        self, tmp_path: Path
    ) -> None:
        dates = (_TRADING_DATE, datetime.date(2024, 1, 3))

        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            partitions = list(
                executor.map(
                    lambda date: _write(
                        processed_dir=tmp_path,
                        data=_bars(trading_date=date),
                        trading_date=date,
                    ),
                    dates,
                )
            )

        assert [item.metadata.trading_date for item in partitions] == list(dates)


class TestReadBars:
    def test_read_bars_rejects_content_changed_after_publication(
        self, tmp_path: Path
    ) -> None:
        partition = _write(processed_dir=tmp_path, data=_bars())
        _bars().head(1).write_parquet(partition.data_path)

        with pytest.raises(processed.UnableToReadBarsError, match="Invalid"):
            processed.read_bars(partition=partition)


def _write(
    *,
    processed_dir: Path,
    data: pl.DataFrame,
    trading_date: datetime.date = _TRADING_DATE,
) -> processed.ProcessedPartition:
    return processed.write_bars(
        processed_dir=processed_dir,
        instrument=_INSTRUMENT,
        trading_date=trading_date,
        data=data,
        maximum_depth=1,
        source_fingerprint="source-a",
        config_fingerprint="config-a",
        quality={"error_rows": 0},
    )


def _bars(*, trading_date: datetime.date = _TRADING_DATE) -> pl.DataFrame:
    start = datetime.datetime.combine(
        trading_date - datetime.timedelta(days=1),
        datetime.time(hour=22),
        tzinfo=datetime.UTC,
    )
    timestamps = [start, start + datetime.timedelta(milliseconds=100)]
    return pl.DataFrame(
        {
            models.COL_TIMESTAMP: timestamps,
            models.COL_INSTRUMENT: [_INSTRUMENT.value] * 2,
            models.COL_TRADING_DATE: [trading_date] * 2,
            models.bid_price_col(level=1): [1.1, 1.11],
            models.ask_price_col(level=1): [1.2, 1.21],
            models.bid_size_col(level=1): [1_000_000, 1_000_000],
            models.ask_size_col(level=1): [1_000_000, 1_000_000],
            models.bid_order_count_col(level=1): [1, 1],
            models.ask_order_count_col(level=1): [1, 1],
            models.COL_MID_PRICE: [1.15, 1.16],
            models.COL_SPREAD: [0.1, 0.1],
            models.COL_QUOTE_IMBALANCE: [0.0, 0.0],
            models.COL_BUY_VOLUME: [0, 0],
            models.COL_SELL_VOLUME: [0, 0],
            models.COL_TRADE_COUNT: [0, 0],
            models.COL_DEAL_FLOW_IMBALANCE: [0.0, 0.0],
            models.COL_EXTREMAL_PRICE_MEAN: [None, None],
            models.COL_QUOTE_AGE_SECONDS: [40.0, 40.0],
            models.COL_QUOTE_UPDATE_COUNT: [2, 2],
            models.COL_BOOK_OBSERVED: [True, True],
            models.COL_DEALS_OBSERVED: [True, True],
        }
    )
