"""Test causal reconstruction of EBS side snapshots and deals."""

from __future__ import annotations

import datetime

import pytest

from ebs_tft.domain.orderbook import models, operations

_INSTRUMENT = models.Instrument.EUR_USD
_TRADING_DATE = datetime.date(2024, 1, 2)


class TestBuildBars:
    def test_build_bars_uses_minute_end_state_not_update_means(self) -> None:
        records = [
            *_side(
                timestamp=_timestamp(minute=0, second=10),
                side=models.QuoteSide.BID,
                prices=(1.10, 1.09),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.20, 1.21),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=40),
                side=models.QuoteSide.BID,
                prices=(1.11, 1.10),
            ),
        ]

        actual = operations.build_bars(
            records=records,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            maximum_depth=2,
            maximum_staleness_seconds=60,
        )

        assert actual.height == 1
        assert actual[models.bid_price_col(level=1)][0] == pytest.approx(1.11)
        assert actual[models.ask_price_col(level=1)][0] == pytest.approx(1.20)
        assert actual[models.COL_MID_PRICE][0] == pytest.approx(1.155)
        assert actual[models.COL_QUOTE_UPDATE_COUNT][0] == 3
        assert actual[models.COL_QUOTE_AGE_SECONDS][0] == pytest.approx(40)

    def test_build_bars_clears_a_side_on_reset(self) -> None:
        records = [
            *_side(
                timestamp=_timestamp(minute=0, second=10),
                side=models.QuoteSide.BID,
                prices=(1.10,),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.20,),
            ),
            _quote(
                timestamp=_timestamp(minute=1, second=10),
                side=models.QuoteSide.BID,
                level=1,
                price=None,
            ),
            *_side(
                timestamp=_timestamp(minute=1, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.21,),
            ),
        ]
        audit = operations.BuildAudit()

        actual = operations.build_bars(
            records=records,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            maximum_depth=1,
            maximum_staleness_seconds=60,
            audit=audit,
        )

        assert actual.height == 2
        assert actual[models.COL_BOOK_OBSERVED].to_list() == [True, False]
        assert actual[models.bid_price_col(level=1)][1] is None
        assert audit.quote_resets == 1

    def test_build_bars_marks_state_missing_after_staleness_limit(self) -> None:
        records: list[models.RawRecord] = [
            *_side(
                timestamp=_timestamp(minute=0, second=10),
                side=models.QuoteSide.BID,
                prices=(1.10,),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.20,),
            ),
            _deal(timestamp=_timestamp(minute=2, second=1)),
        ]

        actual = operations.build_bars(
            records=records,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            maximum_depth=1,
            maximum_staleness_seconds=60,
        )

        assert actual[models.COL_BOOK_OBSERVED].to_list() == [True, False, False]
        assert actual[models.COL_BUY_VOLUME].to_list() == [0, None, 1_000_000]

    def test_build_bars_uses_total_side_volume_and_does_not_claim_vwap(self) -> None:
        records: list[models.RawRecord] = [
            *_side(
                timestamp=_timestamp(minute=0, second=10),
                side=models.QuoteSide.BID,
                prices=(1.10,),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.20,),
            ),
            _deal(timestamp=_timestamp(minute=0, second=30), total_volume=3_000_000),
        ]

        actual = operations.build_bars(
            records=records,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            maximum_depth=1,
        )

        assert actual[models.COL_BUY_VOLUME][0] == 3_000_000
        assert actual[models.COL_EXTREMAL_PRICE_MEAN][0] == pytest.approx(1.2)
        assert "vwap" not in actual.columns

    def test_build_bars_never_applies_a_future_side_to_an_earlier_minute(self) -> None:
        records: list[models.RawRecord] = [
            *_side(
                timestamp=_timestamp(minute=0, second=10),
                side=models.QuoteSide.BID,
                prices=(1.10,),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.20,),
            ),
            _deal(timestamp=_timestamp(minute=0, second=30)),
            *_side(
                timestamp=_timestamp(minute=1, second=10),
                side=models.QuoteSide.BID,
                prices=(1.11,),
            ),
            *_side(
                timestamp=_timestamp(minute=1, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.21,),
            ),
        ]

        actual = operations.build_bars(
            records=records,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            maximum_depth=1,
        )

        assert actual[models.bid_price_col(level=1)].to_list() == [1.10, 1.11]
        assert actual[models.ask_price_col(level=1)].to_list() == [1.20, 1.21]
        assert all(age >= 0 for age in actual[models.COL_QUOTE_AGE_SECONDS])

    def test_build_bars_rejects_duplicate_levels_in_one_side_snapshot(self) -> None:
        timestamp = _timestamp(minute=0, second=10)
        records = [
            _quote(timestamp=timestamp, side=models.QuoteSide.BID, level=1, price=1.1),
            _quote(timestamp=timestamp, side=models.QuoteSide.BID, level=1, price=1.1),
        ]

        with pytest.raises(operations.InvalidBookStateError, match="duplicate"):
            operations.build_bars(
                records=records,
                instrument=_INSTRUMENT,
                trading_date=_TRADING_DATE,
                maximum_depth=1,
            )

    def test_build_bars_quarantines_a_crossed_book(self) -> None:
        records = [
            *_side(
                timestamp=_timestamp(minute=0, second=10),
                side=models.QuoteSide.BID,
                prices=(1.20,),
            ),
            *_side(
                timestamp=_timestamp(minute=0, second=20),
                side=models.QuoteSide.OFFER,
                prices=(1.10,),
            ),
        ]
        audit = operations.BuildAudit()

        actual = operations.build_bars(
            records=records,
            instrument=_INSTRUMENT,
            trading_date=_TRADING_DATE,
            maximum_depth=1,
            audit=audit,
        )

        assert actual[models.COL_BOOK_OBSERVED][0] is False
        assert actual[models.COL_MID_PRICE][0] is None
        assert audit.invalid_book_bars == 1


def _timestamp(*, minute: int, second: int) -> datetime.datetime:
    return datetime.datetime(2024, 1, 1, 22, minute, second, tzinfo=datetime.UTC)


def _side(
    *, timestamp: datetime.datetime, side: models.QuoteSide, prices: tuple[float, ...]
) -> list[models.RawQuote]:
    return [
        _quote(timestamp=timestamp, side=side, level=level, price=price)
        for level, price in enumerate(prices, start=1)
    ]


def _quote(
    *,
    timestamp: datetime.datetime,
    side: models.QuoteSide,
    level: int,
    price: float | None,
) -> models.RawQuote:
    return models.RawQuote(
        timestamp=timestamp,
        instrument=_INSTRUMENT,
        side=side,
        level=level,
        price=price,
        size=1_000_000 if price is not None else 0,
        order_count=1 if price is not None else 0,
        source_line=level,
    )


def _deal(
    *, timestamp: datetime.datetime, total_volume: int = 1_000_000
) -> models.RawDeal:
    return models.RawDeal(
        timestamp=timestamp,
        instrument=_INSTRUMENT,
        side=models.DealSide.PAID,
        extremal_price=1.2,
        extremal_price_volume=1_000_000,
        deal_count=1,
        total_volume=total_volume,
        source_line=100,
    )
