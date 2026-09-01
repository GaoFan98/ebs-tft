"""Test native-grid reconstruction and direction targets."""

import datetime

import pytest

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import operations


class TestBuildNativeStates:
    def test_reconstructs_sides_and_slice_deals_without_temporal_aggregation(
        self,
    ) -> None:
        timestamp = datetime.datetime(2024, 1, 2, 22, tzinfo=datetime.UTC)
        records: list[orderbook_models.RawRecord] = [
            _quote(
                timestamp=timestamp,
                side=orderbook_models.QuoteSide.BID,
                price=1.0,
            ),
            _quote(
                timestamp=timestamp,
                side=orderbook_models.QuoteSide.OFFER,
                price=1.2,
            ),
            orderbook_models.RawDeal(
                timestamp=timestamp + datetime.timedelta(milliseconds=100),
                instrument=orderbook_models.Instrument.EUR_USD,
                side=orderbook_models.DealSide.PAID,
                extremal_price=1.2,
                extremal_price_volume=1_000_000,
                deal_count=1,
                total_volume=2_000_000,
                source_line=3,
            ),
            _quote(
                timestamp=timestamp + datetime.timedelta(milliseconds=200),
                side=orderbook_models.QuoteSide.BID,
                price=1.1,
            ),
        ]

        actual = operations.build_native_states(
            records=records,
            instrument=orderbook_models.Instrument.EUR_USD,
            trading_date=datetime.date(2024, 1, 3),
            grid_steps=4,
            maximum_staleness_steps=10,
        )

        assert actual.height == 4
        assert actual[orderbook_models.COL_BOOK_OBSERVED].to_list() == [
            True,
            True,
            True,
            True,
        ]
        assert actual[orderbook_models.COL_BUY_VOLUME].to_list() == [
            0,
            2_000_000,
            0,
            0,
        ]
        assert actual[orderbook_models.COL_MID_PRICE].to_list() == [
            1.1,
            1.1,
            1.15,
            1.15,
        ]

    def test_rejects_a_timestamp_outside_the_vendor_grid(self) -> None:
        records = [
            _quote(
                timestamp=datetime.datetime(
                    2024, 1, 2, 22, microsecond=50_000, tzinfo=datetime.UTC
                ),
                side=orderbook_models.QuoteSide.BID,
                price=1.0,
            )
        ]

        with pytest.raises(
            operations.InvalidNativeStateError, match="outside the 100 ms grid"
        ):
            operations.build_native_states(
                records=records,
                instrument=orderbook_models.Instrument.EUR_USD,
                trading_date=datetime.date(2024, 1, 3),
                grid_steps=1,
                maximum_staleness_steps=10,
            )


class TestAddDirectionTargets:
    def test_uses_the_exact_future_grid_state(self) -> None:
        timestamp = datetime.datetime(2024, 1, 2, 22, tzinfo=datetime.UTC)
        records: list[orderbook_models.RawRecord] = [
            _quote(
                timestamp=timestamp,
                side=orderbook_models.QuoteSide.BID,
                price=1.0,
            ),
            _quote(
                timestamp=timestamp,
                side=orderbook_models.QuoteSide.OFFER,
                price=1.2,
            ),
            _quote(
                timestamp=timestamp + datetime.timedelta(milliseconds=200),
                side=orderbook_models.QuoteSide.BID,
                price=1.1,
            ),
        ]
        states = operations.build_native_states(
            records=records,
            instrument=orderbook_models.Instrument.EUR_USD,
            trading_date=datetime.date(2024, 1, 3),
            grid_steps=4,
            maximum_staleness_steps=10,
        )

        actual = operations.add_direction_targets(data=states, horizon_steps=(1,))

        assert actual["direction_h1"].to_list() == [0, 1, 0, None]


def _quote(
    *,
    timestamp: datetime.datetime,
    side: orderbook_models.QuoteSide,
    price: float,
) -> orderbook_models.RawQuote:
    return orderbook_models.RawQuote(
        timestamp=timestamp,
        instrument=orderbook_models.Instrument.EUR_USD,
        side=side,
        level=1,
        price=price,
        size=1_000_000,
        order_count=1,
        source_line=1,
    )
