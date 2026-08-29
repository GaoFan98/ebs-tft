"""
Test order-book domain models.
"""

from __future__ import annotations

import pytest

from ebs_tft.domain.orderbook import models


class TestInstrument:
    def test_from_symbol_returns_each_supported_instrument(self) -> None:
        symbols = {
            "EUR/USD": models.Instrument.EUR_USD,
            "EUR/JPY": models.Instrument.EUR_JPY,
            "USD/JPY": models.Instrument.USD_JPY,
        }

        actual = {
            symbol: models.Instrument.from_symbol(symbol=symbol) for symbol in symbols
        }

        assert actual == symbols

    def test_from_symbol_rejects_an_unknown_pair(self) -> None:
        with pytest.raises(ValueError):
            models.Instrument.from_symbol(symbol="GBP/USD")

    def test_from_symbol_rejects_filename_notation(self) -> None:
        with pytest.raises(ValueError, match="Invalid EBS instrument symbol"):
            models.Instrument.from_symbol(symbol="EUR_USD")

    def test_from_filename_part_round_trips_each_supported_instrument(self) -> None:
        for instrument in models.Instrument:
            actual = models.Instrument.from_filename_part(instrument=instrument.value)

            assert actual is instrument
            assert actual.to_symbol().replace("/", "_") == instrument.value


class TestLevelColumns:
    def test_all_level_cols_builds_each_cumulative_level(self) -> None:
        columns = models.all_level_cols(max_level=2)

        assert columns == [
            "bid_price_l1",
            "ask_price_l1",
            "bid_size_l1",
            "ask_size_l1",
            "bid_order_count_l1",
            "ask_order_count_l1",
            "bid_price_l2",
            "ask_price_l2",
            "bid_size_l2",
            "ask_size_l2",
            "bid_order_count_l2",
            "ask_order_count_l2",
        ]

    @pytest.mark.parametrize("level", [0, models.MAX_LEVELS + 1])
    def test_bid_price_col_rejects_a_level_outside_the_supported_range(
        self, level: int
    ) -> None:
        with pytest.raises(ValueError, match="level must be between"):
            models.bid_price_col(level=level)

    def test_bid_price_col_rejects_a_boolean_level(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            models.bid_price_col(level=True)

    @pytest.mark.parametrize("max_level", [0, models.MAX_LEVELS + 1])
    def test_all_level_cols_rejects_an_invalid_maximum(self, max_level: int) -> None:
        with pytest.raises(ValueError, match="max_level must be between"):
            models.all_level_cols(max_level=max_level)
