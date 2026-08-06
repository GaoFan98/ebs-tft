"""
Domain value objects and constants for the order book subdomain.
"""

from __future__ import annotations

from enum import Enum

# Constants
# num of order books depth level in EBS level 2 data
MAX_LEVELS: int = 10


# Enums
class RecordType(Enum):
    # order book depth snapshot
    QUOTE = "Q"
    # executed trade
    DEAL = "D"


class Instrument(Enum):
    # target currency pairs
    EUR_USD = "EUR_USD"
    EUR_JPY = "EUR_JPY"
    USD_JPY = "USD_JPY"

    @classmethod
    def from_symbol(cls, symbol: str) -> Instrument:
        """
        Convert from the raw symbol string in the CSV ("EUR/USD").

        :raises ValueError: if the symbol is not one of the three target pairs
        """
        normalized = symbol.replace("/", "_")
        return normalized

    @classmethod
    def from_filename_part(cls, instrument_str=str) -> Instrument:
        """
        Convert from instrument string in the filename ("EUR_USD" ).

        :raises ValueError: if the string is not a known instrument
        """
        return cls(instrument_str)

    def to_symbol(self) -> str:
        return self.value.replace("_", "/")


# Column name constants for processed bar DataFrames
#
# Naming convention:
#   COL_{DESCRIPTION} for scalar columns (one value per bar)
#   bid_price_col(), ask_size_col() etc. for per-level columns

# Identifiers
# Datetime of the 1-minute bar (the minute boundary, e.g. 22:01:00)
COL_TIMESTAMP: str = "timestamp"
# Currency pair as underscore string: "EUR_USD"
COL_INSTRUMENT: str = "instrument"
"""
Mid-price and spread (derived from best bid and best ask)
Mid-price = (best_bid_price + best_ask_price) / 2
Primary target variable: we predict the direction of series.
"""
COL_MID_PRICE: str = "mid_price"
"""
Spread = best_ask_price - best_bid_price
Measure of transaction cost and liquidity. Wide spread = illiquid.
"""
COL_SPREAD: str = "spread"
"""
Quote-side order imbalance
(total_bid_size_l1 - total_ask_size_l1) / (total_bid_size_l1 + total_ask_size_l1)
Computed from L1 sizes. Positive = more buy-side liquidity.
"""
COL_QUOTE_IMBALANCE: str = "quote_imbalance"
"""
Total notional volume of buy-initiated deals in this 1-minute bar.
"Buy-initiated" = an aggressive buyer lifted the ask (side=1 in D records).
"""
COL_BUY_VOLUME: str = "buy_volume"
"""
Total notional volume of sell-initiated deals in this 1-minute bar.
"Sell-initiated" = an aggressive seller hit the bid (side=0 in D records).
"""
COL_SELL_VOLUME: str = "sell_volume"
"""
(buy_volume - sell_volume) / (buy_volume + sell_volume)
Primary order flow signal. Strong positive = aggressive buying = bullish.
NaN when total volume is 0 (no deals this minute: common in quiet periods).
"""
COL_DEAL_FLOW_IMBALANCE: str = "deal_flow_imbalance"
# Total number of individual trades (across both sides) in this 1-minute bar.
COL_TRADE_COUNT: str = "trade_count"
"""
Volume-weighted average deal price across all deals in this bar.
Tells you where trades actually cleared vs. where quotes were.
"""
COL_VWAP: str = "vwap"
"""
Direction target.
sign(mid_price[t+H] - mid_price[t]): +1.0 = up, -1.0 = down
H = forecast horizon in bars 
"""
COL_DIRECTION_TARGET: str = "direction_target"

# Per-level column name helpers
#
# These generate column names for the 10 × 4 = 40 per-level columns.
#
# Example:
#   bid_price_col(level=1)  → "bid_price_l1"
#   ask_size_col(level=10)  → "ask_size_l10"
