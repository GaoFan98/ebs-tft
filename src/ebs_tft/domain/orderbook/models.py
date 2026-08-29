"""
Define domain values and column names for the order-book subdomain.
"""

from __future__ import annotations

import enum

MAX_LEVELS: int = 10


class RecordType(enum.StrEnum):
    """
    Identify the record variants supported by the EBS parser.
    """

    QUOTE = "Q"
    DEAL = "D"


class Instrument(enum.StrEnum):
    """
    Identify the currency pairs in the paper-extension study.
    """

    EUR_USD = "EUR_USD"
    EUR_JPY = "EUR_JPY"
    USD_JPY = "USD_JPY"

    @classmethod
    def from_symbol(cls, *, symbol: str) -> Instrument:
        """
        Return the instrument represented by an EBS slash-delimited symbol.

        :raises ValueError: if symbol does not identify a supported instrument
        """
        instrument = cls(symbol.replace("/", "_"))
        if symbol != instrument.to_symbol():
            raise ValueError(f"Invalid EBS instrument symbol: {symbol!r}")
        return instrument

    @classmethod
    def from_filename_part(cls, *, instrument: str) -> Instrument:
        """
        Return the instrument represented by an underscore-delimited filename part.

        :raises ValueError: if instrument does not identify a supported instrument
        """
        return cls(instrument)

    def to_symbol(self) -> str:
        """
        Return the slash-delimited EBS symbol for this instrument.
        """
        return self.value.replace("_", "/")


COL_TIMESTAMP: str = "timestamp"
COL_INSTRUMENT: str = "instrument"
COL_MID_PRICE: str = "mid_price"
COL_SPREAD: str = "spread"
COL_QUOTE_IMBALANCE: str = "quote_imbalance"
COL_BUY_VOLUME: str = "buy_volume"
COL_SELL_VOLUME: str = "sell_volume"
COL_DEAL_FLOW_IMBALANCE: str = "deal_flow_imbalance"
COL_TRADE_COUNT: str = "trade_count"
COL_VWAP: str = "vwap"
COL_DIRECTION_TARGET: str = "direction_target"


def bid_price_col(*, level: int) -> str:
    """
    Return the bid-price column name for a validated depth level.

    :raises ValueError: if level is outside the supported depth range
    """
    _validate_level(level=level)
    return f"bid_price_l{level}"


def ask_price_col(*, level: int) -> str:
    """
    Return the ask-price column name for a validated depth level.

    :raises ValueError: if level is outside the supported depth range
    """
    _validate_level(level=level)
    return f"ask_price_l{level}"


def bid_size_col(*, level: int) -> str:
    """
    Return the bid-size column name for a validated depth level.

    :raises ValueError: if level is outside the supported depth range
    """
    _validate_level(level=level)
    return f"bid_size_l{level}"


def ask_size_col(*, level: int) -> str:
    """
    Return the ask-size column name for a validated depth level.

    :raises ValueError: if level is outside the supported depth range
    """
    _validate_level(level=level)
    return f"ask_size_l{level}"


def all_bid_price_cols(*, max_level: int = MAX_LEVELS) -> list[str]:
    """
    Return bid-price columns from level one through max_level.

    :raises ValueError: if max_level is outside the supported depth range
    """
    _validate_max_level(max_level=max_level)
    return [bid_price_col(level=level) for level in range(1, max_level + 1)]


def all_ask_price_cols(*, max_level: int = MAX_LEVELS) -> list[str]:
    """
    Return ask-price columns from level one through max_level.

    :raises ValueError: if max_level is outside the supported depth range
    """
    _validate_max_level(max_level=max_level)
    return [ask_price_col(level=level) for level in range(1, max_level + 1)]


def all_bid_size_cols(*, max_level: int = MAX_LEVELS) -> list[str]:
    """
    Return bid-size columns from level one through max_level.

    :raises ValueError: if max_level is outside the supported depth range
    """
    _validate_max_level(max_level=max_level)
    return [bid_size_col(level=level) for level in range(1, max_level + 1)]


def all_ask_size_cols(*, max_level: int = MAX_LEVELS) -> list[str]:
    """
    Return ask-size columns from level one through max_level.

    :raises ValueError: if max_level is outside the supported depth range
    """
    _validate_max_level(max_level=max_level)
    return [ask_size_col(level=level) for level in range(1, max_level + 1)]


def all_level_cols(*, max_level: int = MAX_LEVELS) -> list[str]:
    """
    Return every price and size column from level one through max_level.

    :raises ValueError: if max_level is outside the supported depth range
    """
    _validate_max_level(max_level=max_level)
    columns: list[str] = []
    for level in range(1, max_level + 1):
        columns.extend(
            (
                bid_price_col(level=level),
                ask_price_col(level=level),
                bid_size_col(level=level),
                ask_size_col(level=level),
            )
        )
    return columns


def _validate_level(*, level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError(f"level must be an integer; got {level!r}")
    if not 1 <= level <= MAX_LEVELS:
        raise ValueError(
            f"level must be between 1 and {MAX_LEVELS} inclusive; got {level}"
        )


def _validate_max_level(*, max_level: int) -> None:
    if isinstance(max_level, bool) or not isinstance(max_level, int):
        raise ValueError(f"max_level must be an integer; got {max_level!r}")
    if not 1 <= max_level <= MAX_LEVELS:
        raise ValueError(
            f"max_level must be between 1 and {MAX_LEVELS} inclusive; got {max_level}"
        )
