"""
Parse single EBS level 2 csv.gz file into typed records.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from pathlib import Path

import attrs

logger = logging.getLogger(__name__)


class UnableToParseRowError(Exception):
    """
    Raised when a row cannot be parsed into a RawEBSRow.
    """


# Value objects
@attrs.frozen
class RawEBSQuoteRow:
    """
    Single parsed Q record from an EBS Level 2 .csv.gz file.
    Represents one order book depth entry at one point in time.
    Column layout (no header in file, 9 columns):
        0: date         e.g. "2024/01/01"
        1: time         e.g. "22:00:00.100"
        2: symbol       e.g. "EUR/USD"
        3: record_type  always "Q" for quote records
        4: side         0 = Bid, 1 = Ask
        5: level        1-10  (1 = best price)
        6: price        may be None for initialization rows
        7: size         notional in base currency
        8: count        number of aggregated orders at this level
    """

    date: str
    time: str
    # currency pair
    symbol: str
    # always "Q" for quote records
    record_type: str
    # 0 = bid side (descending prices, level 1 = highest bid)
    # 1 = ask side (ascending prices, level 1 = lowest ask)
    side: int
    # depth level: 1 is best (closest to mid), 10 is worst (furthest from mid)
    level: int
    # None on initialization rows (first snapshot of a session with no price yet)
    price: float | None
    # notional size available at this price level, in base currency units.
    # e.g., 1000000 means 1 million EUR available at this bid level.
    size: int
    # number of individual orders aggregated into this depth level.
    # EBS aggregates all orders at the same price into a single row.
    count: int


@attrs.frozen
class RawEBSDealRow:
    """
    Single parsed D record from an EBS Level 2 .csv.gz file.
    Represents one actual executed trade during a 100ms time-slice.
    Deal records are critical for direction prediction — they capture actual
    order flow (who was aggressive: buyers or sellers), which is more informative
    than passive quotes alone.
    Column layout (no header in file, 10 columns):
        0: date          e.g. "2024/01/01"
        1: time          e.g. "22:39:07.900"
        2: symbol        e.g. "EUR/USD"
        3: record_type   always "D" for deal records
        4: side          1 = highest paid (buy-initiated), 0 = lowest given (sell-initiated)
        5: (empty)       no depth level concept for deals
        6: deal_price    price at which the trade executed (None if no deals this slice)
        7: deal_size     volume traded at deal_price
        8: deal_count    number of individual trades at deal_price
        9: total_volume  total volume traded across all deals in this time-slice
    """

    date: str
    time: str
    # currency pair
    symbol: str
    # always "D" for deal records
    record_type: str
    # 1 = highest paid (an aggressive buyer lifted the ask — bullish)
    # 0 = lowest given (an aggressive seller hit the bid — bearish)
    # This is the key field for computing order flow imbalance.
    side: int
    # Price where the trade executed. None if no trade occurred this slice
    # (some D rows are emitted even in quiet periods with empty price fields).
    deal_price: float | None
    # Volume traded at deal_price in base currency units
    deal_size: int
    # Number of individual trades that occurred at deal_price
    deal_count: int
    # Total notional volume across ALL deals in this 100ms slice.
    # May be larger than deal_size if multiple price levels were hit.
    total_volume: int


# Parsers
def parse_quotes(*, path: Path) -> Iterator[RawEBSQuoteRow]:
    """
    Yield parsed Q (Quote) records from a single EBS Level 2 csv.gz file.
    Skips D records, empty lines, and malformed rows.

    :raises FileNotFoundError: if the file does not exist
    :raises OSError: if the file cannot be opened or decompressed
    """
    logger.debug("Parsing quotes from EBS file", extra={"path": str(path)})
    yield from _parse_file(path=path, target_record_type="Q")


def parse_deals(*, path: Path) -> Iterator[RawEBSDealRow]:
    """
    Yield parsed D (Deal) records from a single EBS Level 2 csv.gz file.
    Skips Q records, empty lines, and malformed rows.

    :raises FileNotFoundError: if the file does not exist
    :raises OSError: if the file cannot be opened or decompressed
    """
    logger.debug("Parsing deals from EBS file", extra={"path": str(path)})
    yield from _parse_file(path=path, target_record_type="D")


def _parse_header(
    *,
    line: str,
    expected_record_type: str,
    expected_column_count: int,
) -> list[str]:
    """
    Split a raw line and validate the parts that are common to ALL record types:

    :raises UnableToParseRowError: on any validation failure
    """
    parts = line.split(",")

    if len(parts) < 4:
        raise UnableToParseRowError(f"Too few columns ({len(parts)}): {line!r}")
    if parts[3] != expected_record_type:
        raise UnableToParseRowError(
            f"Expected {expected_record_type!r} record, got {parts[3]!r}: {line!r}"
        )
    if len(parts) != expected_column_count:
        raise UnableToParseRowError(
            f"Expected {expected_column_count} columns for {expected_record_type!r} "
            f"record, got {len(parts)}: {line!r}"
        )
    return parts


def _validate_side(*, side: int, line: str) -> None:
    """
    Side must be 0 (bid/sell-initiated) or 1 (ask/buy-initiated).

    :raises UnableToParseRowError: if side is not 0 or 1
    """
    if side not in (0, 1):
        raise UnableToParseRowError(f"Side must be 0 or 1, got {side}: {line!r}")


def _parse_file(
    *,
    path: Path,
    target_record_type: str,
) -> Iterator[RawEBSQuoteRow | RawEBSDealRow]:
    # validate if file exist first
    if not path.exists():
        raise FileNotFoundError(f"EBS data file not found: {path}")

    parse_fn = _parse_quote_line if target_record_type == "Q" else _parse_deal_line

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_num, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                row = parse_fn(line=line)
            except UnableToParseRowError:
                logger.debug(
                    "Skipping unparseable row",
                    extra={
                        "path": str(path),
                        "line_number": line_num,
                        "target_type": target_record_type,
                    },
                )
                continue

            yield row


def _parse_quote_line(*, line: str) -> RawEBSQuoteRow:
    """
    Parse a single line and return a RawEBSQuoteRow.

    :raises UnableToParseRowError: if not a Q record, wrong column count, or bad types
    """
    parts = _parse_header(
        line=line,
        expected_record_type="Q",
        expected_column_count=9,
    )
    (
        date,
        time_,
        symbol,
        record_type,
        raw_side,
        raw_level,
        raw_price,
        raw_size,
        raw_count,
    ) = parts

    side = _parse_int(raw_side, field="side", line=line)
    _validate_side(side=side, line=line)

    return RawEBSQuoteRow(
        date=date,
        time=time_,
        symbol=symbol,
        record_type=record_type,
        side=side,
        level=_parse_int(raw_level, field="level", line=line),
        price=_parse_optional_float(raw_price, field="price", line=line),
        size=_parse_int(raw_size, field="size", line=line),
        count=_parse_int(raw_count, field="count", line=line),
    )


def _parse_deal_line(*, line: str) -> RawEBSDealRow:
    """
    Parse a single line and return a RawEBSDealRow.

    :raises UnableToParseRowError: if not a D record, wrong column count, or bad types
    """
    parts = _parse_header(
        line=line,
        expected_record_type="D",
        expected_column_count=10,
    )
    # Column 5 is always empty for D records (no level concept for deals)
    (
        date,
        time_,
        symbol,
        record_type,
        raw_side,
        _,
        raw_deal_price,
        raw_deal_size,
        raw_deal_count,
        raw_total_volume,
    ) = parts

    side = _parse_int(raw_side, field="side", line=line)
    _validate_side(side=side, line=line)

    return RawEBSDealRow(
        date=date,
        time=time_,
        symbol=symbol,
        record_type=record_type,
        side=side,
        deal_price=_parse_optional_float(raw_deal_price, field="deal_price", line=line),
        deal_size=_parse_int(raw_deal_size, field="deal_size", line=line),
        deal_count=_parse_int(raw_deal_count, field="deal_count", line=line),
        total_volume=_parse_int(raw_total_volume, field="total_volume", line=line),
    )


# Type conversion helpers
def _parse_int(value: str, *, field: str, line: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise UnableToParseRowError(f"Invalid {field} {value!r}: {line!r}")


def _parse_optional_float(value: str, *, field: str, line: str) -> float | None:
    if value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise UnableToParseRowError(f"Invalid {field} {value!r}: {line!r}")
