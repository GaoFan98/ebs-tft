"""Parse one compressed EBS Level 2 CSV in a single, accountable pass."""

from __future__ import annotations

import csv
import datetime
import gzip
import math
from collections.abc import Iterator
from pathlib import Path

import attrs

from ebs_tft.domain.orderbook import models

_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


class UnableToParseRowError(Exception):
    """Indicate that a source row violates the documented parser contract."""


@attrs.frozen
class ParseIssue:
    """Describe one sanitized parse failure without retaining raw row content."""

    path: Path
    line_number: int
    reason: str


@attrs.define
class ParseAudit:
    """Account for every physical input line consumed by the parser."""

    physical_lines: int = 0
    empty_lines: int = 0
    quote_rows: int = 0
    deal_rows: int = 0
    error_rows: int = 0
    issues: list[ParseIssue] = attrs.field(factory=list)

    @property
    def accounted_lines(self) -> int:
        """Return the number of lines assigned to exactly one outcome."""
        return self.empty_lines + self.quote_rows + self.deal_rows + self.error_rows


def parse_rows(
    *,
    path: Path,
    expected_instrument: models.Instrument,
    expected_trading_date: datetime.date,
    strict: bool = True,
    audit: ParseAudit | None = None,
    maximum_issues: int = 100,
) -> Iterator[models.RawRecord]:
    """Yield validated Q and D records in source order from one gzip member.

    Source timestamps are UTC. An EBS trading-date file can begin on the preceding
    UTC calendar date because the FX trading date rolls at 17:00 New York.
    """
    if maximum_issues < 0:
        raise ValueError("maximum_issues must be non-negative")
    if not path.is_file():
        raise FileNotFoundError(f"EBS data file not found: {path}")

    parse_audit = audit if audit is not None else ParseAudit()
    previous_timestamp: datetime.datetime | None = None
    try:
        with gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
            errors="strict",
            newline="",
        ) as stream:
            reader = csv.reader(stream, strict=True)
            for line_number, columns in enumerate(reader, start=1):
                parse_audit.physical_lines += 1
                if not columns or all(not value for value in columns):
                    parse_audit.empty_lines += 1
                    continue
                try:
                    record = _parse_columns(
                        columns=columns,
                        line_number=line_number,
                        expected_instrument=expected_instrument,
                        expected_trading_date=expected_trading_date,
                    )
                    if (
                        previous_timestamp is not None
                        and record.timestamp < previous_timestamp
                    ):
                        raise UnableToParseRowError("timestamp is out of source order")
                    previous_timestamp = record.timestamp
                except UnableToParseRowError as exc:
                    issue = ParseIssue(
                        path=path, line_number=line_number, reason=str(exc)
                    )
                    parse_audit.error_rows += 1
                    if len(parse_audit.issues) < maximum_issues:
                        parse_audit.issues.append(issue)
                    if strict:
                        raise UnableToParseRowError(
                            f"Unable to parse {path} at line {line_number}: {exc}"
                        ) from exc
                    continue

                if isinstance(record, models.RawQuote):
                    parse_audit.quote_rows += 1
                else:
                    parse_audit.deal_rows += 1
                yield record
    except (UnicodeError, csv.Error) as exc:
        raise UnableToParseRowError(f"Unable to decode EBS CSV file: {path}") from exc


def _parse_columns(
    *,
    columns: list[str],
    line_number: int,
    expected_instrument: models.Instrument,
    expected_trading_date: datetime.date,
) -> models.RawRecord:
    if len(columns) < 4:
        raise UnableToParseRowError(f"expected at least 4 columns; got {len(columns)}")
    try:
        record_type = models.RecordType(columns[3])
    except ValueError as exc:
        raise UnableToParseRowError(f"unknown record marker {columns[3]!r}") from exc

    timestamp = _parse_timestamp(date_value=columns[0], time_value=columns[1])
    allowed_dates = {
        expected_trading_date,
        expected_trading_date - datetime.timedelta(days=1),
    }
    if timestamp.date() not in allowed_dates:
        raise UnableToParseRowError(
            "row calendar date is outside its EBS trading-date file"
        )
    try:
        instrument = models.Instrument.from_symbol(symbol=columns[2])
    except ValueError as exc:
        raise UnableToParseRowError(f"invalid symbol {columns[2]!r}") from exc
    if instrument is not expected_instrument:
        raise UnableToParseRowError(
            f"symbol {columns[2]!r} does not match {expected_instrument.to_symbol()!r}"
        )

    if record_type is models.RecordType.QUOTE:
        return _parse_quote(
            columns=columns,
            timestamp=timestamp,
            instrument=instrument,
            line_number=line_number,
        )
    return _parse_deal(
        columns=columns,
        timestamp=timestamp,
        instrument=instrument,
        line_number=line_number,
    )


def _parse_quote(
    *,
    columns: list[str],
    timestamp: datetime.datetime,
    instrument: models.Instrument,
    line_number: int,
) -> models.RawQuote:
    if len(columns) != 9:
        raise UnableToParseRowError(f"Q record requires 9 columns; got {len(columns)}")
    side = _enum_int(value=columns[4], enum_type=models.QuoteSide, field="quote side")
    level = _integer(value=columns[5], field="level")
    if not 1 <= level <= models.MAX_LEVELS:
        raise UnableToParseRowError(f"level must be in 1..{models.MAX_LEVELS}")
    price = _optional_positive_float(value=columns[6], field="price")
    size = _non_negative_integer(value=columns[7], field="size")
    order_count = _non_negative_integer(value=columns[8], field="order count")
    if price is None and (size != 0 or order_count != 0):
        raise UnableToParseRowError("empty quote price requires zero size and count")
    if price is not None and (size == 0 or order_count == 0):
        raise UnableToParseRowError("priced quote requires positive size and count")
    return models.RawQuote(
        timestamp=timestamp,
        instrument=instrument,
        side=side,
        level=level,
        price=price,
        size=size,
        order_count=order_count,
        source_line=line_number,
    )


def _parse_deal(
    *,
    columns: list[str],
    timestamp: datetime.datetime,
    instrument: models.Instrument,
    line_number: int,
) -> models.RawDeal:
    if len(columns) != 10:
        raise UnableToParseRowError(f"D record requires 10 columns; got {len(columns)}")
    if columns[5] != "":
        raise UnableToParseRowError("D record level column must be empty")
    side = _enum_int(value=columns[4], enum_type=models.DealSide, field="deal side")
    price = _optional_positive_float(value=columns[6], field="deal price")
    if price is None:
        raise UnableToParseRowError("D record deal price must be present")
    price_volume = _positive_integer(value=columns[7], field="deal-price volume")
    deal_count = _positive_integer(value=columns[8], field="deal count")
    total_volume = _positive_integer(value=columns[9], field="total volume")
    if total_volume < price_volume:
        raise UnableToParseRowError("total volume is below deal-price volume")
    return models.RawDeal(
        timestamp=timestamp,
        instrument=instrument,
        side=side,
        extremal_price=price,
        extremal_price_volume=price_volume,
        deal_count=deal_count,
        total_volume=total_volume,
        source_line=line_number,
    )


def _parse_timestamp(*, date_value: str, time_value: str) -> datetime.datetime:
    try:
        parsed = datetime.datetime.strptime(
            f"{date_value} {time_value}", _TIMESTAMP_FORMAT
        )
    except ValueError as exc:
        raise UnableToParseRowError("invalid UTC date/time") from exc
    return parsed.replace(tzinfo=datetime.UTC)


def _enum_int[SideT: (models.QuoteSide, models.DealSide)](
    *, value: str, enum_type: type[SideT], field: str
) -> SideT:
    try:
        return enum_type(_integer(value=value, field=field))
    except ValueError as exc:
        raise UnableToParseRowError(f"{field} must be 0 or 1") from exc


def _integer(*, value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise UnableToParseRowError(f"invalid {field}") from exc


def _non_negative_integer(*, value: str, field: str) -> int:
    parsed = _integer(value=value, field=field)
    if parsed < 0:
        raise UnableToParseRowError(f"{field} must be non-negative")
    return parsed


def _positive_integer(*, value: str, field: str) -> int:
    parsed = _integer(value=value, field=field)
    if parsed <= 0:
        raise UnableToParseRowError(f"{field} must be positive")
    return parsed


def _optional_positive_float(*, value: str, field: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise UnableToParseRowError(f"invalid {field}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise UnableToParseRowError(f"{field} must be finite and positive")
    return parsed
