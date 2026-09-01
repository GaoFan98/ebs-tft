"""Build causal native-resolution states and direction targets for the pilot."""

from __future__ import annotations

import datetime
import itertools
from collections.abc import Iterable, Sequence
from typing import cast

import attrs
import polars as pl

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models


class InvalidNativeStateError(Exception):
    """Indicate that source records cannot form the documented native state."""


@attrs.frozen
class _Level:
    price: float
    size: int
    order_count: int


@attrs.frozen
class _SideState:
    timestamp: datetime.datetime
    levels: tuple[_Level, ...]


def build_native_states(
    *,
    records: Iterable[orderbook_models.RawRecord],
    instrument: orderbook_models.Instrument,
    trading_date: datetime.date,
    grid_steps: int | None,
    maximum_staleness_steps: int,
    maximum_depth: int = 1,
) -> pl.DataFrame:
    """Return end-of-slice causal states on the vendor's native 100 ms grid."""
    if grid_steps is not None and grid_steps <= 0:
        raise ValueError("grid_steps must be positive")
    if maximum_staleness_steps <= 0:
        raise ValueError("maximum_staleness_steps must be positive")

    groups = itertools.groupby(records, key=lambda record: record.timestamp)
    try:
        next_timestamp, next_group = next(groups)
    except StopIteration as exc:
        raise InvalidNativeStateError("source file contains no records") from exc
    _validate_grid_timestamp(timestamp=next_timestamp)

    orderbook_models.all_level_cols(max_level=maximum_depth)
    side_states: dict[orderbook_models.QuoteSide, _SideState] = {}
    rows: list[dict[str, object]] = []
    grid_timestamp = next_timestamp
    step = 0
    while grid_steps is None or step < grid_steps:
        timestamp_records: Sequence[orderbook_models.RawRecord] = ()
        if next_timestamp == grid_timestamp:
            timestamp_records = tuple(next_group)
            _apply_timestamp_records(
                records=timestamp_records,
                timestamp=grid_timestamp,
                instrument=instrument,
                side_states=side_states,
            )
            try:
                next_timestamp, next_group = next(groups)
                _validate_grid_timestamp(timestamp=next_timestamp)
            except StopIteration:
                next_timestamp = datetime.datetime.max.replace(tzinfo=datetime.UTC)
        elif next_timestamp < grid_timestamp:
            raise InvalidNativeStateError("source timestamps are not ordered")

        rows.append(
            _state_row(
                timestamp=grid_timestamp,
                trading_date=trading_date,
                instrument=instrument,
                timestamp_records=timestamp_records,
                side_states=side_states,
                maximum_staleness_steps=maximum_staleness_steps,
                maximum_depth=maximum_depth,
            )
        )
        step += 1
        if grid_steps is None and next_timestamp == datetime.datetime.max.replace(
            tzinfo=datetime.UTC
        ):
            break
        grid_timestamp += models.GRID_INTERVAL
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        orderbook_models.COL_TIMESTAMP
    )


def add_direction_targets(
    *, data: pl.DataFrame, horizon_steps: tuple[int, ...]
) -> pl.DataFrame:
    """Add exact-grid down/flat/up targets without using future features."""
    result = data
    for steps in horizon_steps:
        target_column = models.direction_column(horizon_steps=steps)
        future_mid = pl.col(orderbook_models.COL_MID_PRICE).shift(-steps)
        future_observed = pl.col(orderbook_models.COL_BOOK_OBSERVED).shift(-steps)
        valid = (
            pl.col(orderbook_models.COL_BOOK_OBSERVED)
            & future_observed.fill_null(False)
            & pl.col(orderbook_models.COL_MID_PRICE).is_not_null()
            & future_mid.is_not_null()
        )
        result = result.with_columns(
            pl.when(valid)
            .then(
                (future_mid - pl.col(orderbook_models.COL_MID_PRICE))
                .sign()
                .cast(pl.Int8)
            )
            .otherwise(None)
            .alias(target_column)
        )
    return result


def target_balance(
    *, data: pl.DataFrame, horizon_steps: tuple[int, ...]
) -> pl.DataFrame:
    """Summarize valid three-class labels for every candidate horizon."""
    rows: list[dict[str, int | float]] = []
    for steps in horizon_steps:
        column = models.direction_column(horizon_steps=steps)
        valid = data[column].drop_nulls()
        total = len(valid)
        counts = {value: valid.to_list().count(value) for value in (-1, 0, 1)}
        rows.append(
            {
                "horizon_steps": steps,
                "horizon_milliseconds": steps * models.GRID_INTERVAL_MILLISECONDS,
                "down": counts[-1],
                "flat": counts[0],
                "up": counts[1],
                "total": total,
                "flat_percentage": (100.0 * counts[0] / total) if total else 0.0,
            }
        )
    return pl.DataFrame(rows)


def _validate_grid_timestamp(*, timestamp: datetime.datetime) -> None:
    if timestamp.microsecond % (models.GRID_INTERVAL_MILLISECONDS * 1_000):
        raise InvalidNativeStateError(
            f"timestamp is outside the 100 ms grid: {timestamp.isoformat()}"
        )


def _apply_timestamp_records(
    *,
    records: Sequence[orderbook_models.RawRecord],
    timestamp: datetime.datetime,
    instrument: orderbook_models.Instrument,
    side_states: dict[orderbook_models.QuoteSide, _SideState],
) -> None:
    quote_rows: dict[orderbook_models.QuoteSide, list[orderbook_models.RawQuote]] = {}
    for record in records:
        if record.instrument is not instrument:
            raise InvalidNativeStateError("record instrument does not match pilot")
        if isinstance(record, orderbook_models.RawQuote):
            quote_rows.setdefault(record.side, []).append(record)
    for side, rows in quote_rows.items():
        side_states[side] = _build_side_state(rows=rows, timestamp=timestamp)


def _build_side_state(
    *, rows: Sequence[orderbook_models.RawQuote], timestamp: datetime.datetime
) -> _SideState:
    if len(rows) == 1 and rows[0].price is None:
        return _SideState(timestamp=timestamp, levels=())
    expected_levels = list(range(1, len(rows) + 1))
    actual_levels = [row.level for row in rows]
    if actual_levels != expected_levels:
        raise InvalidNativeStateError(
            f"non-contiguous side snapshot at {timestamp.isoformat()}"
        )
    if any(row.price is None for row in rows):
        raise InvalidNativeStateError(
            f"partially empty side snapshot at {timestamp.isoformat()}"
        )
    return _SideState(
        timestamp=timestamp,
        levels=tuple(
            _Level(
                price=cast(float, row.price),
                size=row.size,
                order_count=row.order_count,
            )
            for row in rows
        ),
    )


def _state_row(
    *,
    timestamp: datetime.datetime,
    trading_date: datetime.date,
    instrument: orderbook_models.Instrument,
    timestamp_records: Sequence[orderbook_models.RawRecord],
    side_states: dict[orderbook_models.QuoteSide, _SideState],
    maximum_staleness_steps: int,
    maximum_depth: int,
) -> dict[str, object]:
    bid = side_states.get(orderbook_models.QuoteSide.BID)
    ask = side_states.get(orderbook_models.QuoteSide.OFFER)
    maximum_age = maximum_staleness_steps * models.GRID_INTERVAL
    bid_age = timestamp - bid.timestamp if bid is not None else None
    ask_age = timestamp - ask.timestamp if ask is not None else None
    book_observed = _book_is_observed(
        bid=bid,
        ask=ask,
        bid_age=bid_age,
        ask_age=ask_age,
        maximum_age=maximum_age,
        maximum_depth=maximum_depth,
    )
    row: dict[str, object] = {
        orderbook_models.COL_TIMESTAMP: timestamp,
        orderbook_models.COL_INSTRUMENT: instrument.value,
        orderbook_models.COL_TRADING_DATE: trading_date,
        models.COL_BID_UPDATED: any(
            isinstance(record, orderbook_models.RawQuote)
            and record.side is orderbook_models.QuoteSide.BID
            for record in timestamp_records
        ),
        models.COL_ASK_UPDATED: any(
            isinstance(record, orderbook_models.RawQuote)
            and record.side is orderbook_models.QuoteSide.OFFER
            for record in timestamp_records
        ),
        models.COL_BID_AGE_MILLISECONDS: _milliseconds(duration=bid_age),
        models.COL_ASK_AGE_MILLISECONDS: _milliseconds(duration=ask_age),
        orderbook_models.COL_BOOK_OBSERVED: book_observed,
        orderbook_models.COL_QUOTE_UPDATE_COUNT: sum(
            isinstance(record, orderbook_models.RawQuote)
            for record in timestamp_records
        ),
        orderbook_models.COL_QUOTE_AGE_SECONDS: (
            max(
                duration.total_seconds()
                for duration in (bid_age, ask_age)
                if duration is not None
            )
            if bid_age is not None or ask_age is not None
            else None
        ),
    }
    _add_levels(row=row, bid=bid, ask=ask, maximum_depth=maximum_depth)
    _add_deals(row=row, records=timestamp_records)
    if book_observed and bid is not None and ask is not None:
        best_bid = bid.levels[0].price
        best_ask = ask.levels[0].price
        mid_price = (best_bid + best_ask) / 2.0
        row[orderbook_models.COL_MID_PRICE] = mid_price
        row[orderbook_models.COL_SPREAD] = best_ask - best_bid
        bid_size = bid.levels[0].size
        ask_size = ask.levels[0].size
        row[orderbook_models.COL_QUOTE_IMBALANCE] = (
            (bid_size - ask_size) / (bid_size + ask_size)
            if bid_size + ask_size
            else 0.0
        )
    else:
        row[orderbook_models.COL_MID_PRICE] = None
        row[orderbook_models.COL_SPREAD] = None
        row[orderbook_models.COL_QUOTE_IMBALANCE] = None
    return row


def _book_is_observed(
    *,
    bid: _SideState | None,
    ask: _SideState | None,
    bid_age: datetime.timedelta | None,
    ask_age: datetime.timedelta | None,
    maximum_age: datetime.timedelta,
    maximum_depth: int,
) -> bool:
    if (
        bid is None
        or ask is None
        or not bid.levels
        or not ask.levels
        or bid_age is None
        or ask_age is None
        or bid_age > maximum_age
        or ask_age > maximum_age
        or len(bid.levels) < maximum_depth
        or len(ask.levels) < maximum_depth
    ):
        return False
    bid_prices = [level.price for level in bid.levels]
    ask_prices = [level.price for level in ask.levels]
    return (
        bid_prices[0] < ask_prices[0]
        and all(left > right for left, right in itertools.pairwise(bid_prices))
        and all(left < right for left, right in itertools.pairwise(ask_prices))
    )


def _add_levels(
    *,
    row: dict[str, object],
    bid: _SideState | None,
    ask: _SideState | None,
    maximum_depth: int,
) -> None:
    for level in range(1, maximum_depth + 1):
        bid_level = bid.levels[level - 1] if bid and len(bid.levels) >= level else None
        ask_level = ask.levels[level - 1] if ask and len(ask.levels) >= level else None
        row[orderbook_models.bid_price_col(level=level)] = (
            bid_level.price if bid_level else None
        )
        row[orderbook_models.ask_price_col(level=level)] = (
            ask_level.price if ask_level else None
        )
        row[orderbook_models.bid_size_col(level=level)] = (
            bid_level.size if bid_level else None
        )
        row[orderbook_models.ask_size_col(level=level)] = (
            ask_level.size if ask_level else None
        )
        row[orderbook_models.bid_order_count_col(level=level)] = (
            bid_level.order_count if bid_level else None
        )
        row[orderbook_models.ask_order_count_col(level=level)] = (
            ask_level.order_count if ask_level else None
        )


def _add_deals(
    *, row: dict[str, object], records: Sequence[orderbook_models.RawRecord]
) -> None:
    deals = [
        record for record in records if isinstance(record, orderbook_models.RawDeal)
    ]
    buy_volume = sum(
        deal.total_volume
        for deal in deals
        if deal.side is orderbook_models.DealSide.PAID
    )
    sell_volume = sum(
        deal.total_volume
        for deal in deals
        if deal.side is orderbook_models.DealSide.GIVEN
    )
    row[orderbook_models.COL_BUY_VOLUME] = buy_volume
    row[orderbook_models.COL_SELL_VOLUME] = sell_volume
    row[orderbook_models.COL_TRADE_COUNT] = sum(deal.deal_count for deal in deals)
    row[orderbook_models.COL_DEALS_OBSERVED] = bool(deals)
    total_volume = buy_volume + sell_volume
    row[orderbook_models.COL_DEAL_FLOW_IMBALANCE] = (
        (buy_volume - sell_volume) / total_volume if total_volume else 0.0
    )
    weighted_volume = sum(deal.extremal_price_volume for deal in deals)
    row[orderbook_models.COL_EXTREMAL_PRICE_MEAN] = (
        sum(deal.extremal_price * deal.extremal_price_volume for deal in deals)
        / weighted_volume
        if weighted_volume
        else None
    )


def _milliseconds(*, duration: datetime.timedelta | None) -> int | None:
    if duration is None:
        return None
    return round(duration.total_seconds() * 1_000)
