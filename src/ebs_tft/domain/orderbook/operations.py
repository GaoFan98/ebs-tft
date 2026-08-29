"""Reconstruct causal minute-end books from EBS time-slice records."""

from __future__ import annotations

import datetime
from collections import defaultdict
from collections.abc import Iterable, Sequence

import attrs
import polars as pl

from ebs_tft.domain.orderbook import models


class InvalidBookStateError(Exception):
    """Indicate an internally inconsistent source snapshot."""


@attrs.define
class BuildAudit:
    """Count reconstruction outcomes that are relevant to data quality."""

    quote_snapshots: int = 0
    quote_resets: int = 0
    deal_rows: int = 0
    invalid_book_bars: int = 0
    stale_book_bars: int = 0


@attrs.frozen
class _Level:
    price: float
    size: int
    order_count: int


@attrs.frozen
class _SideState:
    timestamp: datetime.datetime
    levels: dict[int, _Level]


@attrs.define
class _DealAggregate:
    buy_volume: int = 0
    sell_volume: int = 0
    trade_count: int = 0
    weighted_extremal_price: float = 0.0
    extremal_price_volume: int = 0


def build_bars(
    *,
    records: Iterable[models.RawRecord],
    instrument: models.Instrument,
    trading_date: datetime.date,
    maximum_depth: int = models.MAX_LEVELS,
    maximum_staleness_seconds: int = 60,
    audit: BuildAudit | None = None,
) -> pl.DataFrame:
    """Build regular UTC minute bars using the latest causal side snapshots.

    A Q block with one side and timestamp replaces that entire side. A null-price
    Q block clears the side. Bid and offer ages are tracked separately, so source
    records are never represented as a simultaneous source snapshot.
    """
    models.all_level_cols(max_level=maximum_depth)
    if maximum_staleness_seconds <= 0:
        raise ValueError("maximum_staleness_seconds must be positive")

    build_audit = audit if audit is not None else BuildAudit()
    side_state: dict[models.QuoteSide, _SideState] = {}
    deals: defaultdict[datetime.datetime, _DealAggregate] = defaultdict(_DealAggregate)
    quote_updates: defaultdict[datetime.datetime, int] = defaultdict(int)
    output: list[dict[str, object]] = []
    pending_quotes: list[models.RawQuote] = []
    pending_key: tuple[datetime.datetime, models.QuoteSide] | None = None
    current_minute: datetime.datetime | None = None

    def publish_until(*, minute_before: datetime.datetime) -> None:
        nonlocal current_minute
        if current_minute is None:
            current_minute = minute_before
            return
        while current_minute < minute_before:
            output.append(
                _build_bar(
                    minute=current_minute,
                    instrument=instrument,
                    trading_date=trading_date,
                    maximum_depth=maximum_depth,
                    maximum_staleness_seconds=maximum_staleness_seconds,
                    side_state=side_state,
                    deal=deals.pop(current_minute, None),
                    update_count=quote_updates.pop(current_minute, 0),
                    audit=build_audit,
                )
            )
            current_minute += datetime.timedelta(minutes=1)

    def apply_pending() -> None:
        nonlocal pending_quotes, pending_key
        if not pending_quotes or pending_key is None:
            return
        timestamp, side = pending_key
        side_state[side] = _snapshot(rows=pending_quotes, timestamp=timestamp)
        quote_updates[_minute(timestamp=timestamp)] += 1
        build_audit.quote_snapshots += 1
        if not side_state[side].levels:
            build_audit.quote_resets += 1
        pending_quotes = []
        pending_key = None

    def advance_then_apply_pending() -> None:
        if pending_key is None:
            return
        publish_until(minute_before=_minute(timestamp=pending_key[0]))
        apply_pending()

    for record in records:
        if record.instrument is not instrument:
            raise InvalidBookStateError(
                "record instrument does not match build request"
            )
        record_minute = _minute(timestamp=record.timestamp)
        if current_minute is None:
            current_minute = record_minute

        if isinstance(record, models.RawQuote):
            key = (record.timestamp, record.side)
            if pending_key is not None and key != pending_key:
                advance_then_apply_pending()
            if pending_key is None:
                publish_until(minute_before=record_minute)
            pending_key = key
            pending_quotes.append(record)
            continue

        advance_then_apply_pending()
        publish_until(minute_before=record_minute)
        aggregate = deals[record_minute]
        if record.side is models.DealSide.PAID:
            aggregate.buy_volume += record.total_volume
        else:
            aggregate.sell_volume += record.total_volume
        aggregate.trade_count += record.deal_count
        aggregate.weighted_extremal_price += (
            record.extremal_price * record.extremal_price_volume
        )
        aggregate.extremal_price_volume += record.extremal_price_volume
        build_audit.deal_rows += 1

    advance_then_apply_pending()
    if current_minute is not None:
        publish_until(minute_before=current_minute + datetime.timedelta(minutes=1))
    return _to_frame(rows=output, maximum_depth=maximum_depth)


def _snapshot(
    *, rows: Sequence[models.RawQuote], timestamp: datetime.datetime
) -> _SideState:
    levels: dict[int, _Level] = {}
    for row in rows:
        if row.level in levels:
            raise InvalidBookStateError(
                f"duplicate level {row.level} in side snapshot at "
                f"{timestamp.isoformat()}"
            )
        if row.price is not None:
            levels[row.level] = _Level(
                price=row.price,
                size=row.size,
                order_count=row.order_count,
            )
    if levels and sorted(levels) != list(range(1, max(levels) + 1)):
        raise InvalidBookStateError(
            f"non-contiguous side snapshot at {timestamp.isoformat()}"
        )
    return _SideState(timestamp=timestamp, levels=levels)


def _build_bar(
    *,
    minute: datetime.datetime,
    instrument: models.Instrument,
    trading_date: datetime.date,
    maximum_depth: int,
    maximum_staleness_seconds: int,
    side_state: dict[models.QuoteSide, _SideState],
    deal: _DealAggregate | None,
    update_count: int,
    audit: BuildAudit,
) -> dict[str, object]:
    observation_at = minute + datetime.timedelta(minutes=1)
    bid = side_state.get(models.QuoteSide.BID)
    offer = side_state.get(models.QuoteSide.OFFER)
    ages = [
        (observation_at - state.timestamp).total_seconds()
        for state in (bid, offer)
        if state is not None
    ]
    is_fresh = (
        bid is not None
        and offer is not None
        and bool(bid.levels)
        and bool(offer.levels)
        and all(level in bid.levels for level in range(1, maximum_depth + 1))
        and all(level in offer.levels for level in range(1, maximum_depth + 1))
        and all(0 <= age <= maximum_staleness_seconds for age in ages)
    )
    is_valid = is_fresh and _is_valid_book(bid=bid, offer=offer)
    if is_fresh and not is_valid:
        audit.invalid_book_bars += 1
    elif not is_fresh:
        audit.stale_book_bars += 1

    row: dict[str, object] = {
        models.COL_TIMESTAMP: minute,
        models.COL_INSTRUMENT: instrument.value,
        models.COL_TRADING_DATE: trading_date,
        models.COL_QUOTE_UPDATE_COUNT: update_count,
        models.COL_QUOTE_AGE_SECONDS: max(ages) if ages else None,
        models.COL_BOOK_OBSERVED: is_valid,
        models.COL_DEALS_OBSERVED: is_valid,
    }
    for level in range(1, maximum_depth + 1):
        bid_level = bid.levels.get(level) if is_valid and bid is not None else None
        offer_level = (
            offer.levels.get(level) if is_valid and offer is not None else None
        )
        row[models.bid_price_col(level=level)] = (
            bid_level.price if bid_level is not None else None
        )
        row[models.bid_size_col(level=level)] = (
            bid_level.size if bid_level is not None else None
        )
        row[models.ask_price_col(level=level)] = (
            offer_level.price if offer_level is not None else None
        )
        row[models.ask_size_col(level=level)] = (
            offer_level.size if offer_level is not None else None
        )
        row[models.bid_order_count_col(level=level)] = (
            bid_level.order_count if bid_level is not None else None
        )
        row[models.ask_order_count_col(level=level)] = (
            offer_level.order_count if offer_level is not None else None
        )

    bid_l1 = row[models.bid_price_col(level=1)]
    offer_l1 = row[models.ask_price_col(level=1)]
    bid_size_l1 = row[models.bid_size_col(level=1)]
    offer_size_l1 = row[models.ask_size_col(level=1)]
    if (
        isinstance(bid_l1, (int, float))
        and isinstance(offer_l1, (int, float))
        and isinstance(bid_size_l1, (int, float))
        and isinstance(offer_size_l1, (int, float))
    ):
        bid_price = float(bid_l1)
        offer_price = float(offer_l1)
        bid_size = int(bid_size_l1)
        offer_size = int(offer_size_l1)
        row[models.COL_MID_PRICE] = (bid_price + offer_price) / 2
        row[models.COL_SPREAD] = offer_price - bid_price
        row[models.COL_QUOTE_IMBALANCE] = (bid_size - offer_size) / (
            bid_size + offer_size
        )
    else:
        row[models.COL_MID_PRICE] = None
        row[models.COL_SPREAD] = None
        row[models.COL_QUOTE_IMBALANCE] = None

    if deal is None and is_valid:
        row.update(
            {
                models.COL_BUY_VOLUME: 0,
                models.COL_SELL_VOLUME: 0,
                models.COL_TRADE_COUNT: 0,
                models.COL_DEAL_FLOW_IMBALANCE: 0.0,
                models.COL_EXTREMAL_PRICE_MEAN: None,
            }
        )
    elif deal is None:
        row.update(
            {
                models.COL_BUY_VOLUME: None,
                models.COL_SELL_VOLUME: None,
                models.COL_TRADE_COUNT: None,
                models.COL_DEAL_FLOW_IMBALANCE: None,
                models.COL_EXTREMAL_PRICE_MEAN: None,
            }
        )
    else:
        total = deal.buy_volume + deal.sell_volume
        row[models.COL_DEALS_OBSERVED] = True
        row.update(
            {
                models.COL_BUY_VOLUME: deal.buy_volume,
                models.COL_SELL_VOLUME: deal.sell_volume,
                models.COL_TRADE_COUNT: deal.trade_count,
                models.COL_DEAL_FLOW_IMBALANCE: (
                    (deal.buy_volume - deal.sell_volume) / total if total else None
                ),
                models.COL_EXTREMAL_PRICE_MEAN: (
                    deal.weighted_extremal_price / deal.extremal_price_volume
                    if deal.extremal_price_volume
                    else None
                ),
            }
        )
    return row


def _is_valid_book(*, bid: _SideState | None, offer: _SideState | None) -> bool:
    if bid is None or offer is None or not bid.levels or not offer.levels:
        return False
    bid_prices = [bid.levels[level].price for level in sorted(bid.levels)]
    offer_prices = [offer.levels[level].price for level in sorted(offer.levels)]
    return (
        bid_prices[0] < offer_prices[0]
        and all(left > right for left, right in zip(bid_prices, bid_prices[1:]))
        and all(left < right for left, right in zip(offer_prices, offer_prices[1:]))
    )


def _minute(*, timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(second=0, microsecond=0)


def _to_frame(*, rows: list[dict[str, object]], maximum_depth: int) -> pl.DataFrame:
    columns = models.canonical_bar_columns(max_level=maximum_depth)
    if not rows:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(rows).select(columns).sort(models.COL_TIMESTAMP)
