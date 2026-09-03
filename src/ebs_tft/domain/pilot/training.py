"""Prepare native session tensors without crossing chronological boundaries."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import cast

import attrs
import numpy as np
import polars as pl

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models as pilot_models

LOB_FEATURE_ORDER: tuple[str, ...] = (
    "bid_price_basis_points_from_mid",
    "ask_price_basis_points_from_mid",
    "log_bid_size",
    "log_ask_size",
    "log_bid_order_count",
    "log_ask_order_count",
)
AUXILIARY_FEATURE_ORDER: tuple[str, ...] = (
    "log_buy_volume",
    "log_sell_volume",
    "log_trade_count",
    "deal_flow_imbalance",
    "deals_observed",
    "bid_updated",
    "ask_updated",
    "bid_age_milliseconds",
    "ask_age_milliseconds",
    "spread_basis_points",
)


@attrs.frozen
class RawSessionData:
    """Hold unscaled features and labels for one complete session."""

    trading_date: datetime.date
    lob_features: np.ndarray
    auxiliary_features: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    mid_prices: np.ndarray
    observed: np.ndarray


@attrs.frozen
class FeatureScaler:
    """Hold training-session moments for deterministic feature scaling."""

    lob_means: np.ndarray
    lob_standard_deviations: np.ndarray
    auxiliary_means: np.ndarray
    auxiliary_standard_deviations: np.ndarray


@attrs.frozen
class SessionWindowSummary:
    """Describe valid model windows selected from one session."""

    trading_date: datetime.date
    candidates: int
    selected: int
    stride_steps: int
    timestamp_from: datetime.datetime
    timestamp_to: datetime.datetime


@attrs.frozen
class PreparedCorpus:
    """Hold scaled, concatenated sessions and boundary-safe target indices."""

    lob_features: np.ndarray
    auxiliary_features: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    mid_prices: np.ndarray
    target_indices: np.ndarray
    session_offsets: tuple[int, ...]
    session_lengths: tuple[int, ...]
    session_windows: tuple[SessionWindowSummary, ...]


def extract_session(
    *, data: pl.DataFrame, trading_date: datetime.date, depth: int, horizon_steps: int
) -> RawSessionData:
    """Return raw model arrays for one session and one target horizon."""
    row_count = data.height
    mid_prices = data[orderbook_models.COL_MID_PRICE].fill_null(float("nan")).to_numpy()
    lob_features = np.zeros(
        (row_count, depth, len(LOB_FEATURE_ORDER)), dtype=np.float32
    )
    valid_mid = np.isfinite(mid_prices) & (mid_prices != 0)
    for level in range(1, depth + 1):
        level_index = level - 1
        bid_price = _numeric_column(
            data=data, name=orderbook_models.bid_price_col(level=level)
        )
        ask_price = _numeric_column(
            data=data, name=orderbook_models.ask_price_col(level=level)
        )
        lob_features[valid_mid, level_index, 0] = (
            (bid_price[valid_mid] / mid_prices[valid_mid]) - 1.0
        ) * 10_000
        lob_features[valid_mid, level_index, 1] = (
            (ask_price[valid_mid] / mid_prices[valid_mid]) - 1.0
        ) * 10_000
        lob_features[:, level_index, 2] = np.log1p(
            _numeric_column(data=data, name=orderbook_models.bid_size_col(level=level))
        )
        lob_features[:, level_index, 3] = np.log1p(
            _numeric_column(data=data, name=orderbook_models.ask_size_col(level=level))
        )
        lob_features[:, level_index, 4] = np.log1p(
            _numeric_column(
                data=data,
                name=orderbook_models.bid_order_count_col(level=level),
            )
        )
        lob_features[:, level_index, 5] = np.log1p(
            _numeric_column(
                data=data,
                name=orderbook_models.ask_order_count_col(level=level),
            )
        )
    auxiliary_features = np.column_stack(
        (
            np.log1p(_numeric_column(data=data, name=orderbook_models.COL_BUY_VOLUME)),
            np.log1p(_numeric_column(data=data, name=orderbook_models.COL_SELL_VOLUME)),
            np.log1p(_numeric_column(data=data, name=orderbook_models.COL_TRADE_COUNT)),
            _numeric_column(data=data, name=orderbook_models.COL_DEAL_FLOW_IMBALANCE),
            _numeric_column(data=data, name=orderbook_models.COL_DEALS_OBSERVED),
            _numeric_column(data=data, name=pilot_models.COL_BID_UPDATED),
            _numeric_column(data=data, name=pilot_models.COL_ASK_UPDATED),
            _numeric_column(data=data, name=pilot_models.COL_BID_AGE_MILLISECONDS),
            _numeric_column(data=data, name=pilot_models.COL_ASK_AGE_MILLISECONDS),
            np.nan_to_num(
                _numeric_column(data=data, name=orderbook_models.COL_SPREAD)
                / mid_prices
                * 10_000,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )
    ).astype(np.float32)
    target_column = pilot_models.direction_column(horizon_steps=horizon_steps)
    raw_labels = data[target_column].fill_null(-99).to_numpy()
    labels = np.where(raw_labels == -99, -1, raw_labels + 1).astype(np.int64)
    return RawSessionData(
        trading_date=trading_date,
        lob_features=lob_features,
        auxiliary_features=auxiliary_features,
        labels=labels,
        timestamps=data[orderbook_models.COL_TIMESTAMP].to_numpy(),
        mid_prices=mid_prices,
        observed=data[orderbook_models.COL_BOOK_OBSERVED].to_numpy(),
    )


def fit_feature_scaler(*, sessions: Sequence[RawSessionData]) -> FeatureScaler:
    """Return feature moments fitted exclusively on the supplied sessions."""
    if not sessions:
        raise ValueError("at least one training session is required")
    lob_count = 0
    lob_sum = np.zeros(len(LOB_FEATURE_ORDER), dtype=np.float64)
    lob_square_sum = np.zeros(len(LOB_FEATURE_ORDER), dtype=np.float64)
    auxiliary_count = 0
    auxiliary_sum = np.zeros(len(AUXILIARY_FEATURE_ORDER), dtype=np.float64)
    auxiliary_square_sum = np.zeros(len(AUXILIARY_FEATURE_ORDER), dtype=np.float64)
    for session in sessions:
        active_lob = session.lob_features[session.observed].reshape(
            -1, len(LOB_FEATURE_ORDER)
        )
        active_auxiliary = session.auxiliary_features[session.observed]
        lob_count += len(active_lob)
        lob_sum += active_lob.sum(axis=0, dtype=np.float64)
        lob_square_sum += np.square(active_lob, dtype=np.float64).sum(axis=0)
        auxiliary_count += len(active_auxiliary)
        auxiliary_sum += active_auxiliary.sum(axis=0, dtype=np.float64)
        auxiliary_square_sum += np.square(active_auxiliary, dtype=np.float64).sum(
            axis=0
        )
    if lob_count == 0 or auxiliary_count == 0:
        raise ValueError("training sessions contain no observed book states")
    lob_means = lob_sum / lob_count
    auxiliary_means = auxiliary_sum / auxiliary_count
    lob_deviations = np.sqrt(
        np.maximum((lob_square_sum / lob_count) - np.square(lob_means), 0.0)
    )
    auxiliary_deviations = np.sqrt(
        np.maximum(
            (auxiliary_square_sum / auxiliary_count) - np.square(auxiliary_means),
            0.0,
        )
    )
    lob_deviations[lob_deviations < 1e-6] = 1.0
    auxiliary_deviations[auxiliary_deviations < 1e-6] = 1.0
    return FeatureScaler(
        lob_means=lob_means.reshape(1, 1, -1).astype(np.float32),
        lob_standard_deviations=lob_deviations.reshape(1, 1, -1).astype(np.float32),
        auxiliary_means=auxiliary_means.reshape(1, -1).astype(np.float32),
        auxiliary_standard_deviations=auxiliary_deviations.reshape(1, -1).astype(
            np.float32
        ),
    )


def apply_feature_scaler(
    *, session: RawSessionData, scaler: FeatureScaler
) -> RawSessionData:
    """Return one session standardized with training-only moments."""
    return attrs.evolve(
        session,
        lob_features=(
            (session.lob_features - scaler.lob_means) / scaler.lob_standard_deviations
        ).astype(np.float32),
        auxiliary_features=(
            (session.auxiliary_features - scaler.auxiliary_means)
            / scaler.auxiliary_standard_deviations
        ).astype(np.float32),
    )


def combine_sessions(
    *,
    sessions: Sequence[RawSessionData],
    context_steps: int,
    horizon_steps: int,
    maximum_windows: int | None,
    stride_steps: int = 1,
) -> PreparedCorpus:
    """Concatenate sessions while selecting windows inside each boundary."""
    if not sessions:
        raise ValueError("at least one session is required")
    if isinstance(stride_steps, bool) or stride_steps <= 0:
        raise ValueError("stride_steps must be a positive integer")
    raw_local_candidates = tuple(
        _candidate_indices(
            session=session,
            context_steps=context_steps,
            horizon_steps=horizon_steps,
        )
        for session in sessions
    )
    local_candidates = tuple(
        candidates[::stride_steps] for candidates in raw_local_candidates
    )
    offsets: list[int] = []
    offset = 0
    shifted_candidates: list[np.ndarray] = []
    for session, candidates in zip(sessions, local_candidates, strict=True):
        offsets.append(offset)
        shifted_candidates.append(candidates + offset)
        offset += len(session.labels)
    all_candidates = np.concatenate(shifted_candidates)
    selected = _bounded_indices(indices=all_candidates, maximum=maximum_windows)
    session_windows: list[SessionWindowSummary] = []
    for session, candidates, raw_candidates, session_offset in zip(
        sessions, local_candidates, raw_local_candidates, offsets, strict=True
    ):
        upper = session_offset + len(session.labels)
        selected_count = int(((selected >= session_offset) & (selected < upper)).sum())
        session_windows.append(
            SessionWindowSummary(
                trading_date=session.trading_date,
                candidates=len(raw_candidates),
                selected=selected_count,
                stride_steps=stride_steps,
                timestamp_from=_datetime_at(
                    timestamps=session.timestamps, index=int(candidates[0])
                ),
                timestamp_to=_datetime_at(
                    timestamps=session.timestamps, index=int(candidates[-1])
                ),
            )
        )
    return PreparedCorpus(
        lob_features=np.concatenate([item.lob_features for item in sessions]),
        auxiliary_features=np.concatenate(
            [item.auxiliary_features for item in sessions]
        ),
        labels=np.concatenate([item.labels for item in sessions]),
        timestamps=np.concatenate([item.timestamps for item in sessions]),
        mid_prices=np.concatenate([item.mid_prices for item in sessions]),
        target_indices=selected,
        session_offsets=tuple(offsets),
        session_lengths=tuple(len(item.labels) for item in sessions),
        session_windows=tuple(session_windows),
    )


def _candidate_indices(
    *, session: RawSessionData, context_steps: int, horizon_steps: int
) -> np.ndarray:
    candidates: list[int] = []
    for target_index in range(context_steps - 1, len(session.labels) - horizon_steps):
        start = target_index - context_steps + 1
        if session.labels[target_index] >= 0 and bool(
            session.observed[start : target_index + 1].all()
        ):
            candidates.append(target_index)
    if not candidates:
        raise ValueError(f"session {session.trading_date} contains no valid windows")
    return np.asarray(candidates, dtype=np.int64)


def _bounded_indices(*, indices: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[positions]


def _numeric_column(*, data: pl.DataFrame, name: str) -> np.ndarray:
    return data[name].cast(pl.Float64).fill_null(0.0).to_numpy()


def _datetime_at(*, timestamps: np.ndarray, index: int) -> datetime.datetime:
    value = timestamps[index].astype("datetime64[us]").astype(datetime.datetime)
    return cast(datetime.datetime, value)
