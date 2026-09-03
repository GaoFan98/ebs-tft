"""Apply reusable audit, split, sampling, and paired-inference rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ebs_tft.domain.research import models


def evaluate_session_eligibility(
    *,
    duration_milliseconds: int,
    required_depth_observed_states: int,
    maximum_source_depth: int,
    parse_error: str | None,
    policy: models.AuditPolicy,
) -> tuple[bool, tuple[str, ...]]:
    """Return technical eligibility and every outcome-independent failure reason."""
    reasons: list[str] = []
    if parse_error is not None:
        reasons.append("parse_or_reconstruction_error")
    if duration_milliseconds < policy.minimum_duration_milliseconds:
        reasons.append("insufficient_duration")
    if required_depth_observed_states < policy.minimum_observed_states:
        reasons.append("insufficient_required_depth_observed_states")
    if maximum_source_depth < policy.required_depth:
        reasons.append("insufficient_source_depth")
    return not reasons, tuple(reasons)


def build_rolling_folds(
    *,
    sessions: Sequence[models.SessionIdentity],
    policy: models.SplitPolicy,
) -> tuple[models.RollingFold, ...]:
    """Return deterministic expanding-window folds from eligible development data."""
    locked = frozenset(policy.locked_evaluation_dates)
    eligible = tuple(
        sorted(
            (
                item
                for item in sessions
                if item.trading_date <= policy.development_end_date
                and item.trading_date not in locked
            ),
            key=lambda item: item.trading_date,
        )
    )
    validation_size = policy.validation_sessions_per_fold
    training_size = policy.minimum_training_sessions
    folds: list[models.RollingFold] = []
    validation_start = training_size
    while validation_start + validation_size <= len(eligible):
        validation_end = validation_start + validation_size
        folds.append(
            models.RollingFold(
                identifier=f"fold_{len(folds) + 1:02d}",
                training_sessions=eligible[:validation_start],
                validation_sessions=eligible[validation_start:validation_end],
            )
        )
        validation_start += policy.fold_step_sessions
    if not folds:
        raise ValueError("eligible sessions cannot form one complete rolling fold")
    return tuple(folds)


def training_stride_steps(
    *, protocol: models.ResearchProtocol, horizon_milliseconds: int
) -> int:
    """Return the predeclared training stride as an exact native-grid offset."""
    try:
        stride = dict(protocol.training_stride_milliseconds)[horizon_milliseconds]
    except KeyError as exc:
        raise ValueError("horizon has no predeclared training stride") from exc
    return stride // protocol.state_interval_milliseconds


def paired_session_interval(
    *,
    shallower_by_session: Mapping[str, float],
    deeper_by_session: Mapping[str, float],
    repetitions: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float | int]:
    """Return a session-block bootstrap interval for deeper-minus-shallower change."""
    if set(shallower_by_session) != set(deeper_by_session):
        raise ValueError("paired metrics must contain identical session keys")
    keys = tuple(sorted(shallower_by_session))
    if len(keys) < 2:
        raise ValueError("paired inference requires at least two sessions")
    differences = np.asarray(
        [deeper_by_session[key] - shallower_by_session[key] for key in keys],
        dtype=np.float64,
    )
    generator = np.random.default_rng(random_seed)
    selections = generator.integers(
        0, len(differences), size=(repetitions, len(differences))
    )
    bootstrap_means = differences[selections].mean(axis=1)
    alpha = 1.0 - confidence_level
    return {
        "sessions": len(keys),
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "confidence_lower": float(np.quantile(bootstrap_means, alpha / 2.0)),
        "confidence_upper": float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0)),
    }
