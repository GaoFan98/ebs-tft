"""Define immutable values for the defensible forecasting protocol."""

from __future__ import annotations

import datetime
import enum
from pathlib import Path

import attrs

from ebs_tft.domain.orderbook import models as orderbook_models


class EvaluationMetric(enum.StrEnum):
    """Identify one predeclared direction-evaluation metric."""

    MACRO_F1 = "macro_f1"
    MCC = "mcc"
    BALANCED_ACCURACY = "balanced_accuracy"
    LOG_LOSS = "log_loss"
    MULTICLASS_BRIER = "multiclass_brier"


@attrs.frozen
class AuditPolicy:
    """Define technical, outcome-independent session eligibility rules."""

    minimum_duration_milliseconds: int
    minimum_observed_states: int
    required_depth: int
    redact_locked_outcomes: bool

    def __attrs_post_init__(self) -> None:
        """Reject eligibility rules that could silently select unusable data."""
        if self.minimum_duration_milliseconds <= 0:
            raise ValueError("minimum_duration_milliseconds must be positive")
        if self.minimum_observed_states <= 0:
            raise ValueError("minimum_observed_states must be positive")
        orderbook_models.all_level_cols(max_level=self.required_depth)
        if not self.redact_locked_outcomes:
            raise ValueError("locked evaluation outcomes must remain redacted")


@attrs.frozen
class SplitPolicy:
    """Define chronological development folds and locked evaluation dates."""

    development_end_date: datetime.date
    minimum_training_sessions: int
    validation_sessions_per_fold: int
    fold_step_sessions: int
    locked_evaluation_dates: tuple[datetime.date, ...]

    def __attrs_post_init__(self) -> None:
        """Reject ambiguous or overlapping chronological split rules."""
        positive_values = (
            self.minimum_training_sessions,
            self.validation_sessions_per_fold,
            self.fold_step_sessions,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive_values):
            raise ValueError("session counts and fold step must be positive")
        if not self.locked_evaluation_dates:
            raise ValueError("locked_evaluation_dates must be non-empty")
        if tuple(sorted(self.locked_evaluation_dates)) != self.locked_evaluation_dates:
            raise ValueError("locked_evaluation_dates must be chronological")
        if len(set(self.locked_evaluation_dates)) != len(self.locked_evaluation_dates):
            raise ValueError("locked_evaluation_dates must be unique")
        if self.fold_step_sessions < self.validation_sessions_per_fold:
            raise ValueError("fold step must keep validation session blocks disjoint")


@attrs.frozen
class ResearchProtocol:
    """Configure audit, splitting, sampling, and finite model gates."""

    data_dir: Path
    output_dir: Path
    instruments: tuple[orderbook_models.Instrument, ...]
    years: tuple[int, ...]
    state_interval_milliseconds: int
    forecast_horizons_milliseconds: tuple[int, ...]
    context_milliseconds: int
    maximum_staleness_milliseconds: int
    audit_workers: int
    training_stride_milliseconds: tuple[tuple[int, int], ...]
    evaluation_stride_milliseconds: int
    audit_policy: AuditPolicy
    split_policy: SplitPolicy
    development_instrument: orderbook_models.Instrument
    depths: tuple[int, ...]
    models: tuple[str, ...]
    random_seeds: tuple[int, ...]
    validation_checks_per_epoch: int
    primary_metrics: tuple[EvaluationMetric, ...]
    supporting_metrics: tuple[EvaluationMetric, ...]
    bootstrap_repetitions: int
    confidence_level: float

    def __attrs_post_init__(self) -> None:
        """Reject protocols that permit leakage, aggregation, or moving gates."""
        if not self.instruments or len(set(self.instruments)) != len(self.instruments):
            raise ValueError("instruments must be non-empty and unique")
        if self.development_instrument not in self.instruments:
            raise ValueError("development_instrument must be an audited instrument")
        if not self.years or len(set(self.years)) != len(self.years):
            raise ValueError("years must be non-empty and unique")
        if any(year <= 0 for year in self.years):
            raise ValueError("years must be positive")
        if self.state_interval_milliseconds != 100:
            raise ValueError("state_interval_milliseconds must preserve native 100 ms")
        if (
            not self.forecast_horizons_milliseconds
            or len(set(self.forecast_horizons_milliseconds))
            != len(self.forecast_horizons_milliseconds)
            or any(value <= 0 for value in self.forecast_horizons_milliseconds)
        ):
            raise ValueError("forecast horizons must be unique positive values")
        duration_values = (
            *self.forecast_horizons_milliseconds,
            self.context_milliseconds,
            self.maximum_staleness_milliseconds,
            self.evaluation_stride_milliseconds,
        )
        if isinstance(self.audit_workers, bool) or not 1 <= self.audit_workers <= 8:
            raise ValueError("audit_workers must be between one and eight")
        if any(
            value <= 0 or value % self.state_interval_milliseconds
            for value in duration_values
        ):
            raise ValueError("durations must be positive native-grid multiples")
        stride_mapping = dict(self.training_stride_milliseconds)
        if set(stride_mapping) != set(self.forecast_horizons_milliseconds):
            raise ValueError("training strides must map every forecast horizon exactly")
        if len(stride_mapping) != len(self.training_stride_milliseconds):
            raise ValueError("training stride horizons must be unique")
        for horizon, stride in stride_mapping.items():
            minimum_stride = min(self.context_milliseconds, horizon)
            if stride < minimum_stride or stride % self.state_interval_milliseconds:
                raise ValueError(
                    "training strides must be native-grid multiples no shorter than "
                    "the smaller of context and horizon"
                )
        if not self.depths or len(set(self.depths)) != len(self.depths):
            raise ValueError("depths must be non-empty and unique")
        orderbook_models.all_level_cols(max_level=max(self.depths))
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        if (
            not self.random_seeds
            or len(set(self.random_seeds)) != len(self.random_seeds)
            or any(seed < 0 for seed in self.random_seeds)
        ):
            raise ValueError("random_seeds must be unique non-negative values")
        if self.validation_checks_per_epoch < 2:
            raise ValueError("validation_checks_per_epoch must be at least two")
        if not self.primary_metrics or len(set(self.primary_metrics)) != len(
            self.primary_metrics
        ):
            raise ValueError("primary_metrics must be non-empty and unique")
        if set(self.primary_metrics) & set(self.supporting_metrics):
            raise ValueError("primary and supporting metrics must not overlap")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("bootstrap_repetitions must be at least 1000")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        """Return target horizons as native-grid offsets."""
        return tuple(
            value // self.state_interval_milliseconds
            for value in self.forecast_horizons_milliseconds
        )


@attrs.frozen
class SessionIdentity:
    """Identify one audited source without carrying mutable table data."""

    instrument: orderbook_models.Instrument
    trading_date: datetime.date
    path: Path
    sha256: str


@attrs.frozen
class RollingFold:
    """Hold one expanding-window development split."""

    identifier: str
    training_sessions: tuple[SessionIdentity, ...]
    validation_sessions: tuple[SessionIdentity, ...]

    def __attrs_post_init__(self) -> None:
        """Reject a fold whose validation does not strictly follow training."""
        if not self.training_sessions or not self.validation_sessions:
            raise ValueError("fold sessions must be non-empty")
        training_dates = tuple(item.trading_date for item in self.training_sessions)
        validation_dates = tuple(item.trading_date for item in self.validation_sessions)
        if tuple(sorted(training_dates)) != training_dates:
            raise ValueError("training sessions must be chronological")
        if tuple(sorted(validation_dates)) != validation_dates:
            raise ValueError("validation sessions must be chronological")
        if training_dates[-1] >= validation_dates[0]:
            raise ValueError("validation sessions must strictly follow training")
