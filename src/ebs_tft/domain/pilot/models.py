"""Define immutable values for the native-resolution local pilot."""

from __future__ import annotations

import datetime
import enum
from pathlib import Path

import attrs

from ebs_tft.domain.orderbook import models as orderbook_models

GRID_INTERVAL: datetime.timedelta = datetime.timedelta(milliseconds=100)
GRID_INTERVAL_MILLISECONDS: int = 100

COL_BID_UPDATED: str = "bid_updated"
COL_ASK_UPDATED: str = "ask_updated"
COL_BID_AGE_MILLISECONDS: str = "bid_age_ms"
COL_ASK_AGE_MILLISECONDS: str = "ask_age_ms"


class ModelName(enum.StrEnum):
    """Identify one approved native-state direction adapter."""

    DEEP_LOB = "deeplob_direction"
    TFT = "tft_direction"


@attrs.frozen
class PilotSession:
    """Identify one bounded native-data session."""

    raw_path: Path
    trading_date: datetime.date
    grid_steps: int

    def __attrs_post_init__(self) -> None:
        """Reject a session that cannot contain a positive native-state range."""
        if isinstance(self.grid_steps, bool) or self.grid_steps <= 0:
            raise ValueError("session grid_steps must be positive")


def direction_column(*, horizon_steps: int) -> str:
    """Return the target column for one native-grid forecast horizon."""
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    return f"direction_h{horizon_steps}"


@attrs.frozen
class PilotSpecification:
    """Configure one explicitly bounded, non-publication local pilot."""

    raw_path: Path
    instrument: orderbook_models.Instrument
    trading_date: datetime.date
    output_dir: Path
    grid_steps: int
    state_interval_milliseconds: int
    forecast_horizons_milliseconds: tuple[int, ...]
    modeled_horizons_milliseconds: tuple[int, ...]
    class_weighted_horizons_milliseconds: tuple[int, ...]
    context_milliseconds: int
    maximum_staleness_milliseconds: int
    depths: tuple[int, ...]
    models: tuple[ModelName, ...]
    maximum_training_windows: int | None
    maximum_validation_windows: int | None
    maximum_test_windows: int | None
    maximum_epochs: int
    early_stopping_patience: int
    early_stopping_minimum_delta: float
    gradient_clip_norm: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    random_seeds: tuple[int, ...]
    device: str

    def __attrs_post_init__(self) -> None:
        """Reject configurations that would make the pilot ambiguous or invalid."""
        positive_values = {
            "grid_steps": self.grid_steps,
            "state_interval_milliseconds": self.state_interval_milliseconds,
            "context_milliseconds": self.context_milliseconds,
            "maximum_staleness_milliseconds": self.maximum_staleness_milliseconds,
            "maximum_epochs": self.maximum_epochs,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
        }
        for name, value in positive_values.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.state_interval_milliseconds != GRID_INTERVAL_MILLISECONDS:
            raise ValueError("state_interval_milliseconds must preserve native 100 ms")
        if not self.forecast_horizons_milliseconds or any(
            item <= 0 for item in self.forecast_horizons_milliseconds
        ):
            raise ValueError(
                "forecast_horizons_milliseconds must contain positive values"
            )
        if len(set(self.forecast_horizons_milliseconds)) != len(
            self.forecast_horizons_milliseconds
        ):
            raise ValueError("forecast_horizons_milliseconds must be unique")
        durations = (
            *self.forecast_horizons_milliseconds,
            self.context_milliseconds,
            self.maximum_staleness_milliseconds,
        )
        if any(item % self.state_interval_milliseconds for item in durations):
            raise ValueError("durations must align to the native state interval")
        if (
            not self.modeled_horizons_milliseconds
            or len(set(self.modeled_horizons_milliseconds))
            != len(self.modeled_horizons_milliseconds)
            or not set(self.modeled_horizons_milliseconds).issubset(
                self.forecast_horizons_milliseconds
            )
        ):
            raise ValueError(
                "modeled_horizons_milliseconds must be unique forecast horizons"
            )
        if len(set(self.class_weighted_horizons_milliseconds)) != len(
            self.class_weighted_horizons_milliseconds
        ) or not set(self.class_weighted_horizons_milliseconds).issubset(
            self.modeled_horizons_milliseconds
        ):
            raise ValueError(
                "class_weighted_horizons_milliseconds must be modeled horizons"
            )
        if not self.depths or len(set(self.depths)) != len(self.depths):
            raise ValueError("depths must be non-empty and unique")
        orderbook_models.all_level_cols(max_level=max(self.depths))
        if min(self.depths) < 1:
            raise ValueError("depths must be positive")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        window_limits = (
            self.maximum_training_windows,
            self.maximum_validation_windows,
            self.maximum_test_windows,
        )
        if any(item is not None and item <= 0 for item in window_limits):
            raise ValueError("maximum window values must be positive or null")
        if self.grid_steps <= self.context_steps + max(self.horizon_steps):
            raise ValueError("grid_steps is too short for context and horizon")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_minimum_delta < 0:
            raise ValueError("early_stopping_minimum_delta must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if (
            not self.random_seeds
            or len(set(self.random_seeds)) != len(self.random_seeds)
            or any(seed < 0 for seed in self.random_seeds)
        ):
            raise ValueError("random_seeds must be unique non-negative values")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        """Return forecast horizons as exact native-state offsets."""
        return tuple(
            item // self.state_interval_milliseconds
            for item in self.forecast_horizons_milliseconds
        )

    @property
    def modeled_horizon_steps(self) -> tuple[int, ...]:
        """Return modeled horizons as exact native-state offsets."""
        return tuple(
            item // self.state_interval_milliseconds
            for item in self.modeled_horizons_milliseconds
        )

    @property
    def class_weighted_horizon_steps(self) -> tuple[int, ...]:
        """Return weighted-sensitivity horizons as native-state offsets."""
        return tuple(
            item // self.state_interval_milliseconds
            for item in self.class_weighted_horizons_milliseconds
        )

    @property
    def context_steps(self) -> int:
        """Return causal context length in native-state offsets."""
        return self.context_milliseconds // self.state_interval_milliseconds

    @property
    def maximum_staleness_steps(self) -> int:
        """Return maximum quote age in native-state offsets."""
        return self.maximum_staleness_milliseconds // self.state_interval_milliseconds


@attrs.frozen
class MultiSessionSpecification:
    """Configure day-aware training and later-session development validation."""

    training_sessions: tuple[PilotSession, ...]
    validation_session: PilotSession
    instrument: orderbook_models.Instrument
    output_dir: Path
    state_interval_milliseconds: int
    forecast_horizons_milliseconds: tuple[int, ...]
    modeled_horizons_milliseconds: tuple[int, ...]
    context_milliseconds: int
    maximum_staleness_milliseconds: int
    depths: tuple[int, ...]
    models: tuple[ModelName, ...]
    maximum_training_windows: int | None
    maximum_validation_windows: int | None
    maximum_epochs: int
    early_stopping_patience: int
    early_stopping_minimum_delta: float
    gradient_clip_norm: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    random_seeds: tuple[int, ...]
    device: str

    def __attrs_post_init__(self) -> None:
        """Reject temporal leakage and ambiguous multi-session protocols."""
        if not self.training_sessions:
            raise ValueError("training_sessions must be non-empty")
        training_dates = tuple(item.trading_date for item in self.training_sessions)
        if len(set(training_dates)) != len(training_dates):
            raise ValueError("training session dates must be unique")
        if tuple(sorted(training_dates)) != training_dates:
            raise ValueError("training sessions must be ordered chronologically")
        if any(item >= self.validation_session.trading_date for item in training_dates):
            raise ValueError("training sessions must precede validation")
        if len({item.raw_path for item in self.all_sessions}) != len(self.all_sessions):
            raise ValueError("session raw paths must be unique")
        positive_values = {
            "state_interval_milliseconds": self.state_interval_milliseconds,
            "context_milliseconds": self.context_milliseconds,
            "maximum_staleness_milliseconds": self.maximum_staleness_milliseconds,
            "maximum_epochs": self.maximum_epochs,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
        }
        for name, value in positive_values.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.state_interval_milliseconds != GRID_INTERVAL_MILLISECONDS:
            raise ValueError("state_interval_milliseconds must preserve native 100 ms")
        if (
            not self.forecast_horizons_milliseconds
            or len(set(self.forecast_horizons_milliseconds))
            != len(self.forecast_horizons_milliseconds)
            or any(item <= 0 for item in self.forecast_horizons_milliseconds)
        ):
            raise ValueError("forecast horizons must be unique positive values")
        durations = (
            *self.forecast_horizons_milliseconds,
            self.context_milliseconds,
            self.maximum_staleness_milliseconds,
        )
        if any(item % self.state_interval_milliseconds for item in durations):
            raise ValueError("durations must align to the native state interval")
        if (
            not self.modeled_horizons_milliseconds
            or len(set(self.modeled_horizons_milliseconds))
            != len(self.modeled_horizons_milliseconds)
            or not set(self.modeled_horizons_milliseconds).issubset(
                self.forecast_horizons_milliseconds
            )
        ):
            raise ValueError("modeled horizons must be unique forecast horizons")
        if not self.depths or len(set(self.depths)) != len(self.depths):
            raise ValueError("depths must be non-empty and unique")
        if min(self.depths) < 1:
            raise ValueError("depths must be positive")
        orderbook_models.all_level_cols(max_level=max(self.depths))
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        window_limits = (
            self.maximum_training_windows,
            self.maximum_validation_windows,
        )
        if any(item is not None and item <= 0 for item in window_limits):
            raise ValueError("maximum window values must be positive or null")
        minimum_steps = self.context_steps + max(self.horizon_steps)
        if any(item.grid_steps <= minimum_steps for item in self.all_sessions):
            raise ValueError("every session must be longer than context plus horizon")
        if self.maximum_epochs <= 0 or self.early_stopping_patience <= 0:
            raise ValueError("epoch and patience values must be positive")
        if self.early_stopping_minimum_delta < 0:
            raise ValueError("early_stopping_minimum_delta must be non-negative")
        if self.gradient_clip_norm <= 0 or self.learning_rate <= 0:
            raise ValueError("gradient_clip_norm and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if (
            not self.random_seeds
            or len(set(self.random_seeds)) != len(self.random_seeds)
            or any(seed < 0 for seed in self.random_seeds)
        ):
            raise ValueError("random_seeds must be unique non-negative values")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")

    @property
    def all_sessions(self) -> tuple[PilotSession, ...]:
        """Return training sessions followed by validation."""
        return (*self.training_sessions, self.validation_session)

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        """Return every target horizon as a native-state offset."""
        return tuple(
            item // self.state_interval_milliseconds
            for item in self.forecast_horizons_milliseconds
        )

    @property
    def modeled_horizon_steps(self) -> tuple[int, ...]:
        """Return modeled target horizons as native-state offsets."""
        return tuple(
            item // self.state_interval_milliseconds
            for item in self.modeled_horizons_milliseconds
        )

    @property
    def context_steps(self) -> int:
        """Return causal context length in native-state offsets."""
        return self.context_milliseconds // self.state_interval_milliseconds

    @property
    def maximum_staleness_steps(self) -> int:
        """Return maximum quote age in native-state offsets."""
        return self.maximum_staleness_milliseconds // self.state_interval_milliseconds
