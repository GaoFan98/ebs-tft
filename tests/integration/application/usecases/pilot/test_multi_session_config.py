"""Test the exact-schema multi-session configuration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from ebs_tft.application.usecases import pilot


class TestLoadSpecification:
    def test_returns_chronologically_typed_sessions(self, tmp_path: Path) -> None:
        path = tmp_path / "multi.yaml"
        path.write_text(_yaml(), encoding="utf-8")

        actual = pilot.load_multi_session_specification(path=path)

        assert tuple(
            item.trading_date.isoformat() for item in actual.training_sessions
        ) == ("2024-01-03", "2024-02-01")
        assert actual.validation_session.trading_date.isoformat() == "2024-03-01"
        assert actual.horizon_steps == (50,)
        assert actual.training_sessions[0].raw_path == (tmp_path / "jan.csv.gz")

    def test_rejects_a_validation_date_before_training(self, tmp_path: Path) -> None:
        path = tmp_path / "multi.yaml"
        path.write_text(
            _yaml().replace('trading_date: "2024-03-01"', 'trading_date: "2024-01-01"'),
            encoding="utf-8",
        )

        with pytest.raises(
            pilot.UnableToLoadMultiSessionConfigError,
            match="must precede validation",
        ):
            pilot.load_multi_session_specification(path=path)

    def test_rejects_an_unknown_top_level_key(self, tmp_path: Path) -> None:
        path = tmp_path / "multi.yaml"
        path.write_text(_yaml() + "unknown: true\n", encoding="utf-8")

        with pytest.raises(
            pilot.UnableToLoadMultiSessionConfigError,
            match="keys differ",
        ):
            pilot.load_multi_session_specification(path=path)


def _yaml() -> str:
    return """\
schema_version: 1
training_sessions:
  - raw_path: jan.csv.gz
    trading_date: "2024-01-03"
    grid_steps: 1000
  - raw_path: feb.csv.gz
    trading_date: "2024-02-01"
    grid_steps: 1000
validation_session:
  raw_path: mar.csv.gz
  trading_date: "2024-03-01"
  grid_steps: 1000
instrument: EUR_USD
output_dir: multi_outputs
state_interval_milliseconds: 100
forecast_horizons_milliseconds: [5000]
modeled_horizons_milliseconds: [5000]
context_milliseconds: 10000
maximum_staleness_milliseconds: 60000
depths: [1, 10]
models: [deeplob_direction, tft_direction]
maximum_training_windows: null
maximum_validation_windows: null
maximum_epochs: 10
early_stopping_patience: 2
early_stopping_minimum_delta: 0.0001
gradient_clip_norm: 1.0
batch_size: 64
learning_rate: 0.0003
weight_decay: 0.0001
hidden_size: 32
random_seeds: [7, 19]
device: cpu
"""
