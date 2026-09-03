"""Test the exact-schema research-protocol configuration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from ebs_tft.application.usecases import research_protocol


class TestLoadProtocol:
    def test_returns_native_typed_research_rules(self, tmp_path: Path) -> None:
        path = tmp_path / "research.yaml"
        path.write_text(_yaml(), encoding="utf-8")

        actual = research_protocol.load_protocol(path=path)

        assert actual.development_instrument.value == "EUR_USD"
        assert actual.horizon_steps == (50, 100)
        assert actual.validation_checks_per_epoch == 4
        assert actual.audit_workers == 3
        assert dict(actual.training_stride_milliseconds) == {
            5_000: 10_000,
            10_000: 10_000,
        }

    def test_rejects_a_stride_shorter_than_the_target_and_context_floor(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "research.yaml"
        path.write_text(_yaml().replace("5000: 10000", "5000: 100"), encoding="utf-8")

        with pytest.raises(
            research_protocol.UnableToLoadResearchProtocolError,
            match="training strides",
        ):
            research_protocol.load_protocol(path=path)


def _yaml() -> str:
    return """\
schema_version: 1
data_dir: data/raw
output_dir: research_outputs
instruments: [EUR_USD, USD_JPY, EUR_JPY]
years: [2024]
state_interval_milliseconds: 100
forecast_horizons_milliseconds: [5000, 10000]
context_milliseconds: 10000
maximum_staleness_milliseconds: 60000
audit_workers: 3
training_stride_milliseconds:
  5000: 10000
  10000: 10000
evaluation_stride_milliseconds: 100
audit_policy:
  minimum_duration_milliseconds: 70000
  minimum_observed_states: 700
  required_depth: 10
  redact_locked_outcomes: true
split_policy:
  development_end_date: "2024-03-01"
  minimum_training_sessions: 20
  validation_sessions_per_fold: 5
  fold_step_sessions: 5
  locked_evaluation_dates: ["2024-03-06"]
development_instrument: EUR_USD
depths: [1, 10]
models: [deeplob_direction, tft_direction]
random_seeds: [7, 19]
validation_checks_per_epoch: 4
primary_metrics: [macro_f1, mcc]
supporting_metrics: [balanced_accuracy, log_loss, multiclass_brier]
bootstrap_repetitions: 1000
confidence_level: 0.95
"""
