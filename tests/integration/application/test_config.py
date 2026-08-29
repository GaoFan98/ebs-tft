"""
Test project configuration loading through real YAML files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ebs_tft.application import config
from ebs_tft.domain.orderbook import models

_INSTRUMENTS_YAML = """\
schema_version: 1
instruments:
  - EUR_USD
  - EUR_JPY
  - USD_JPY
maximum_depth: 3
"""

_TRAINING_YAML = """\
schema_version: 1
raw_data_dir: data/raw
processed_data_dir: data/processed
bar_frequency: 1m
forecast_horizons_minutes: []
source_timezone: null
session_calendar: null
maximum_quote_staleness_seconds: null
flat_target_policy: null
random_seeds: []
"""

_MODEL_DEFAULTS_YAML = """\
schema_version: 1
engine: null
"""


class TestLoadProjectConfig:
    def test_load_project_config_returns_one_typed_configuration(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(parent=tmp_path)

        actual = config.load_project_config(config_dir=config_dir)

        assert actual.instruments.instruments == (
            models.Instrument.EUR_USD,
            models.Instrument.EUR_JPY,
            models.Instrument.USD_JPY,
        )
        assert actual.instruments.depths == (1, 2, 3)
        assert actual.training.raw_data_dir == tmp_path / "data/raw"
        assert actual.training.processed_data_dir == tmp_path / "data/processed"
        assert actual.training.is_research_ready is False
        assert actual.model_defaults.engine is None

    def test_load_project_config_rejects_an_unknown_key(self, tmp_path: Path) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            instruments_yaml=_INSTRUMENTS_YAML + "unexpected: true\n",
        )

        with pytest.raises(config.InvalidConfigError, match="Unknown instruments"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_rejects_duplicate_instruments(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            instruments_yaml=_INSTRUMENTS_YAML.replace(
                "  - USD_JPY\n", "  - EUR_USD\n"
            ),
        )

        with pytest.raises(config.InvalidConfigError, match="duplicates"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_rejects_an_unknown_instrument(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            instruments_yaml=_INSTRUMENTS_YAML.replace("EUR_USD", "GBP_USD"),
        )

        with pytest.raises(config.InvalidConfigError, match="Unsupported instrument"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_rejects_depth_above_feed_maximum(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            instruments_yaml=_INSTRUMENTS_YAML.replace(
                "maximum_depth: 3", "maximum_depth: 11"
            ),
        )

        with pytest.raises(config.InvalidConfigError, match="maximum_depth"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_rejects_an_unsupported_frequency(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            training_yaml=_TRAINING_YAML.replace(
                "bar_frequency: 1m", "bar_frequency: 100ms"
            ),
        )

        with pytest.raises(config.InvalidConfigError, match="bar_frequency"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_rejects_a_non_positive_horizon(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            training_yaml=_TRAINING_YAML.replace(
                "forecast_horizons_minutes: []",
                "forecast_horizons_minutes: [0]",
            ),
        )

        with pytest.raises(config.InvalidConfigError, match="must be positive"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_rejects_an_unsupported_schema_version(
        self, tmp_path: Path
    ) -> None:
        config_dir = _write_config_dir(
            parent=tmp_path,
            model_defaults_yaml=_MODEL_DEFAULTS_YAML.replace(
                "schema_version: 1", "schema_version: 2"
            ),
        )

        with pytest.raises(config.InvalidConfigError, match="schema_version"):
            config.load_project_config(config_dir=config_dir)

    def test_load_project_config_reports_a_missing_file(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        with pytest.raises(config.UnableToLoadConfigError):
            config.load_project_config(config_dir=config_dir)


def _write_config_dir(
    *,
    parent: Path,
    instruments_yaml: str = _INSTRUMENTS_YAML,
    training_yaml: str = _TRAINING_YAML,
    model_defaults_yaml: str = _MODEL_DEFAULTS_YAML,
) -> Path:
    config_dir = parent / "configs"
    config_dir.mkdir()
    (config_dir / "instruments.yaml").write_text(instruments_yaml, encoding="utf-8")
    (config_dir / "training.yaml").write_text(training_yaml, encoding="utf-8")
    (config_dir / "model_defaults.yaml").write_text(
        model_defaults_yaml, encoding="utf-8"
    )
    return config_dir
