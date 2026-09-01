"""Test bounded pilot-matrix comparisons."""

import datetime

import polars as pl
import pytest

from ebs_tft.application.usecases.pilot import _comparison


class TestCumulativeDepthComparison:
    def test_pairs_each_depth_with_the_preceding_configured_depth(self) -> None:
        common = {
            "instrument": ["EUR_USD", "EUR_USD", "EUR_USD"],
            "trading_date": [
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 3),
            ],
            "horizon_steps": [50, 50, 50],
            "training_mode": ["unweighted", "unweighted", "unweighted"],
            "model": ["deeplob_direction"] * 3,
            "seed": [7, 7, 7],
            "depth": [1, 2, 3],
            "balanced_accuracy": [0.30, 0.35, 0.36],
            "macro_f1": [0.20, 0.24, 0.25],
            "mcc": [0.01, 0.03, 0.04],
            "log_loss": [1.10, 1.00, 0.99],
        }

        actual = _comparison.cumulative_depth_comparison(metrics=pl.DataFrame(common))

        assert actual.height == 2
        first = actual.row(0, named=True)
        assert first["shallower_depth"] == 1
        assert first["deeper_depth"] == 2
        assert first["balanced_accuracy_delta"] == pytest.approx(0.05)
        assert first["macro_f1_delta"] == pytest.approx(0.04)
        assert first["mcc_delta"] == pytest.approx(0.02)
        assert first["log_loss_delta"] == pytest.approx(-0.1)
