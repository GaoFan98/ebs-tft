"""Test the predeclared EUR/USD Level-1 versus Level-10 comparison."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ebs_tft.application.usecases import research_protocol
from ebs_tft.application.usecases.research_protocol import _depth_extension


def _protocol():  # type: ignore[no-untyped-def]
    return research_protocol.load_protocol(
        path=Path("notebooks/research_protocol.yaml")
    )


def _metrics(*, improvement: float) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2024, 1, 1)
    for model in ("deeplob_direction", "tft_direction"):
        for depth in (1, 10):
            for fold_index in range(4):
                for session_index in range(5):
                    for seed in (7, 19):
                        gain = improvement if depth == 10 else 0.0
                        rows.append(
                            {
                                "instrument": "EUR_USD",
                                "fold": f"fold_{fold_index + 1:02d}",
                                "validation_date": start
                                + timedelta(days=fold_index * 5 + session_index),
                                "model": model,
                                "depth": depth,
                                "horizon_steps": 300,
                                "horizon_milliseconds": 30_000,
                                "seed": seed,
                                "macro_f1": 0.4 + gain,
                                "mcc": 0.2 + gain,
                                "balanced_accuracy": 0.42 + gain,
                                "log_loss": 0.9 - gain,
                                "multiclass_brier": 0.5 - gain,
                            }
                        )
    return pl.DataFrame(rows)


def test_paired_depth_comparisons_respect_metric_direction() -> None:
    protocol = _protocol()

    actual = _depth_extension._paired_depth_comparisons(
        metrics=_metrics(improvement=0.02), protocol=protocol
    )

    assert actual.height == 10
    assert actual["sessions"].unique().to_list() == [20]
    assert actual["metric_passed"].all()
    assert actual.filter(pl.col("metric") == "log_loss")[
        "confidence_upper"
    ].max() < 0


def test_depth_decision_requires_both_primary_metrics() -> None:
    protocol = _protocol()
    comparisons = _depth_extension._paired_depth_comparisons(
        metrics=_metrics(improvement=0.02), protocol=protocol
    ).with_columns(
        pl.when(
            (pl.col("model") == "deeplob_direction")
            & (pl.col("metric") == "macro_f1")
        )
        .then(False)
        .otherwise(pl.col("metric_passed"))
        .alias("metric_passed")
    )

    actual = _depth_extension._decision(
        comparisons=comparisons, protocol=protocol
    )

    assert actual["deeper_depth_supported_by_model"] == {
        "deeplob_direction": False,
        "tft_direction": True,
    }
    assert actual["supported_models"] == ["tft_direction"]
    assert actual["minute_aggregation_used"] is False


def test_session_deltas_average_seeds_before_pairing() -> None:
    protocol = _protocol()

    actual = _depth_extension._session_depth_deltas(
        metrics=_metrics(improvement=0.02), protocol=protocol
    )

    assert actual.height == 40
    assert abs(actual["macro_f1_delta"].min() - 0.02) < 1e-12
    assert abs(actual["log_loss_delta"].max() + 0.02) < 1e-12
