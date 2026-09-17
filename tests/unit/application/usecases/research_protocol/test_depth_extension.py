"""Test the predeclared EUR/USD Level-1 versus Level-10 comparison."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from ebs_tft.application.usecases import research_protocol
from ebs_tft.application.usecases.research_protocol import _depth_extension
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models


def _protocol() -> research_models.ResearchProtocol:
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
    confidence_upper = cast(
        float,
        actual.filter(pl.col("metric") == "log_loss")["confidence_upper"].max(),
    )
    assert confidence_upper < 0


def test_depth_decision_requires_both_primary_metrics() -> None:
    protocol = _protocol()
    comparisons = _depth_extension._paired_depth_comparisons(
        metrics=_metrics(improvement=0.02), protocol=protocol
    ).with_columns(
        pl.when(
            (pl.col("model") == "deeplob_direction") & (pl.col("metric") == "macro_f1")
        )
        .then(False)
        .otherwise(pl.col("metric_passed"))
        .alias("metric_passed")
    )

    actual = _depth_extension._decision(comparisons=comparisons, protocol=protocol)

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
    macro_f1_delta = cast(float, actual["macro_f1_delta"].min())
    log_loss_delta = cast(float, actual["log_loss_delta"].max())
    assert abs(macro_f1_delta - 0.02) < 1e-12
    assert abs(log_loss_delta + 0.02) < 1e-12


def test_fixed_period_design_reuses_declared_level_1_evidence() -> None:
    protocol = research_protocol.load_protocol(
        path=Path("notebooks/research_2023_2024_protocol.yaml")
    )

    _depth_extension._validate_design(protocol=protocol)


def test_memory_bounded_target_selection_matches_canonical_corpus() -> None:
    rows = 20
    session = pilot_training.RawSessionData(
        trading_date=date(2024, 1, 2),
        lob_features=np.zeros((rows, 1, 6), dtype=np.float32),
        auxiliary_features=np.zeros((rows, 10), dtype=np.float32),
        labels=np.asarray([0, 1, 2, -1, 0] * 4, dtype=np.int64),
        timestamps=np.arange(rows).astype("datetime64[us]"),
        mid_prices=np.ones(rows, dtype=np.float64),
        observed=np.asarray([True] * 8 + [False] + [True] * 11),
    )

    expected = pilot_training.combine_sessions(
        sessions=(session,),
        context_steps=3,
        horizon_steps=2,
        maximum_windows=None,
        stride_steps=2,
    ).target_indices

    actual = _depth_extension._selected_targets(
        session=session,
        context_steps=3,
        horizon_steps=2,
        stride_steps=2,
    )

    np.testing.assert_array_equal(actual, expected)


def test_identity_upgrade_preserves_completed_cache_before_first_cell(
    tmp_path: Path,
) -> None:
    legacy = {"schema_version": 1, "input": "unchanged"}
    identity = {
        **legacy,
        "corpus_preparation": "disk_backed_window_preserving_v1",
    }
    path = tmp_path / "run_identity.json"
    path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    _depth_extension._verify_or_write_identity(
        output_dir=tmp_path,
        identity=identity,
    )

    assert json.loads(path.read_text(encoding="utf-8")) == identity
