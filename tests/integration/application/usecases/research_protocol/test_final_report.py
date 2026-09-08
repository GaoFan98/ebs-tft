"""Integration coverage for the immutable final evidence report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import attrs
import polars as pl
import pytest

from ebs_tft.application.usecases import research_protocol
from ebs_tft.application.usecases.research_protocol import (
    _cross_instrument,
    _locked,
    _neural,
)

_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "log_loss",
    "multiclass_brier",
)


def test_builds_verified_final_report(tmp_path: Path) -> None:
    protocol_path = Path("notebooks/research_protocol.yaml").resolve()
    protocol = attrs.evolve(
        research_protocol.load_protocol(path=protocol_path),
        output_dir=tmp_path / "evidence",
    )
    _write_evidence(
        root=protocol.output_dir,
        protocol_path=protocol_path,
    )
    output = tmp_path / "final_analysis_outputs"

    result = research_protocol.run_final_report(
        protocol=protocol,
        protocol_path=protocol_path,
        output_dir=output,
        replace_output=False,
    )

    assert result.locked_confirmed_candidates == 2
    assert result.confirmed_cross_instrument_transfers == 1
    assert (output / "report.md").is_file()
    assert (output / "primary_evidence.csv").is_file()
    assert (output / "confirmatory_primary_effects.svg").is_file()
    summary = json.loads((output / "study_summary.json").read_text())
    assert summary["study_year"] == 2024
    assert summary["retuning_permitted"] is False
    assert summary["external_year_evaluation_status"] == (
        "not_run_no_external_year_data"
    )


def test_refuses_decision_that_does_not_match_evidence(tmp_path: Path) -> None:
    protocol_path = Path("notebooks/research_protocol.yaml").resolve()
    protocol = attrs.evolve(
        research_protocol.load_protocol(path=protocol_path),
        output_dir=tmp_path / "evidence",
    )
    _write_evidence(
        root=protocol.output_dir,
        protocol_path=protocol_path,
    )
    decision_path = protocol.output_dir / "locked_evaluation" / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["confirmed_candidates"] = []
    _write_json(decision_path, decision)

    with pytest.raises(ValueError, match="locked decision does not match"):
        research_protocol.run_final_report(
            protocol=protocol,
            protocol_path=protocol_path,
            output_dir=tmp_path / "final_analysis_outputs",
            replace_output=False,
        )


def _write_evidence(*, root: Path, protocol_path: Path) -> None:
    loaded = research_protocol.load_protocol(path=protocol_path)
    neural = root / "neural_benchmark"
    locked = root / "locked_evaluation"
    cross = root / "cross_instrument_evaluation"
    for path in (neural, locked, cross):
        path.mkdir(parents=True)

    neural_comparisons = _neural_comparisons()
    locked_comparisons = _locked_comparisons()
    cross_comparisons = _cross_comparisons()
    neural_comparisons.write_csv(neural / "paired_baseline_comparisons.csv")
    locked_comparisons.write_csv(locked / "paired_baseline_comparisons.csv")
    cross_comparisons.write_csv(cross / "paired_baseline_comparisons.csv")
    _session_metrics(instruments=("EUR_USD",)).write_csv(locked / "session_metrics.csv")
    _session_metrics(instruments=("EUR_JPY", "USD_JPY")).write_csv(
        cross / "session_metrics.csv"
    )
    _write_json(
        neural / "gate_decision.json",
        _neural._gate_decision(comparisons=neural_comparisons, protocol=loaded),
    )
    _write_json(
        locked / "decision.json",
        _locked._locked_decision(comparisons=locked_comparisons, protocol=loaded),
    )
    _write_json(
        cross / "decision.json",
        _cross_instrument._decision(comparisons=cross_comparisons, protocol=loaded),
    )
    _write_json(
        neural / "run_summary.json",
        {
            "protocol_sha256": _sha256(protocol_path),
            "cells": 64,
        },
    )
    dates = ("2024-03-06", "2024-03-13", "2024-03-20", "2024-03-27")
    _write_json(
        locked / "plan.json",
        {
            "stage": "locked",
            "development_sessions": [
                {"trading_date": "2024-01-02"},
                {"trading_date": "2024-01-03"},
            ],
            "final_test_sessions": [{"trading_date": date} for date in dates],
        },
    )
    _write_json(cross / "plan.json", {"stage": "cross"})
    _write_json(
        locked / "run_summary.json",
        {
            "plan_sha256": _sha256(locked / "plan.json"),
            "locked_evaluation_used": True,
        },
    )
    _write_json(
        cross / "run_summary.json",
        {
            "plan_sha256": _sha256(cross / "plan.json"),
            "cross_instrument_outcomes_used": True,
            "neural_retraining_used": False,
        },
    )


def _neural_comparisons() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ("deeplob_direction", "tft_direction"):
        for horizon in (5000, 10000, 30000, 60000):
            for metric in _METRICS:
                primary_pass = horizon == 30000 and metric in {"macro_f1", "mcc"}
                rows.append(
                    _comparison_row(
                        model=model,
                        metric=metric,
                        horizon=horizon,
                        sessions=20,
                        lower=0.01 if primary_pass else -0.01,
                    )
                )
    return pl.DataFrame(rows)


def _locked_comparisons() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _comparison_row(
                model=model,
                metric=metric,
                horizon=30000,
                sessions=4,
                lower=0.01 if metric in {"macro_f1", "mcc"} else -0.01,
            )
            for model in ("deeplob_direction", "tft_direction")
            for metric in _METRICS
        ]
    )


def _cross_comparisons() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for instrument in ("EUR_JPY", "USD_JPY"):
        for model in ("deeplob_direction", "tft_direction"):
            for metric in _METRICS:
                passed = (
                    instrument == "USD_JPY"
                    and model == "tft_direction"
                    and metric in {"macro_f1", "mcc"}
                )
                rows.append(
                    {
                        "instrument": instrument,
                        **_comparison_row(
                            model=model,
                            metric=metric,
                            horizon=30000,
                            sessions=4,
                            lower=0.01 if passed else -0.01,
                        ),
                    }
                )
    return pl.DataFrame(rows)


def _comparison_row(
    *, model: str, metric: str, horizon: int, sessions: int, lower: float
) -> dict[str, object]:
    return {
        "comparison": "seed_mean_neural_minus_logistic",
        "model": model,
        "depth": 1,
        "horizon_steps": horizon // 100,
        "horizon_milliseconds": horizon,
        "metric": metric,
        "favorable_direction": (
            "negative" if metric in {"log_loss", "multiclass_brier"} else "positive"
        ),
        "sessions": sessions,
        "mean_delta": lower + 0.01,
        "median_delta": lower + 0.01,
        "confidence_lower": lower,
        "confidence_upper": lower + 0.02,
    }


def _session_metrics(*, instruments: tuple[str, ...]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for instrument in instruments:
        for date_index, date in enumerate(
            ("2024-03-06", "2024-03-13", "2024-03-20", "2024-03-27")
        ):
            for model, seeds, advantage in (
                ("logistic", (-1,), 0.0),
                ("deeplob_direction", (7, 19), 0.01),
                (
                    "tft_direction",
                    (7, 19),
                    0.02 if instrument != "EUR_JPY" else -0.01,
                ),
            ):
                for seed in seeds:
                    base = 0.35 + date_index * 0.001 + advantage
                    rows.append(
                        {
                            "instrument": instrument,
                            "model": model,
                            "seed": seed,
                            "validation_date": date,
                            "balanced_accuracy": base,
                            "macro_f1": base,
                            "mcc": base - 0.2,
                            "log_loss": 1.0 - base,
                            "multiclass_brier": 0.7 - base,
                        }
                    )
    return pl.DataFrame(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
