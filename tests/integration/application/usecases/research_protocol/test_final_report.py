"""Integration coverage for the immutable final evidence report."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import attrs
import polars as pl
import pytest

from ebs_tft.application.usecases import research_protocol
from ebs_tft.application.usecases.research_protocol import (
    _cross_instrument,
    _depth_extension,
    _depth_replication,
    _final_report,
    _locked,
    _neural,
    _paper_lstm,
)
from ebs_tft.domain.research import models as research_models

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
    workbook = output / "ebs_tft_2024_analysis.xlsx"
    assert workbook.is_file()
    with zipfile.ZipFile(workbook) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode()
    assert "Executive Summary" in workbook_xml
    assert "Model Summary" in workbook_xml
    assert "Data Usage" in workbook_xml
    assert "Class Balance" in workbook_xml
    assert "Confusion Matrices" in workbook_xml
    assert "Training Details" in workbook_xml
    assert "EURUSD Raw Sessions" in workbook_xml
    assert "Transfer Raw Sessions" in workbook_xml
    assert (output / "primary_evidence.csv").is_file()
    assert (output / "model_performance_summary.csv").is_file()
    assert (output / "data_usage.csv").is_file()
    assert (output / "class_balance.csv").is_file()
    assert (output / "confusion_matrices.csv").is_file()
    assert (output / "training_details.csv").is_file()
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


def test_builds_verified_longitudinal_report(tmp_path: Path) -> None:
    protocol_path = Path("notebooks/research_2023_2024_protocol.yaml").resolve()
    protocol = attrs.evolve(
        research_protocol.load_protocol(path=protocol_path),
        output_dir=tmp_path / "longitudinal_evidence",
    )
    _write_longitudinal_evidence(
        root=protocol.output_dir,
        protocol_path=protocol_path,
    )
    _write_longitudinal_depth_evidence(root=protocol.output_dir, protocol=protocol)
    output = tmp_path / "longitudinal_report_outputs"

    result = research_protocol.run_final_report(
        protocol=protocol,
        protocol_path=protocol_path,
        output_dir=output,
        replace_output=False,
    )

    assert result.locked_confirmed_candidates == 2
    assert result.confirmed_cross_instrument_transfers == 0
    workbook = output / "ebs_tft_2023_2024_longitudinal_analysis.xlsx"
    assert workbook.is_file()
    with zipfile.ZipFile(workbook) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode()
    assert "Executive Summary" in workbook_xml
    assert "Final Model Tests" in workbook_xml
    assert "Development Validation" in workbook_xml
    assert "March Replication" in workbook_xml
    assert "Depth Development" in workbook_xml
    assert "Depth March Replication" in workbook_xml
    assert "Depth Absolute Metrics" in workbook_xml
    assert "Depth Session Deltas" in workbook_xml
    assert "Methodology Notes" in workbook_xml
    summary = json.loads((output / "study_summary.json").read_text())
    assert summary["training_sessions"] == 2
    assert summary["validation_sessions"] == 2
    assert summary["replication_sessions"] == 4
    assert summary["minute_aggregation_used"] is False
    assert summary["evidence_status"] == "retrospective_replication_not_pristine"
    assert summary["depth_comparison_complete"] is True
    assert summary["depth_development_supported_models"] == ["deeplob_direction"]
    assert summary["depth_replication_supported_models"] == ["deeplob_direction"]
    assert (output / "artifact_manifest.json").is_file()


def test_longitudinal_paths_require_completed_depth_replication(
    tmp_path: Path,
) -> None:
    paths = _final_report._longitudinal_required_paths(source_root=tmp_path)
    assert "depth_replication_summary" not in paths

    replication = tmp_path / "depth_replication"
    replication.mkdir()
    _write_json(replication / "run_summary.json", {"status": "complete"})

    paths = _final_report._longitudinal_required_paths(source_root=tmp_path)
    assert paths["depth_replication_summary"] == replication / "run_summary.json"
    assert "depth_development_comparisons" in paths


def test_longitudinal_report_includes_verified_paper_lstm(tmp_path: Path) -> None:
    protocol_path = Path("notebooks/research_2023_2024_protocol.yaml").resolve()
    protocol = attrs.evolve(
        research_protocol.load_protocol(path=protocol_path),
        output_dir=tmp_path / "longitudinal_evidence",
    )
    _write_longitudinal_evidence(root=protocol.output_dir, protocol_path=protocol_path)
    _write_longitudinal_depth_evidence(root=protocol.output_dir, protocol=protocol)
    _write_longitudinal_paper_lstm_evidence(
        root=protocol.output_dir,
        protocol=protocol,
        protocol_path=protocol_path,
    )
    output = tmp_path / "report"

    research_protocol.run_final_report(
        protocol=protocol,
        protocol_path=protocol_path,
        output_dir=output,
        replace_output=False,
    )

    with zipfile.ZipFile(
        output / "ebs_tft_2023_2024_longitudinal_analysis.xlsx"
    ) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode()
    assert "LSTM March Scores" in workbook_xml
    assert "LSTM March Effects" in workbook_xml
    assert "LSTM Primary Verdicts" in workbook_xml
    assert "LSTM Interpretation" in workbook_xml
    assert "LSTM Validation Effects" in workbook_xml
    assert "LSTM March Training" in workbook_xml
    assert (output / "lstm_replication_comparisons.csv").is_file()
    assert pl.read_csv(output / "lstm_primary_verdicts.csv").height == 6
    summary = json.loads((output / "study_summary.json").read_text())
    assert summary["paper_inspired_lstm_comparison_complete"] is True
    assert summary["paper_reproduction"] is False
    assert summary["paper_lstm_march_retrospective"] is True
    assert summary["paper_lstm_validation_sessions"] == 2
    assert summary["paper_lstm_march_sessions"] == 4


def test_longitudinal_report_refuses_modified_paper_lstm_results(
    tmp_path: Path,
) -> None:
    protocol_path = Path("notebooks/research_2023_2024_protocol.yaml").resolve()
    protocol = attrs.evolve(
        research_protocol.load_protocol(path=protocol_path),
        output_dir=tmp_path / "longitudinal_evidence",
    )
    _write_longitudinal_evidence(root=protocol.output_dir, protocol_path=protocol_path)
    _write_longitudinal_paper_lstm_evidence(
        root=protocol.output_dir,
        protocol=protocol,
        protocol_path=protocol_path,
    )
    comparison_path = (
        protocol.output_dir
        / "paper_lstm_comparison"
        / "replication"
        / "paired_comparisons.csv"
    )
    comparison_path.write_text(comparison_path.read_text() + "\n")

    with pytest.raises(ValueError, match="comparisons checksum mismatch"):
        research_protocol.run_final_report(
            protocol=protocol,
            protocol_path=protocol_path,
            output_dir=tmp_path / "report",
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
    _write_json(
        cross / "plan.json",
        {
            "stage": "cross",
            "target_sessions": {
                instrument: [{"trading_date": date} for date in dates]
                for instrument in ("EUR_JPY", "USD_JPY")
            },
        },
    )
    for model, parameter_count in (
        ("deeplob_direction", 15_860),
        ("tft_direction", 52_581),
    ):
        for seed, fixed_epochs in ((7, 3), (19, 4)):
            cell = locked / "cells" / "h30000" / "depth_1" / model / f"seed_{seed}"
            cell.mkdir(parents=True)
            _write_json(
                cell / "cell_summary.json",
                {
                    "fixed_epochs": fixed_epochs,
                    "parameter_count": parameter_count,
                    "fit_elapsed_seconds": 10.0,
                    "locked_evaluation_used": True,
                },
            )
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


def _write_longitudinal_evidence(*, root: Path, protocol_path: Path) -> None:
    loaded = research_protocol.load_protocol(path=protocol_path)
    baseline = root / "baseline_gate"
    neural = root / "neural_benchmark"
    locked = root / "locked_evaluation"
    for path in (root, baseline, neural, locked):
        path.mkdir(parents=True, exist_ok=True)

    neural_comparisons = pl.DataFrame(
        [
            _comparison_row(
                model=model,
                metric=metric,
                horizon=30000,
                sessions=2,
                lower=0.01 if metric in {"macro_f1", "mcc"} else -0.01,
            )
            for model in ("deeplob_direction", "tft_direction")
            for metric in _METRICS
        ]
    )
    locked_comparisons = _locked_comparisons()
    locked_metrics = _session_metrics(instruments=("EUR_USD",))
    neural_comparisons.write_csv(neural / "paired_baseline_comparisons.csv")
    locked_comparisons.write_csv(locked / "paired_baseline_comparisons.csv")
    locked_metrics.write_csv(locked / "session_metrics.csv")
    _write_json(
        neural / "gate_decision.json",
        _neural._gate_decision(comparisons=neural_comparisons, protocol=loaded),
    )
    _write_json(
        locked / "decision.json",
        _locked._locked_decision(comparisons=locked_comparisons, protocol=loaded),
    )
    _write_json(
        neural / "run_summary.json",
        {"protocol_sha256": _sha256(protocol_path), "cells": 4},
    )
    _write_json(
        baseline / "gate_decision.json",
        {"eligible_for_neural_benchmark": True},
    )
    _write_json(baseline / "run_summary.json", {"folds": 1})
    development_dates = (
        "2023-12-28",
        "2023-12-29",
        "2024-01-02",
        "2024-01-03",
    )
    final_dates = ("2024-03-06", "2024-03-13", "2024-03-20", "2024-03-27")
    _write_json(
        locked / "plan.json",
        {
            "development_sessions": [
                {"trading_date": date} for date in development_dates
            ],
            "final_test_sessions": [{"trading_date": date} for date in final_dates],
            "locked_outcomes_inspected": False,
        },
    )
    for model, parameter_count in (
        ("deeplob_direction", 15_860),
        ("tft_direction", 52_581),
    ):
        for seed, fixed_epochs in ((7, 2), (19, 4)):
            cell = locked / "cells" / "h30000" / "depth_1" / model / f"seed_{seed}"
            cell.mkdir(parents=True)
            _write_json(
                cell / "cell_summary.json",
                {
                    "fixed_epochs": fixed_epochs,
                    "parameter_count": parameter_count,
                    "fit_elapsed_seconds": 10.0,
                    "locked_evaluation_used": True,
                },
            )
    _write_json(
        locked / "run_summary.json",
        {
            "plan_sha256": _sha256(locked / "plan.json"),
            "locked_evaluation_used": True,
        },
    )
    _write_json(
        root / "audit_summary.json",
        {"discovered_sessions": 10, "eligible_sessions": 8},
    )
    (root / "session_audit.csv").write_text(
        "instrument,trading_date\n", encoding="utf-8"
    )
    (root / "split_manifest.yaml").write_text("schema_version: 1\n", encoding="utf-8")


def _write_longitudinal_depth_evidence(
    *, root: Path, protocol: research_models.ResearchProtocol
) -> None:
    development = root / "depth_extension"
    replication = root / "depth_replication"
    development.mkdir()
    replication.mkdir()
    development_comparisons = pl.DataFrame(
        [
            _depth_comparison_row(
                model=model,
                metric=metric,
                sessions=51,
                passed=model == "deeplob_direction" and metric in {"macro_f1", "mcc"},
            )
            for model in ("deeplob_direction", "tft_direction")
            for metric in _METRICS
        ]
    )
    replication_comparisons = pl.DataFrame(
        [
            _depth_comparison_row(
                model="deeplob_direction",
                metric=metric,
                sessions=4,
                passed=metric in {"macro_f1", "mcc"},
            )
            for metric in _METRICS
        ]
    )
    development_comparisons.write_csv(development / "paired_depth_comparisons.csv")
    replication_comparisons.write_csv(replication / "paired_depth_comparisons.csv")
    _write_json(
        development / "decision.json",
        _depth_extension._decision(
            comparisons=development_comparisons, protocol=protocol
        ),
    )
    _write_json(development / "run_summary.json", {"status": "complete"})
    _write_json(
        replication / "decision.json",
        _depth_replication._replication_decision(
            comparisons=replication_comparisons, protocol=protocol
        ),
    )
    plan_path = replication / "plan.json"
    _write_json(plan_path, {"stage": "retrospective_depth_replication"})
    _write_json(
        replication / "run_summary.json",
        {
            "status": "complete",
            "plan_sha256": _sha256(plan_path),
            "retrospective_replication": True,
            "retuning_permitted": False,
        },
    )
    replication_comparisons.write_csv(replication / "absolute_metric_summary.csv")
    replication_comparisons.write_csv(replication / "session_depth_deltas.csv")


def _write_longitudinal_paper_lstm_evidence(
    *,
    root: Path,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
) -> None:
    paper_root = root / "paper_lstm_comparison"
    development_dates = ("2024-01-02", "2024-01-03")
    replication_dates = ("2024-03-06", "2024-03-13", "2024-03-20", "2024-03-27")
    locked_metrics_path = root / "locked_evaluation" / "session_metrics.csv"
    locked_metrics = pl.read_csv(locked_metrics_path).with_columns(
        pl.col("balanced_accuracy").alias("accuracy")
    )
    locked_metrics.write_csv(locked_metrics_path)
    development_reference = pl.DataFrame(
        [
            {
                "model": model,
                "seed": seed,
                "validation_date": date,
                "observations": 1_000 + index,
                "accuracy": score,
                "macro_f1": score,
                "mcc": score - 0.2,
            }
            for index, date in enumerate(development_dates)
            for model, score in (
                ("deeplob_direction", 0.36),
                ("tft_direction", 0.37),
            )
            for seed in (7, 19)
        ]
    )
    development_reference_path = root / "neural_benchmark" / "session_metrics.csv"
    development_reference.write_csv(development_reference_path)
    references = {
        "development": development_reference,
        "replication": locked_metrics.filter(pl.col("model").is_in(protocol.models)),
    }
    for stage, reference in references.items():
        stage_root = paper_root / stage
        stage_root.mkdir(parents=True)
        dates = development_dates if stage == "development" else replication_dates
        metrics = pl.DataFrame(
            [
                {
                    "feature_set": feature_set,
                    "seed": seed,
                    "validation_date": date,
                    "observations": 1_000 + index,
                    "accuracy": score,
                    "macro_f1": score,
                    "mcc": score - 0.2,
                }
                for index, date in enumerate(dates)
                for feature_set, score in (
                    ("level_1_and_transactions", 0.365),
                    ("transactions_only", 0.30),
                )
                for seed in (7, 19)
            ]
        )
        comparisons = _paper_lstm._comparisons(
            metrics=metrics,
            reference=reference,
            protocol=protocol,
            stage=stage,
        )
        absolute = _paper_lstm._absolute_summary(metrics=metrics, reference=reference)
        training = pl.DataFrame(
            [
                {
                    "feature_set": feature_set,
                    "seed": seed,
                    "epochs": 2,
                    "training_windows": 100,
                    "evaluation_windows": 1_000 * len(dates),
                    "parameter_count": 11_000,
                    "fit_elapsed_seconds": 1.0,
                }
                for feature_set in (
                    "level_1_and_transactions",
                    "transactions_only",
                )
                for seed in (7, 19)
            ]
        )
        metrics.write_csv(stage_root / "session_metrics.csv")
        comparisons.write_csv(stage_root / "paired_comparisons.csv")
        absolute.write_csv(stage_root / "absolute_summary.csv")
        training.write_csv(stage_root / "training_details.csv")
        (stage_root / "paper_lstm_comparison.xlsx").write_bytes(b"fixture")
        identity = {
            "stage": stage,
            "paper_reproduction": False,
            "training_data_semantics": "native_100ms_snapshots_not_order_events",
            "protocol_sha256": _sha256(protocol_path),
            "policy_sha256": _sha256(
                protocol_path.with_name("research_2023_2024_neural.yaml")
            ),
            "manifest_sha256": _sha256(root / "split_manifest.yaml"),
            "audit_sha256": _sha256(root / "session_audit.csv"),
            "reference_metrics_sha256": _sha256(
                development_reference_path
                if stage == "development"
                else locked_metrics_path
            ),
        }
        _write_json(stage_root / "run_identity.json", identity)
        _write_json(
            stage_root / "run_summary.json",
            {
                "status": "complete",
                "stage": stage,
                "paper_reproduction": False,
                "minute_aggregation_used": False,
                "retrospective_march": stage == "replication",
                "retuning_permitted": False,
                "cells": 4,
                "sessions": len(dates),
                "identity_sha256": _sha256(stage_root / "run_identity.json"),
                "metrics_sha256": _sha256(stage_root / "session_metrics.csv"),
                "comparisons_sha256": _sha256(stage_root / "paired_comparisons.csv"),
                "absolute_summary_sha256": _sha256(stage_root / "absolute_summary.csv"),
                "training_details_sha256": _sha256(stage_root / "training_details.csv"),
                "workbook_sha256": _sha256(stage_root / "paper_lstm_comparison.xlsx"),
            },
        )
    _write_json(
        paper_root / "replication" / "plan.json",
        {
            "retrospective_not_pristine": True,
            "development_summary_sha256": _sha256(
                paper_root / "development" / "run_summary.json"
            ),
            "replication_dates": list(replication_dates),
        },
    )


def _depth_comparison_row(
    *, model: str, metric: str, sessions: int, passed: bool
) -> dict[str, object]:
    lower = 0.001 if passed else -0.001
    return {
        "comparison": "seed_mean_level_10_minus_level_1",
        "instrument": "EUR_USD",
        "model": model,
        "shallow_depth": 1,
        "deep_depth": 10,
        "horizon_milliseconds": 30_000,
        "metric": metric,
        "favorable_direction": (
            "negative" if metric in {"log_loss", "multiclass_brier"} else "positive"
        ),
        "sessions": sessions,
        "mean_delta": 0.005,
        "median_delta": 0.005,
        "confidence_lower": lower,
        "confidence_upper": 0.01,
        "metric_passed": passed,
    }


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
                            "observations": 1_000 + date_index,
                            "parameter_count": {
                                "logistic": 0,
                                "deeplob_direction": 15_860,
                                "tft_direction": 52_581,
                            }[model],
                            "balanced_accuracy": base,
                            "macro_f1": base,
                            "mcc": base - 0.2,
                            "log_loss": 1.0 - base,
                            "multiclass_brier": 0.7 - base,
                            "confusion_down_down": 150,
                            "confusion_down_flat": 100,
                            "confusion_down_up": 50,
                            "confusion_flat_down": 90,
                            "confusion_flat_flat": 250,
                            "confusion_flat_up": 60,
                            "confusion_up_down": 40,
                            "confusion_up_flat": 110,
                            "confusion_up_up": 150 + date_index,
                        }
                    )
    return pl.DataFrame(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
