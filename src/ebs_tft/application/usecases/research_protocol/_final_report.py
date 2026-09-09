"""Build the immutable 2024 study report from completed evaluation artifacts."""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import cast

import attrs
import polars as pl
import xlsxwriter

from ebs_tft.application.usecases.research_protocol import (
    _cross_instrument,
    _locked,
    _neural,
)
from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain.research import models as research_models

_PRIMARY_METRICS = ("macro_f1", "mcc")
_REPORTED_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "log_loss",
    "multiclass_brier",
)


@attrs.frozen
class FinalReportResult:
    """Paths and headline counts from one completed report build."""

    output_dir: Path
    report_path: Path
    locked_confirmed_candidates: int
    confirmed_cross_instrument_transfers: int


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    output_dir: Path,
    replace_output: bool,
) -> FinalReportResult:
    """Verify completed evidence and render deterministic tables and figures."""
    source_root = protocol.output_dir
    paths = _required_paths(source_root=source_root)
    _require_files(paths=paths)
    artifact_repository.prepare_run_directory(
        path=output_dir,
        replace=replace_output,
        replacement_parent=output_dir.parent,
    )

    neural_comparisons = _read_csv(paths["neural_comparisons"])
    locked_comparisons = _read_csv(paths["locked_comparisons"])
    cross_comparisons = _read_csv(paths["cross_comparisons"])
    locked_metrics = _read_csv(paths["locked_metrics"])
    cross_metrics = _read_csv(paths["cross_metrics"])
    neural_decision = _json_mapping(paths["neural_decision"])
    locked_decision = _json_mapping(paths["locked_decision"])
    cross_decision = _json_mapping(paths["cross_decision"])
    locked_plan = _json_mapping(paths["locked_plan"])
    cross_plan = _json_mapping(paths["cross_plan"])
    neural_summary = _json_mapping(paths["neural_summary"])
    locked_summary = _json_mapping(paths["locked_summary"])
    cross_summary = _json_mapping(paths["cross_summary"])

    _verify_evidence(
        protocol=protocol,
        protocol_path=protocol_path,
        paths=paths,
        neural_comparisons=neural_comparisons,
        locked_comparisons=locked_comparisons,
        cross_comparisons=cross_comparisons,
        locked_metrics=locked_metrics,
        cross_metrics=cross_metrics,
        neural_decision=neural_decision,
        locked_decision=locked_decision,
        cross_decision=cross_decision,
        neural_summary=neural_summary,
        locked_summary=locked_summary,
        cross_summary=cross_summary,
    )

    development = _stage_comparisons(data=neural_comparisons, stage="development")
    locked = _stage_comparisons(data=locked_comparisons, stage="locked_evaluation")
    cross = _stage_comparisons(data=cross_comparisons, stage="cross_instrument")
    primary_evidence = pl.concat(
        [
            development.filter(pl.col("metric").is_in(_PRIMARY_METRICS)),
            locked.filter(pl.col("metric").is_in(_PRIMARY_METRICS)),
            cross.filter(pl.col("metric").is_in(_PRIMARY_METRICS)),
        ],
        how="diagonal_relaxed",
    ).sort(["stage", "instrument", "model", "horizon_milliseconds", "metric"])
    absolute_metrics = pl.concat(
        [
            _absolute_summary(data=locked_metrics, stage="locked_evaluation"),
            _absolute_summary(data=cross_metrics, stage="cross_instrument"),
        ]
    ).sort(["stage", "instrument", "model", "metric"])
    session_deltas = pl.concat(
        [
            _session_deltas(data=locked_metrics, stage="locked_evaluation"),
            _session_deltas(data=cross_metrics, stage="cross_instrument"),
        ]
    ).sort(["stage", "instrument", "model", "validation_date"])
    model_summary = _model_summary(
        absolute_metrics=absolute_metrics,
        comparisons=pl.concat([locked, cross]),
        locked_metrics=locked_metrics,
        cross_metrics=cross_metrics,
    )
    data_usage = _data_usage(
        locked_plan=locked_plan,
        cross_plan=cross_plan,
        locked_metrics=locked_metrics,
        cross_metrics=cross_metrics,
        development_validation_sessions=_single_integer(
            neural_comparisons, column="sessions"
        ),
    )
    class_balance = _class_balance(
        metrics=pl.concat([locked_metrics, cross_metrics], how="diagonal_relaxed")
    )
    confusion_matrices = _confusion_matrices(
        metrics=pl.concat([locked_metrics, cross_metrics], how="diagonal_relaxed")
    )
    training_details = _training_details(paths=paths, metrics=locked_metrics)

    development.write_csv(output_dir / "development_comparisons.csv")
    locked.write_csv(output_dir / "locked_comparisons.csv")
    cross.write_csv(output_dir / "cross_instrument_comparisons.csv")
    primary_evidence.write_csv(output_dir / "primary_evidence.csv")
    absolute_metrics.write_csv(output_dir / "absolute_metric_summary.csv")
    session_deltas.write_csv(output_dir / "session_primary_deltas.csv")
    model_summary.write_csv(output_dir / "model_performance_summary.csv")
    data_usage.write_csv(output_dir / "data_usage.csv")
    class_balance.write_csv(output_dir / "class_balance.csv")
    confusion_matrices.write_csv(output_dir / "confusion_matrices.csv")
    training_details.write_csv(output_dir / "training_details.csv")
    (output_dir / "development_primary_effects.svg").write_text(
        _forest_svg(
            evidence=primary_evidence.filter(pl.col("stage") == "development"),
            title="Development benchmark: primary effects vs logistic",
        ),
        encoding="utf-8",
    )
    (output_dir / "confirmatory_primary_effects.svg").write_text(
        _forest_svg(
            evidence=primary_evidence.filter(pl.col("stage") != "development"),
            title="Locked and cross-instrument primary effects vs logistic",
        ),
        encoding="utf-8",
    )

    locked_confirmed = _list_length(locked_decision, "confirmed_candidates")
    cross_confirmed = _list_length(cross_decision, "confirmed_transfers")
    dates = sorted(set(locked_metrics["validation_date"].cast(pl.String).to_list()))
    development_dates = _plan_session_dates(
        plan=locked_plan, key="development_sessions"
    )
    frozen_final_dates = _plan_session_dates(
        plan=locked_plan, key="final_test_sessions"
    )
    if frozen_final_dates != dates:
        raise ValueError("locked metric dates do not match the frozen plan")
    study_summary: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "study_year": _single_year(dates=dates),
        "development_sessions": _single_integer(neural_comparisons, column="sessions"),
        "development_training_sessions": len(development_dates),
        "locked_evaluation_sessions": len(dates),
        "evidence_period_start": min(development_dates),
        "evidence_period_end": max(dates),
        "locked_confirmed_candidates": locked_confirmed,
        "cross_instrument_sessions_per_instrument": _single_integer(
            cross_comparisons, column="sessions"
        ),
        "confirmed_cross_instrument_transfers": cross_confirmed,
        "neural_benchmark_cells": _integer(neural_summary, "cells"),
        "locked_outcomes_used": True,
        "cross_instrument_outcomes_used": True,
        "retuning_permitted": False,
        "external_year_evaluation_status": "not_run_no_external_year_data",
        "headline_conclusion": (
            "Both frozen 30-second EUR/USD neural candidates passed the locked "
            "gate; only TFT passed the strict USD/JPY transfer gate, and neither "
            "candidate passed the EUR/JPY transfer gate."
        ),
    }
    _write_json(path=output_dir / "study_summary.json", value=study_summary)
    workbook_path = output_dir / "ebs_tft_2024_analysis.xlsx"
    _write_workbook(
        path=workbook_path,
        summary=study_summary,
        development=development,
        locked=locked,
        cross=cross,
        absolute_metrics=absolute_metrics,
        model_summary=model_summary,
        data_usage=data_usage,
        class_balance=class_balance,
        confusion_matrices=confusion_matrices,
        training_details=training_details,
        session_deltas=session_deltas,
        locked_metrics=locked_metrics,
        cross_metrics=cross_metrics,
        dates=dates,
    )

    _write_json(
        path=output_dir / "run_summary.json",
        value={
            **study_summary,
            "report": str(workbook_path.resolve()),
            "artifact_manifest": str((output_dir / "artifact_manifest.json").resolve()),
        },
    )
    manifest = {
        "schema_version": 1,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(paths.items())
        },
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.name != "artifact_manifest.json"
        },
    }
    _write_json(path=output_dir / "artifact_manifest.json", value=manifest)
    print("EBS 2024 final evidence report completed")
    print("WARNING: reporting only; locked outcomes were not used for retuning.")
    print(f"locked_confirmed_candidates={locked_confirmed}")
    print(f"confirmed_cross_instrument_transfers={cross_confirmed}")
    print(f"outputs={output_dir.resolve()}")
    return FinalReportResult(
        output_dir=output_dir,
        report_path=workbook_path,
        locked_confirmed_candidates=locked_confirmed,
        confirmed_cross_instrument_transfers=cross_confirmed,
    )


def _required_paths(*, source_root: Path) -> dict[str, Path]:
    neural = source_root / "neural_benchmark"
    locked = source_root / "locked_evaluation"
    cross = source_root / "cross_instrument_evaluation"
    paths = {
        "protocol": Path(),  # Replaced with the explicit protocol path below.
        "neural_summary": neural / "run_summary.json",
        "neural_decision": neural / "gate_decision.json",
        "neural_comparisons": neural / "paired_baseline_comparisons.csv",
        "locked_plan": locked / "plan.json",
        "locked_summary": locked / "run_summary.json",
        "locked_decision": locked / "decision.json",
        "locked_comparisons": locked / "paired_baseline_comparisons.csv",
        "locked_metrics": locked / "session_metrics.csv",
        "cross_plan": cross / "plan.json",
        "cross_summary": cross / "run_summary.json",
        "cross_decision": cross / "decision.json",
        "cross_comparisons": cross / "paired_baseline_comparisons.csv",
        "cross_metrics": cross / "session_metrics.csv",
    }
    for model in ("deeplob_direction", "tft_direction"):
        for seed in (7, 19):
            paths[f"locked_cell_{model}_seed_{seed}"] = (
                locked
                / "cells"
                / "h30000"
                / "depth_1"
                / model
                / f"seed_{seed}"
                / "cell_summary.json"
            )
    return paths


def _require_files(*, paths: dict[str, Path]) -> None:
    # The protocol placeholder is populated only after existence checks.
    for name, path in paths.items():
        if name != "protocol" and not path.is_file():
            raise FileNotFoundError(f"missing completed 2024 evidence: {path}")


def _verify_evidence(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    paths: dict[str, Path],
    neural_comparisons: pl.DataFrame,
    locked_comparisons: pl.DataFrame,
    cross_comparisons: pl.DataFrame,
    locked_metrics: pl.DataFrame,
    cross_metrics: pl.DataFrame,
    neural_decision: dict[str, object],
    locked_decision: dict[str, object],
    cross_decision: dict[str, object],
    neural_summary: dict[str, object],
    locked_summary: dict[str, object],
    cross_summary: dict[str, object],
) -> None:
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing protocol: {protocol_path}")
    paths["protocol"] = protocol_path
    if _string(neural_summary, "protocol_sha256") != _sha256(protocol_path):
        raise ValueError("neural evidence does not match the current protocol")
    if _integer(neural_summary, "cells") != 64:
        raise ValueError("neural benchmark must contain exactly 64 completed cells")
    if locked_summary.get("locked_evaluation_used") is not True:
        raise ValueError("locked evaluation is incomplete")
    if cross_summary.get("cross_instrument_outcomes_used") is not True:
        raise ValueError("cross-instrument evaluation is incomplete")
    if cross_summary.get("neural_retraining_used") is not False:
        raise ValueError("cross-instrument evidence unexpectedly retrained a model")
    if _string(locked_summary, "plan_sha256") != _sha256(paths["locked_plan"]):
        raise ValueError("locked evaluation plan hash mismatch")
    if _string(cross_summary, "plan_sha256") != _sha256(paths["cross_plan"]):
        raise ValueError("cross-instrument plan hash mismatch")

    _validate_comparison_table(data=neural_comparisons, instrument_required=False)
    _validate_comparison_table(data=locked_comparisons, instrument_required=False)
    _validate_comparison_table(data=cross_comparisons, instrument_required=True)
    _validate_metric_table(data=locked_metrics, expected_instruments={"EUR_USD"})
    _validate_metric_table(
        data=cross_metrics, expected_instruments={"EUR_JPY", "USD_JPY"}
    )

    expected_neural = _neural._gate_decision(
        comparisons=neural_comparisons, protocol=protocol
    )
    expected_locked = _locked._locked_decision(
        comparisons=locked_comparisons, protocol=protocol
    )
    expected_cross = _cross_instrument._decision(
        comparisons=cross_comparisons, protocol=protocol
    )
    if _normalized_decision(neural_decision) != _normalized_decision(expected_neural):
        raise ValueError("neural gate decision does not match its comparison evidence")
    if _normalized_decision(locked_decision) != _normalized_decision(expected_locked):
        raise ValueError("locked decision does not match its comparison evidence")
    if _normalized_decision(cross_decision) != _normalized_decision(expected_cross):
        raise ValueError("cross-instrument decision does not match its evidence")


def _validate_comparison_table(
    *, data: pl.DataFrame, instrument_required: bool
) -> None:
    required = {
        "model",
        "depth",
        "horizon_milliseconds",
        "metric",
        "sessions",
        "mean_delta",
        "confidence_lower",
        "confidence_upper",
    }
    if instrument_required:
        required.add("instrument")
    if not required.issubset(data.columns) or data.height == 0:
        raise ValueError("comparison evidence has an invalid schema")
    if data.select(list(required)).null_count().sum_horizontal().item() != 0:
        raise ValueError("comparison evidence contains nulls")
    numeric = data.select(
        "mean_delta", "confidence_lower", "confidence_upper"
    ).to_numpy()
    if not all(math.isfinite(float(value)) for value in numeric.flat):
        raise ValueError("comparison evidence contains non-finite values")
    if bool((data["confidence_lower"] > data["confidence_upper"]).any()):
        raise ValueError("comparison evidence contains inverted intervals")


def _validate_metric_table(
    *, data: pl.DataFrame, expected_instruments: set[str]
) -> None:
    required = {
        "instrument",
        "model",
        "seed",
        "validation_date",
        *_REPORTED_METRICS,
    }
    if not required.issubset(data.columns) or data.height == 0:
        raise ValueError("session metrics have an invalid schema")
    if set(data["instrument"].unique().to_list()) != expected_instruments:
        raise ValueError("session metrics contain unexpected instruments")
    dates = data.group_by("instrument").agg(pl.col("validation_date").n_unique())
    if bool((dates["validation_date"] != 4).any()):
        raise ValueError(
            "confirmatory evidence must contain four sessions per instrument"
        )
    models = set(data["model"].unique().to_list())
    if models != {"logistic", "deeplob_direction", "tft_direction"}:
        raise ValueError("session metrics contain unexpected models")


def _stage_comparisons(*, data: pl.DataFrame, stage: str) -> pl.DataFrame:
    result = data.with_columns(
        pl.lit(stage).alias("stage"),
        pl.when(
            (
                (pl.col("favorable_direction") == "positive")
                & (pl.col("confidence_lower") > 0.0)
            )
            | (
                (pl.col("favorable_direction") == "negative")
                & (pl.col("confidence_upper") < 0.0)
            )
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("metric_passed"),
    )
    if "instrument" not in result.columns:
        result = result.with_columns(pl.lit("EUR_USD").alias("instrument"))
    return result.select(
        "stage",
        "instrument",
        "model",
        "depth",
        "horizon_steps",
        "horizon_milliseconds",
        "metric",
        "favorable_direction",
        "sessions",
        "mean_delta",
        "median_delta",
        "confidence_lower",
        "confidence_upper",
        "metric_passed",
    )


def _absolute_summary(*, data: pl.DataFrame, stage: str) -> pl.DataFrame:
    by_session = data.group_by("instrument", "validation_date", "model").agg(
        [pl.col(metric).mean().alias(metric) for metric in _REPORTED_METRICS]
    )
    rows: list[dict[str, object]] = []
    for dimension in (
        by_session.select("instrument", "model").unique().iter_rows(named=True)
    ):
        selected = by_session.filter(
            (pl.col("instrument") == dimension["instrument"])
            & (pl.col("model") == dimension["model"])
        )
        for metric in _REPORTED_METRICS:
            values = selected[metric]
            rows.append(
                {
                    "stage": stage,
                    **dimension,
                    "metric": metric,
                    "sessions": selected.height,
                    "mean": _finite_number(values.mean()),
                    "standard_deviation": _finite_number(values.std(ddof=1)),
                    "minimum": _finite_number(values.min()),
                    "maximum": _finite_number(values.max()),
                }
            )
    return pl.DataFrame(rows)


def _session_deltas(*, data: pl.DataFrame, stage: str) -> pl.DataFrame:
    by_session = data.group_by("instrument", "validation_date", "model").agg(
        [pl.col(metric).mean().alias(metric) for metric in _PRIMARY_METRICS]
    )
    neural = by_session.filter(pl.col("model") != "logistic")
    baseline = by_session.filter(pl.col("model") == "logistic").drop("model")
    joined = neural.join(
        baseline,
        on=["instrument", "validation_date"],
        suffix="_logistic",
        validate="m:1",
    )
    return joined.select(
        pl.lit(stage).alias("stage"),
        "instrument",
        "validation_date",
        "model",
        *[
            (pl.col(metric) - pl.col(f"{metric}_logistic")).alias(f"{metric}_delta")
            for metric in _PRIMARY_METRICS
        ],
    )


def _model_summary(
    *,
    absolute_metrics: pl.DataFrame,
    comparisons: pl.DataFrame,
    locked_metrics: pl.DataFrame,
    cross_metrics: pl.DataFrame,
) -> pl.DataFrame:
    absolute = {
        (
            str(row["stage"]),
            str(row["instrument"]),
            str(row["model"]),
            str(row["metric"]),
        ): float(row["mean"])
        for row in absolute_metrics.iter_rows(named=True)
    }
    comparison = {
        (
            str(row["stage"]),
            str(row["instrument"]),
            str(row["model"]),
            str(row["metric"]),
        ): row
        for row in comparisons.iter_rows(named=True)
    }
    all_metrics = pl.concat([locked_metrics, cross_metrics], how="diagonal_relaxed")
    observations = {
        str(row["instrument"]): int(row["observations"])
        for row in (
            all_metrics.filter(pl.col("model") == "logistic")
            .group_by("instrument")
            .agg(pl.col("observations").sum())
            .iter_rows(named=True)
        )
    }
    parameters = {
        str(row["model"]): int(row["parameter_count"])
        for row in all_metrics.select("model", "parameter_count")
        .unique()
        .iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    dimensions = (
        comparisons.select("stage", "instrument", "model")
        .unique()
        .sort("stage", "instrument", "model")
    )
    for dimension in dimensions.iter_rows(named=True):
        stage = str(dimension["stage"])
        instrument = str(dimension["instrument"])
        model = str(dimension["model"])
        macro = comparison[(stage, instrument, model, "macro_f1")]
        mcc = comparison[(stage, instrument, model, "mcc")]
        log_loss = comparison[(stage, instrument, model, "log_loss")]
        rows.append(
            {
                "stage": stage,
                "instrument": instrument,
                "model": model,
                "horizon_seconds": int(macro["horizon_milliseconds"]) // 1000,
                "sessions": int(macro["sessions"]),
                "evaluation_observations": observations[instrument],
                "parameter_count": parameters[model],
                "macro_f1": absolute[(stage, instrument, model, "macro_f1")],
                "logistic_macro_f1": absolute[
                    (stage, instrument, "logistic", "macro_f1")
                ],
                "macro_f1_delta": float(macro["mean_delta"]),
                "macro_f1_ci_lower": float(macro["confidence_lower"]),
                "macro_f1_ci_upper": float(macro["confidence_upper"]),
                "mcc": absolute[(stage, instrument, model, "mcc")],
                "logistic_mcc": absolute[(stage, instrument, "logistic", "mcc")],
                "mcc_delta": float(mcc["mean_delta"]),
                "mcc_ci_lower": float(mcc["confidence_lower"]),
                "mcc_ci_upper": float(mcc["confidence_upper"]),
                "log_loss": absolute[(stage, instrument, model, "log_loss")],
                "logistic_log_loss": absolute[
                    (stage, instrument, "logistic", "log_loss")
                ],
                "log_loss_delta": float(log_loss["mean_delta"]),
                "primary_gate_passed": bool(macro["metric_passed"])
                and bool(mcc["metric_passed"]),
            }
        )
    return pl.DataFrame(rows)


def _data_usage(
    *,
    locked_plan: dict[str, object],
    cross_plan: dict[str, object],
    locked_metrics: pl.DataFrame,
    cross_metrics: pl.DataFrame,
    development_validation_sessions: int,
) -> pl.DataFrame:
    development_dates = _plan_session_dates(
        plan=locked_plan, key="development_sessions"
    )
    final_dates = _plan_session_dates(plan=locked_plan, key="final_test_sessions")
    target_sessions = cross_plan.get("target_sessions")
    if not isinstance(target_sessions, dict):
        raise ValueError("cross-instrument plan target_sessions must be an object")
    all_metrics = pl.concat([locked_metrics, cross_metrics], how="diagonal_relaxed")
    baseline = all_metrics.filter(pl.col("model") == "logistic")
    rows: list[dict[str, object]] = []
    for instrument in ("EUR_USD", "USD_JPY", "EUR_JPY"):
        selected = baseline.filter(pl.col("instrument") == instrument)
        if selected.height != 4:
            raise ValueError(f"expected four final sessions for {instrument}")
        if instrument == "EUR_USD":
            training_sessions = len(development_dates)
            rolling_validation = development_validation_sessions
            evaluation_dates = final_dates
            training_note = "Neural models and logistic reference trained on EUR/USD"
        else:
            training_sessions = 0
            rolling_validation = 0
            raw_targets = target_sessions.get(instrument)
            if not isinstance(raw_targets, list):
                raise ValueError(f"missing frozen target sessions for {instrument}")
            evaluation_dates = sorted(
                str(item["trading_date"])
                for item in raw_targets
                if isinstance(item, dict) and isinstance(item.get("trading_date"), str)
            )
            if len(evaluation_dates) != 4:
                raise ValueError(f"expected four frozen target dates for {instrument}")
            training_note = "Frozen EUR/USD models applied without neural retraining"
        rows.append(
            {
                "instrument": instrument,
                "neural_training_sessions": training_sessions,
                "rolling_development_validation_sessions": rolling_validation,
                "final_evaluation_sessions": len(evaluation_dates),
                "final_evaluation_dates": ", ".join(evaluation_dates),
                "unique_final_observations": int(selected["observations"].sum()),
                "minimum_observations_per_session": int(
                    _finite_number(selected["observations"].min())
                ),
                "maximum_observations_per_session": int(
                    _finite_number(selected["observations"].max())
                ),
                "observation_interval_milliseconds": 100,
                "note": training_note,
            }
        )
    return pl.DataFrame(rows)


def _class_balance(*, metrics: pl.DataFrame) -> pl.DataFrame:
    baseline = metrics.filter(pl.col("model") == "logistic")
    rows: list[dict[str, object]] = []
    for instrument in sorted(baseline["instrument"].unique().to_list()):
        selected = baseline.filter(pl.col("instrument") == instrument)
        down = int(
            selected.select(
                pl.sum_horizontal(
                    "confusion_down_down",
                    "confusion_down_flat",
                    "confusion_down_up",
                ).sum()
            ).item()
        )
        flat = int(
            selected.select(
                pl.sum_horizontal(
                    "confusion_flat_down",
                    "confusion_flat_flat",
                    "confusion_flat_up",
                ).sum()
            ).item()
        )
        up = int(
            selected.select(
                pl.sum_horizontal(
                    "confusion_up_down", "confusion_up_flat", "confusion_up_up"
                ).sum()
            ).item()
        )
        total = down + flat + up
        rows.append(
            {
                "instrument": instrument,
                "observations": total,
                "down_count": down,
                "down_percent": down / total,
                "flat_count": flat,
                "flat_percent": flat / total,
                "up_count": up,
                "up_percent": up / total,
            }
        )
    return pl.DataFrame(rows)


def _confusion_matrices(*, metrics: pl.DataFrame) -> pl.DataFrame:
    classes = ("down", "flat", "up")
    rows: list[dict[str, object]] = []
    dimensions = (
        metrics.select("instrument", "model").unique().sort("instrument", "model")
    )
    for dimension in dimensions.iter_rows(named=True):
        selected = metrics.filter(
            (pl.col("instrument") == dimension["instrument"])
            & (pl.col("model") == dimension["model"])
        )
        seeds = int(selected["seed"].n_unique())
        for true_class in classes:
            counts = [
                float(selected[f"confusion_{true_class}_{predicted}"].sum()) / seeds
                for predicted in classes
            ]
            total = sum(counts)
            rows.append(
                {
                    **dimension,
                    "true_class": true_class,
                    "predicted_down_count": counts[0],
                    "predicted_flat_count": counts[1],
                    "predicted_up_count": counts[2],
                    "true_class_count": total,
                    "predicted_down_percent": counts[0] / total,
                    "predicted_flat_percent": counts[1] / total,
                    "predicted_up_percent": counts[2] / total,
                }
            )
    return pl.DataFrame(rows)


def _training_details(*, paths: dict[str, Path], metrics: pl.DataFrame) -> pl.DataFrame:
    parameters = {
        str(row["model"]): int(row["parameter_count"])
        for row in metrics.select("model", "parameter_count")
        .unique()
        .iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for model in ("deeplob_direction", "tft_direction"):
        for seed in (7, 19):
            summary = _json_mapping(paths[f"locked_cell_{model}_seed_{seed}"])
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "horizon_seconds": 30,
                    "depth": 1,
                    "fixed_epochs": _integer(summary, "fixed_epochs"),
                    "parameter_count": parameters[model],
                    "fit_elapsed_seconds": _finite_number(
                        summary.get("fit_elapsed_seconds")
                    ),
                    "locked_evaluation_used": summary.get("locked_evaluation_used"),
                }
            )
    return pl.DataFrame(rows)


def _forest_svg(*, evidence: pl.DataFrame, title: str) -> str:
    rows = list(evidence.iter_rows(named=True))
    width = 1100
    left = 430
    right = 40
    top = 70
    row_height = 29
    height = top + row_height * len(rows) + 45
    lower = min(float(row["confidence_lower"]) for row in rows)
    upper = max(float(row["confidence_upper"]) for row in rows)
    extent = max(abs(lower), abs(upper), 0.001) * 1.12
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + ((value + extent) / (2.0 * extent)) * plot_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="32" font-family="sans-serif" font-size="20" '
        f'font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="{x(0):.2f}" y1="52" x2="{x(0):.2f}" '
        f'y2="{height - 30}" stroke="#555" stroke-width="1"/>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        instrument = str(row["instrument"]).replace("_", "/")
        model = str(row["model"]).replace("_direction", "")
        horizon = int(cast(int, row["horizon_milliseconds"])) // 1000
        label = f"{instrument} | {model} | {horizon}s | {row['metric']}"
        color = "#1769aa" if bool(row["metric_passed"]) else "#b23a48"
        lo = x(float(row["confidence_lower"]))
        hi = x(float(row["confidence_upper"]))
        mean = x(float(row["mean_delta"]))
        elements.extend(
            [
                f'<text x="20" y="{y + 5}" font-family="monospace" '
                f'font-size="13">{html.escape(label)}</text>',
                f'<line x1="{lo:.2f}" y1="{y}" x2="{hi:.2f}" y2="{y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{mean:.2f}" cy="{y}" r="5" fill="{color}"/>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left}" y="{height - 10}" font-family="sans-serif" '
            f'font-size="12">{-extent:.3f}</text>',
            f'<text x="{x(0) - 10:.2f}" y="{height - 10}" '
            'font-family="sans-serif" font-size="12">0</text>',
            f'<text x="{width - right - 45}" y="{height - 10}" '
            f'font-family="sans-serif" font-size="12">{extent:.3f}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def _write_workbook(
    *,
    path: Path,
    summary: dict[str, object],
    development: pl.DataFrame,
    locked: pl.DataFrame,
    cross: pl.DataFrame,
    absolute_metrics: pl.DataFrame,
    model_summary: pl.DataFrame,
    data_usage: pl.DataFrame,
    class_balance: pl.DataFrame,
    confusion_matrices: pl.DataFrame,
    training_details: pl.DataFrame,
    session_deltas: pl.DataFrame,
    locked_metrics: pl.DataFrame,
    cross_metrics: pl.DataFrame,
    dates: list[str],
) -> None:
    development_primary = development.filter(pl.col("metric").is_in(_PRIMARY_METRICS))
    locked_primary = locked.filter(pl.col("metric").is_in(_PRIMARY_METRICS))
    cross_primary = cross.filter(pl.col("metric").is_in(_PRIMARY_METRICS))
    stability = (
        session_deltas.group_by("stage", "instrument", "model")
        .agg(
            pl.len().alias("sessions"),
            (pl.col("macro_f1_delta") > 0).sum().alias("macro_f1_positive"),
            (pl.col("mcc_delta") > 0).sum().alias("mcc_positive"),
            ((pl.col("macro_f1_delta") > 0) & (pl.col("mcc_delta") > 0))
            .sum()
            .alias("both_positive"),
        )
        .sort("stage", "instrument", "model")
    )
    workbook = xlsxwriter.Workbook(path)
    workbook.set_properties(
        {
            "title": "EBS TFT 2024 evidence report",
            "subject": "Frozen Q1 2024 direction-classification results",
            "author": "EBS TFT research workflow",
            "comments": "Reporting only; no post-locked retuning permitted.",
        }
    )
    formats = _workbook_formats(workbook=workbook)
    _write_summary_sheet(
        workbook=workbook,
        formats=formats,
        summary=summary,
        dates=dates,
        model_summary=model_summary,
        data_usage=data_usage,
    )
    model_sheet = _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Model Summary",
        data=model_summary,
    )
    _add_model_charts(
        workbook=workbook,
        sheet=model_sheet,
        data=model_summary,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Data Usage",
        data=data_usage,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Class Balance",
        data=class_balance,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Confusion Matrices",
        data=confusion_matrices,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Training Details",
        data=training_details,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Primary Evidence",
        data=pl.concat([development_primary, locked_primary, cross_primary]),
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Absolute Metrics",
        data=absolute_metrics,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Development",
        data=development,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Locked Evaluation",
        data=locked,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Cross Instrument",
        data=cross,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Session Deltas",
        data=session_deltas,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Session Stability",
        data=stability,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="EURUSD Raw Sessions",
        data=locked_metrics,
    )
    _write_frame_sheet(
        workbook=workbook,
        formats=formats,
        name="Transfer Raw Sessions",
        data=cross_metrics,
    )
    _write_notes_sheet(workbook=workbook, formats=formats)
    workbook.close()


def _workbook_formats(
    *, workbook: xlsxwriter.Workbook
) -> dict[str, xlsxwriter.format.Format]:
    return {
        "title": workbook.add_format(
            {"bold": True, "font_size": 18, "font_color": "#17365D"}
        ),
        "subtitle": workbook.add_format(
            {"bold": True, "font_size": 11, "font_color": "#404040"}
        ),
        "section": workbook.add_format(
            {
                "bold": True,
                "font_size": 12,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
            }
        ),
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#4472C4",
                "border": 1,
                "text_wrap": True,
            }
        ),
        "text": workbook.add_format({"text_wrap": True, "valign": "top"}),
        "number": workbook.add_format({"num_format": "0.000000"}),
        "percent": workbook.add_format({"num_format": "0.00%"}),
        "integer": workbook.add_format({"num_format": "0"}),
        "pass": workbook.add_format(
            {"bold": True, "bg_color": "#C6EFCE", "font_color": "#006100"}
        ),
        "fail": workbook.add_format(
            {"bold": True, "bg_color": "#FFC7CE", "font_color": "#9C0006"}
        ),
        "warning": workbook.add_format(
            {"text_wrap": True, "bg_color": "#FFF2CC", "font_color": "#7F6000"}
        ),
    }


def _write_summary_sheet(
    *,
    workbook: xlsxwriter.Workbook,
    formats: dict[str, xlsxwriter.format.Format],
    summary: dict[str, object],
    dates: list[str],
    model_summary: pl.DataFrame,
    data_usage: pl.DataFrame,
) -> None:
    sheet = workbook.add_worksheet("Executive Summary")
    sheet.hide_gridlines(2)
    sheet.set_tab_color("#1F4E78")
    sheet.set_column("A:A", 22)
    sheet.set_column("B:B", 20)
    sheet.set_column("C:E", 15)
    sheet.set_column("F:F", 18)
    sheet.set_column("G:N", 14)
    sheet.merge_range("A1:N1", "EBS TFT — Q1 2024 Model Evidence", formats["title"])
    sheet.merge_range(
        "A2:N2",
        "Frozen direction-classification evaluation; all comparisons are against "
        "the same-data logistic reference.",
        formats["subtitle"],
    )
    rows = [
        ("Conclusion", summary["headline_conclusion"]),
        (
            "Interpretation",
            "Promising research evidence at 30 seconds, but not enough to claim a "
            "production-ready or profitable trading system.",
        ),
        (
            "Evidence period",
            f"{summary['evidence_period_start']} to {summary['evidence_period_end']}",
        ),
        (
            "Raw-data scope",
            "January–March 2024 only; this is not a full calendar-year study.",
        ),
        ("Development training sessions", summary["development_training_sessions"]),
        ("Rolling validation sessions", summary["development_sessions"]),
        ("Locked final dates", ", ".join(dates)),
        ("Neural benchmark cells", summary["neural_benchmark_cells"]),
        ("Confirmed EUR/USD candidates", summary["locked_confirmed_candidates"]),
        ("Confirmed transfers", summary["confirmed_cross_instrument_transfers"]),
        ("Retuning permitted", "No"),
        (
            "External-year test",
            "Not run; no external-year data is currently available.",
        ),
    ]
    sheet.write_row(3, 0, ("Study scope", "Value"), formats["header"])
    sheet.merge_range(3, 1, 3, 13, "Value", formats["header"])
    for row_index, (label, value) in enumerate(rows, start=4):
        sheet.write(row_index, 0, label)
        sheet.merge_range(row_index, 1, row_index, 13, value, formats["text"])
    sheet.set_row(4, 44)

    table_row = 18
    headers = (
        "Stage",
        "Instrument",
        "Model",
        "Sessions",
        "Observations",
        "Macro F1",
        "Logistic F1",
        "F1 delta",
        "F1 95% CI",
        "MCC",
        "Logistic MCC",
        "MCC delta",
        "MCC 95% CI",
        "Decision",
    )
    sheet.merge_range(
        table_row - 1,
        0,
        table_row - 1,
        len(headers) - 1,
        "Final model tests — absolute scores and paired statistical comparison",
        formats["section"],
    )
    sheet.write_row(table_row, 0, headers, formats["header"])
    for offset, row in enumerate(model_summary.iter_rows(named=True), start=1):
        values = (
            _display_stage(str(row["stage"])),
            str(row["instrument"]).replace("_", "/"),
            _display_model(str(row["model"])),
            row["sessions"],
            row["evaluation_observations"],
            row["macro_f1"],
            row["logistic_macro_f1"],
            row["macro_f1_delta"],
            _interval(row["macro_f1_ci_lower"], row["macro_f1_ci_upper"]),
            row["mcc"],
            row["logistic_mcc"],
            row["mcc_delta"],
            _interval(row["mcc_ci_lower"], row["mcc_ci_upper"]),
            "PASS" if row["primary_gate_passed"] else "FAIL",
        )
        for column_index, value in enumerate(values):
            cell_format = None
            if column_index in {3, 4}:
                cell_format = formats["integer"]
            elif column_index in {5, 6, 7, 9, 10, 11}:
                cell_format = formats["number"]
            elif column_index == 13:
                cell_format = (
                    formats["pass"] if row["primary_gate_passed"] else formats["fail"]
                )
            sheet.write(table_row + offset, column_index, value, cell_format)

    note_row = table_row + model_summary.height + 2
    sheet.merge_range(
        note_row,
        0,
        note_row,
        13,
        "Statistical unit warning: observations are overlapping 100-ms windows. "
        "The confidence intervals resample sessions, so the confirmatory sample "
        "size is four sessions per instrument—not millions of independent cases.",
        formats["warning"],
    )
    sheet.set_row(note_row, 34)
    usage_row = note_row + 2
    sheet.merge_range(
        usage_row,
        0,
        usage_row,
        13,
        "Data actually used in final evaluation",
        formats["section"],
    )
    usage_headers = (
        "Instrument",
        "Training sessions",
        "Rolling validation sessions",
        "Final sessions",
        "Final observations",
        "Minimum/session",
        "Maximum/session",
        "Final dates",
    )
    sheet.write_row(usage_row + 1, 0, usage_headers, formats["header"])
    for offset, row in enumerate(data_usage.iter_rows(named=True), start=2):
        usage_values = (
            str(row["instrument"]).replace("_", "/"),
            row["neural_training_sessions"],
            row["rolling_development_validation_sessions"],
            row["final_evaluation_sessions"],
            row["unique_final_observations"],
            row["minimum_observations_per_session"],
            row["maximum_observations_per_session"],
            row["final_evaluation_dates"],
        )
        sheet.write_row(usage_row + offset, 0, usage_values)
    sheet.freeze_panes(table_row + 1, 0)


def _write_frame_sheet(
    *,
    workbook: xlsxwriter.Workbook,
    formats: dict[str, xlsxwriter.format.Format],
    name: str,
    data: pl.DataFrame,
) -> xlsxwriter.worksheet.Worksheet:
    sheet = workbook.add_worksheet(name)
    sheet.freeze_panes(1, 0)
    sheet.autofilter(0, 0, data.height, len(data.columns) - 1)
    for column_index, column in enumerate(data.columns):
        sheet.write(0, column_index, column, formats["header"])
        values = data[column].to_list()
        width = min(
            42, max(len(column) + 2, *(len(str(value)) + 2 for value in values))
        )
        sheet.set_column(column_index, column_index, width)
        for row_index, value in enumerate(values, start=1):
            cell_format = None
            if isinstance(value, bool):
                cell_format = formats["pass"] if value else formats["fail"]
            elif isinstance(value, float):
                cell_format = (
                    formats["percent"] if "percent" in column else formats["number"]
                )
            elif isinstance(value, int):
                cell_format = formats["integer"]
            sheet.write(row_index, column_index, value, cell_format)
    return sheet


def _add_model_charts(
    *,
    workbook: xlsxwriter.Workbook,
    sheet: xlsxwriter.worksheet.Worksheet,
    data: pl.DataFrame,
) -> None:
    first_row = 1
    last_row = data.height
    categories = ["Model Summary", first_row, 2, last_row, 2]
    for title, neural_column, logistic_column, position in (
        ("Macro F1: neural versus logistic", 7, 8, "W2"),
        ("MCC: neural versus logistic", 12, 13, "W20"),
    ):
        chart = workbook.add_chart({"type": "column"})
        chart.add_series(
            {
                "name": "Neural model",
                "categories": categories,
                "values": [
                    "Model Summary",
                    first_row,
                    neural_column,
                    last_row,
                    neural_column,
                ],
                "fill": {"color": "#4472C4"},
            }
        )
        chart.add_series(
            {
                "name": "Logistic reference",
                "categories": categories,
                "values": [
                    "Model Summary",
                    first_row,
                    logistic_column,
                    last_row,
                    logistic_column,
                ],
                "fill": {"color": "#A5A5A5"},
            }
        )
        chart.set_title({"name": title})
        chart.set_y_axis({"major_gridlines": {"visible": True}})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 720, "height": 330})
        sheet.insert_chart(position, chart)


def _display_stage(stage: str) -> str:
    return "EUR/USD locked" if stage == "locked_evaluation" else "Frozen transfer"


def _display_model(model: str) -> str:
    return {"deeplob_direction": "DeepLOB", "tft_direction": "TFT"}.get(model, model)


def _interval(lower: object, upper: object) -> str:
    return f"[{float(cast(float, lower)):.6f}, {float(cast(float, upper)):.6f}]"


def _write_notes_sheet(
    *,
    workbook: xlsxwriter.Workbook,
    formats: dict[str, xlsxwriter.format.Format],
) -> None:
    sheet = workbook.add_worksheet("Statistical Notes")
    sheet.hide_gridlines(2)
    sheet.set_column("A:A", 28)
    sheet.set_column("B:B", 110)
    sheet.write_row(0, 0, ("Topic", "Explanation"), formats["header"])
    notes = [
        (
            "Macro F1",
            "Ranges from 0 to 1; higher is better. It gives down, flat, and up "
            "equal importance even when their frequencies differ.",
        ),
        (
            "MCC",
            "Ranges from -1 to 1; higher is better. Zero is roughly no overall "
            "correlation and one is perfect classification.",
        ),
        (
            "Log loss and Brier",
            "Probability-quality measures; lower is better. They penalize "
            "confident incorrect forecasts.",
        ),
        (
            "Delta",
            "Neural-model metric minus the logistic-reference metric after "
            "averaging neural seeds within each session.",
        ),
        (
            "Confidence interval",
            "Predeclared 95% paired-session bootstrap interval. Sessions, not "
            "overlapping 100 ms observations, are the resampling units.",
        ),
        (
            "Absolute Metrics sheet",
            "For every final model, instrument, and metric: mean, sample standard "
            "deviation, minimum, and maximum across the four sessions.",
        ),
        (
            "Confusion matrices",
            "Rows are true down/flat/up classes and columns are predicted classes. "
            "Neural counts are averaged across the two frozen random seeds.",
        ),
        (
            "Strict pass rule",
            "Both macro F1 and MCC must have confidence lower bounds strictly "
            "above zero.",
        ),
        (
            "Supporting metrics",
            "Balanced accuracy, log loss, and multiclass Brier score are reported "
            "but do not control the primary decision.",
        ),
        (
            "Small sample",
            "Only four locked dates are available. This sharply limits "
            "generalization even when the interval excludes zero.",
        ),
        (
            "Cross-instrument dependence",
            "The three instruments share evaluation dates, so their results are "
            "not independent replications.",
        ),
        (
            "Economic limitation",
            "Prediction quality is not trading profitability. Costs, spread, "
            "latency, fills, inventory, and a trading policy were not tested.",
        ),
        (
            "Development detail",
            "The downloaded archive contains complete five-metric aggregate "
            "comparisons across 20 validation sessions, but not the development "
            "stage's raw per-session table.",
        ),
        (
            "No retuning",
            "Locked and transfer outcomes may be reported but must not be used to "
            "retune or rerun the frozen experiment.",
        ),
    ]
    for row_index, (topic, explanation) in enumerate(notes, start=1):
        sheet.write(row_index, 0, topic)
        sheet.write(row_index, 1, explanation, formats["text"])
        sheet.set_row(row_index, 34)


def _normalized_decision(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    for key in (
        "accepted_model_depth_horizons",
        "confirmed_candidates",
        "confirmed_transfers",
    ):
        items = normalized.get(key)
        if isinstance(items, list):
            normalized[key] = sorted(
                items, key=lambda item: json.dumps(item, sort_keys=True)
            )
    return normalized


def _read_csv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except Exception as exc:
        raise ValueError(f"unable to read evidence table: {path}") from exc


def _json_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read evidence JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evidence JSON must contain an object: {path}")
    return cast(dict[str, object], value)


def _write_json(*, path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _list_length(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return len(value)


def _plan_session_dates(*, plan: dict[str, object], key: str) -> list[str]:
    value = plan.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"locked plan {key} must be a non-empty list")
    dates: list[str] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("trading_date"), str):
            raise ValueError(f"locked plan {key} contains an invalid session")
        dates.append(cast(str, item["trading_date"]))
    if len(set(dates)) != len(dates):
        raise ValueError(f"locked plan {key} contains duplicate sessions")
    return sorted(dates)


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a finite numeric summary")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite numeric summary")
    return result


def _single_integer(data: pl.DataFrame, *, column: str) -> int:
    values = data[column].unique().to_list()
    if (
        len(values) != 1
        or isinstance(values[0], bool)
        or not isinstance(values[0], int)
    ):
        raise ValueError(f"{column} must contain exactly one integer")
    return values[0]


def _single_year(*, dates: list[str]) -> int:
    years = {int(value[:4]) for value in dates}
    if len(years) != 1:
        raise ValueError("locked dates must belong to one study year")
    return years.pop()
