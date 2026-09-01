"""Run and summarize a bounded collection of comparable pilot samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import attrs
import polars as pl
import yaml

from ebs_tft.application.usecases.pilot import _comparison, _config, _runner


class UnableToLoadPilotMatrixError(Exception):
    """Indicate that the pilot-matrix YAML is unreadable or invalid."""


@attrs.frozen
class PilotMatrixResult:
    """Reference the combined artifacts from a multi-date pilot matrix."""

    output_dir: Path
    metrics_path: Path
    target_balance_path: Path
    depth_comparison_path: Path
    summary_path: Path


def run_from_config(
    *,
    path: Path,
    reuse_existing: bool = False,
    replace_output: bool = False,
) -> PilotMatrixResult:
    """Execute every configured sample and persist comparable combined outputs."""
    config_paths, output_dir = _load_matrix(path=path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pl.DataFrame] = []
    balance_frames: list[pl.DataFrame] = []
    run_summaries: list[dict[str, object]] = []
    for config_path in config_paths:
        specification = _config.load_specification(path=config_path)
        result = (
            _existing_result(output_dir=specification.output_dir)
            if reuse_existing
            else _runner.run(specification=specification, replace_output=replace_output)
        )
        metric_frames.append(
            pl.read_csv(result.metrics_path).with_columns(
                pl.lit(specification.instrument.value).alias("instrument"),
                pl.lit(specification.trading_date).alias("trading_date"),
            )
        )
        balance_frames.append(
            pl.read_csv(result.output_dir / "target_balance.csv").with_columns(
                pl.lit(specification.instrument.value).alias("instrument"),
                pl.lit(specification.trading_date).alias("trading_date"),
            )
        )
        run_summaries.append(
            {
                "config": str(config_path),
                "output_dir": str(result.output_dir),
                "instrument": specification.instrument.value,
                "trading_date": specification.trading_date.isoformat(),
            }
        )

    metrics = pl.concat(metric_frames).sort(
        [
            "trading_date",
            "horizon_steps",
            "training_mode",
            "model",
            "depth",
            "seed",
        ]
    )
    metrics_path = output_dir / "matrix_metrics.csv"
    metrics.write_csv(metrics_path)
    target_balance = pl.concat(balance_frames).sort(["trading_date", "horizon_steps"])
    target_balance_path = output_dir / "matrix_target_balance.csv"
    target_balance.write_csv(target_balance_path)
    depth_comparison = _comparison.cumulative_depth_comparison(metrics=metrics)
    depth_comparison_path = output_dir / "depth_comparison.csv"
    depth_comparison.write_csv(depth_comparison_path)
    aggregated = _aggregate_metrics(metrics=metrics)
    aggregated_path = output_dir / "aggregate_metrics.csv"
    aggregated.write_csv(aggregated_path)
    summary = {
        "warning": (
            "Bounded multi-date engineering pilot only; not publication evidence."
        ),
        "runs": run_summaries,
        "artifacts": {
            "metrics": str(metrics_path),
            "target_balance": str(target_balance_path),
            "aggregate_metrics": str(aggregated_path),
            "depth_comparison": str(depth_comparison_path),
        },
    }
    summary_path = output_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    terminal_summary = "\n".join(
        (
            "EBS bounded multi-date pilot matrix completed",
            "WARNING: diagnostic only; not publication evidence.",
            "",
            "Aggregate neural-model metrics:",
            str(aggregated),
            "",
            "Paired cumulative-depth diagnostic:",
            str(depth_comparison),
            "",
            f"Outputs: {output_dir}",
        )
    )
    (output_dir / "terminal_summary.txt").write_text(terminal_summary, encoding="utf-8")
    print(terminal_summary)
    return PilotMatrixResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        target_balance_path=target_balance_path,
        depth_comparison_path=depth_comparison_path,
        summary_path=summary_path,
    )


def _existing_result(*, output_dir: Path) -> _runner.PilotResult:
    result = _runner.PilotResult(
        output_dir=output_dir,
        native_states_path=output_dir / "native_states.parquet",
        metrics_path=output_dir / "metrics.csv",
        predictions_path=output_dir / "predictions.parquet",
        summary_path=output_dir / "run_summary.json",
        terminal_summary_path=output_dir / "terminal_summary.txt",
    )
    required_paths = (
        result.native_states_path,
        result.metrics_path,
        result.predictions_path,
        result.summary_path,
        result.terminal_summary_path,
        output_dir / "target_balance.csv",
    )
    missing_paths = [str(item) for item in required_paths if not item.is_file()]
    if missing_paths:
        joined = ", ".join(missing_paths)
        raise UnableToLoadPilotMatrixError(
            f"cannot reuse incomplete pilot output; missing: {joined}"
        )
    return result


def _load_matrix(*, path: Path) -> tuple[tuple[Path, ...], Path]:
    try:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise UnableToLoadPilotMatrixError(
            f"Unable to load pilot matrix: {path}"
        ) from exc
    if not isinstance(loaded, dict) or set(loaded) != {
        "schema_version",
        "pilot_configs",
        "output_dir",
    }:
        raise UnableToLoadPilotMatrixError("pilot matrix keys differ from schema")
    data = cast(dict[str, object], loaded)
    if data["schema_version"] != 1:
        raise UnableToLoadPilotMatrixError("unsupported pilot matrix schema_version")
    raw_configs = data["pilot_configs"]
    raw_output = data["output_dir"]
    if (
        not isinstance(raw_configs, list)
        or not raw_configs
        or any(not isinstance(item, str) or not item for item in raw_configs)
    ):
        raise UnableToLoadPilotMatrixError(
            "pilot_configs must be a non-empty list of paths"
        )
    if not isinstance(raw_output, str) or not raw_output:
        raise UnableToLoadPilotMatrixError("output_dir must be a non-empty path")
    base_dir = path.resolve().parent
    config_paths = tuple((base_dir / item).resolve() for item in raw_configs)
    if len(set(config_paths)) != len(config_paths):
        raise UnableToLoadPilotMatrixError("pilot_configs must be unique")
    return config_paths, (base_dir / raw_output).resolve()


def _aggregate_metrics(*, metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        metrics.filter(pl.col("seed") >= 0)
        .group_by(
            [
                "horizon_steps",
                "horizon_milliseconds",
                "training_mode",
                "model",
                "depth",
            ]
        )
        .agg(
            pl.len().alias("runs"),
            pl.col("balanced_accuracy").mean(),
            pl.col("macro_f1").mean(),
            pl.col("weighted_f1").mean(),
            pl.col("mcc").mean(),
            pl.col("log_loss").mean(),
            pl.col("multiclass_brier").mean(),
            pl.col("calibration_error").mean(),
            pl.col("recall_down").mean(),
            pl.col("recall_flat").mean(),
            pl.col("recall_up").mean(),
        )
        .sort(["horizon_steps", "training_mode", "model", "depth"])
    )
