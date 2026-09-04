"""Exercise the pre-GPU research protocol across real file boundaries."""

from __future__ import annotations

import datetime
import gzip
import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from ebs_tft.application.usecases import research_protocol
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.research import models as research_models


def test_audit_manifest_and_baseline_gate_remain_chronological(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    dates = tuple(datetime.date(2024, 1, day) for day in range(1, 6))
    for trading_date in dates:
        _write_session(data_dir=data_dir, trading_date=trading_date)
    protocol = _protocol(
        data_dir=data_dir, output_dir=output_dir, locked_date=dates[-1]
    )
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("schema_version: 1\n", encoding="utf-8")

    audit = research_protocol.run_session_audit(
        protocol=protocol, protocol_path=protocol_path, replace_output=False
    )

    audit_data = pl.read_csv(audit.audit_path)
    assert audit_data.height == 5
    locked = audit_data.filter(pl.col("evaluation_locked"))
    assert locked.height == 1
    assert locked["total_h100"].null_count() == 1
    with audit.manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    folds = manifest["development_folds"]["EUR_USD"]
    assert len(folds) == 3
    assert folds[0]["training_sessions"][0]["trading_date"] == "2024-01-01"
    assert folds[-1]["validation_sessions"][0]["trading_date"] == "2024-01-04"
    assert manifest["final_test_sessions"]["EUR_USD"][0]["trading_date"] == (
        "2024-01-05"
    )

    baseline = research_protocol.run_baseline_gate(
        protocol=protocol, protocol_path=protocol_path, replace_output=False
    )

    metrics = pl.read_csv(baseline.metrics_path)
    comparisons = pl.read_csv(baseline.comparisons_path)
    assert metrics["validation_date"].n_unique() == 3
    assert set(metrics["depth"].unique()) == {1, 10}
    assert set(metrics["model"].unique()) == {
        "empirical_prior",
        "last_move",
        "logistic",
        "majority",
    }
    assert comparisons["sessions"].min() == 3
    assert baseline.gate_path.is_file()

    (baseline.output_dir / "run_summary.json").unlink()
    resumed = research_protocol.run_baseline_gate(
        protocol=protocol, protocol_path=protocol_path, replace_output=False
    )
    assert "resumed_folds=3" in resumed.terminal_summary_path.read_text(
        encoding="utf-8"
    )

    replaced = research_protocol.run_baseline_gate(
        protocol=protocol, protocol_path=protocol_path, replace_output=True
    )
    assert "resumed_folds=0" in replaced.terminal_summary_path.read_text(
        encoding="utf-8"
    )

    research_protocol.run_model_protocol_verification(
        protocol=protocol, replace_output=False
    )
    research_protocol.run_model_protocol_verification(
        protocol=protocol, replace_output=True
    )
    gate = json.loads(replaced.gate_path.read_text(encoding="utf-8"))
    gate["baseline_signal_by_horizon"] = {"100": True}
    gate["depth_support_by_horizon"] = {"100": True}
    gate["eligible_for_neural_benchmark"] = True
    replaced.gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    policy_path = tmp_path / "neural_policy.yaml"
    policy_path.write_text("schema_version: 1\n", encoding="utf-8")
    policy = research_models.NeuralBenchmarkPolicy(
        maximum_epochs=1,
        early_stopping_patience=1,
        early_stopping_minimum_delta=0.0001,
        gradient_clip_norm=1.0,
        batch_size=8,
        learning_rate=0.0003,
        weight_decay=0.0001,
        hidden_size=8,
        device="cpu",
    )

    with pytest.raises(research_protocol.NeuralBenchmarkPausedError) as pause:
        research_protocol.run_neural_benchmark(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            replace_output=False,
            maximum_new_cells=1,
        )
    assert pause.value.completed_cells == 1
    assert (pause.value.output_dir / "progress_summary.json").is_file()

    neural = research_protocol.run_neural_benchmark(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        replace_output=False,
    )

    neural_metrics = pl.read_csv(neural.metrics_path)
    assert neural_metrics.height == 24
    assert neural_metrics["validation_date"].n_unique() == 3
    assert set(neural_metrics["depth"].unique()) == {1, 10}
    comparisons = pl.read_csv(neural.comparisons_path)
    assert set(comparisons["depth"].unique()) == {1, 10}
    assert neural.gate_path.is_file()
    assert "resumed_cells=1" in neural.terminal_summary_path.read_text(encoding="utf-8")
    assert not (neural.output_dir / "progress_summary.json").exists()
    prediction_dates = set(
        pl.scan_parquet(neural.output_dir / "cells" / "**" / "predictions.parquet")
        .select("validation_date")
        .collect()["validation_date"]
        .unique()
    )
    assert dates[-1] not in prediction_dates

    (neural.output_dir / "run_summary.json").unlink()
    resumed_neural = research_protocol.run_neural_benchmark(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        replace_output=False,
    )
    assert "resumed_cells=24" in resumed_neural.terminal_summary_path.read_text(
        encoding="utf-8"
    )


def _protocol(
    *, data_dir: Path, output_dir: Path, locked_date: datetime.date
) -> research_models.ResearchProtocol:
    return research_models.ResearchProtocol(
        data_dir=data_dir,
        output_dir=output_dir,
        instruments=(orderbook_models.Instrument.EUR_USD,),
        years=(2024,),
        state_interval_milliseconds=100,
        forecast_horizons_milliseconds=(100,),
        context_milliseconds=200,
        maximum_staleness_milliseconds=1_000,
        audit_workers=1,
        training_stride_milliseconds=((100, 100),),
        evaluation_stride_milliseconds=100,
        audit_policy=research_models.AuditPolicy(
            minimum_duration_milliseconds=1_000,
            minimum_observed_states=10,
            required_depth=10,
            redact_locked_outcomes=True,
        ),
        split_policy=research_models.SplitPolicy(
            development_end_date=datetime.date(2024, 1, 4),
            minimum_training_sessions=1,
            validation_sessions_per_fold=1,
            fold_step_sessions=1,
            locked_evaluation_dates=(locked_date,),
        ),
        development_instrument=orderbook_models.Instrument.EUR_USD,
        depths=(1, 10),
        models=("deeplob_direction", "tft_direction"),
        random_seeds=(7, 19),
        validation_checks_per_epoch=2,
        primary_metrics=(
            research_models.EvaluationMetric.MACRO_F1,
            research_models.EvaluationMetric.MCC,
        ),
        supporting_metrics=(
            research_models.EvaluationMetric.BALANCED_ACCURACY,
            research_models.EvaluationMetric.LOG_LOSS,
            research_models.EvaluationMetric.MULTICLASS_BRIER,
        ),
        bootstrap_repetitions=1_000,
        confidence_level=0.95,
    )


def _write_session(*, data_dir: Path, trading_date: datetime.date) -> None:
    year_dir = data_dir / str(trading_date.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / (f"{trading_date:%Y%m%d}-EBS_LVL2_EUR_USD_0.csv.gz")
    mid_offsets = (0, 1, 1, 0, -1, -1)
    rows: list[str] = []
    for step in range(30):
        timestamp = datetime.datetime.combine(
            trading_date, datetime.time(), tzinfo=datetime.UTC
        ) + datetime.timedelta(milliseconds=step * 100)
        mid = 1.10000 + mid_offsets[step % len(mid_offsets)] * 0.00001
        for side in (0, 1):
            for level in range(1, 11):
                direction = -1 if side == 0 else 1
                price = mid + direction * (0.00001 * level)
                rows.append(
                    f"{timestamp:%Y/%m/%d},{timestamp:%H:%M:%S.%f}"[:-3]
                    + f",EUR/USD,Q,{side},{level},{price:.5f},1000000,1\n"
                )
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        stream.writelines(rows)
