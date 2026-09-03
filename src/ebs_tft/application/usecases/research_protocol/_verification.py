"""Verify the declared capabilities and limitations of direction adapters."""

from __future__ import annotations

import json
from pathlib import Path

import attrs
import torch

from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models


@attrs.frozen
class ModelProtocolVerificationResult:
    """Reference one machine-readable model-protocol verification report."""

    output_dir: Path
    report_path: Path


def run(
    *, protocol: research_models.ResearchProtocol, replace_output: bool = False
) -> ModelProtocolVerificationResult:
    """Verify adapter shapes, depth capacity, required modules, and disclosure."""
    output_dir = protocol.output_dir / "model_protocol"
    artifact_repository.prepare_run_directory(path=output_dir, replace=replace_output)
    checks = (
        _check_model(model_name="deeplob_direction"),
        _check_model(model_name="tft_direction"),
    )
    report = {
        "model_protocol_version": model_domain.MODEL_PROTOCOL_VERSION,
        "all_capability_checks_passed": all(item["checks_passed"] for item in checks),
        "scientific_disclosure": (
            "These are documented EBS direction adapters, not exact reproductions "
            "of the original DeepLOB or TFT training protocols."
        ),
        "input_contract": {
            "native_state_interval_milliseconds": (
                protocol.state_interval_milliseconds
            ),
            "lob_feature_order": list(pilot_training.LOB_FEATURE_ORDER),
            "auxiliary_feature_order": list(pilot_training.AUXILIARY_FEATURE_ORDER),
            "depth_axis_preserved": True,
            "input_aggregation": False,
        },
        "optimization_contract": {
            "validation_checks_per_epoch": protocol.validation_checks_per_epoch,
            "best_validation_state_restored": True,
            "training_windows_are_horizon_spaced": True,
        },
        "models": list(checks),
    }
    report_path = output_dir / "model_protocol_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "EBS model-protocol verification completed\n"
        f"all_capability_checks_passed={report['all_capability_checks_passed']}\n"
        f"outputs={output_dir}"
    )
    return ModelProtocolVerificationResult(
        output_dir=output_dir, report_path=report_path
    )


def _check_model(*, model_name: str) -> dict[str, object]:
    hidden_size = 32
    auxiliary_size = len(pilot_training.AUXILIARY_FEATURE_ORDER)
    classifier: torch.nn.Module
    required_types: tuple[type[torch.nn.Module], ...]
    if model_name == "deeplob_direction":
        classifier = model_domain.DeepLobDirectionClassifier(
            auxiliary_size=auxiliary_size, hidden_size=hidden_size
        )
        required_types = (torch.nn.Conv1d, torch.nn.LSTM)
    elif model_name == "tft_direction":
        classifier = model_domain.TftDirectionClassifier(
            auxiliary_size=auxiliary_size, hidden_size=hidden_size
        )
        required_types = (torch.nn.LSTM, torch.nn.MultiheadAttention)
    else:
        raise ValueError(f"unsupported model adapter: {model_name}")
    classifier.eval()
    output_shapes: dict[str, list[int]] = {}
    finite_outputs = True
    with torch.no_grad():
        for depth in (1, 10):
            logits = classifier(
                torch.zeros((2, 100, depth, 6), dtype=torch.float32),
                torch.zeros((2, 100, auxiliary_size), dtype=torch.float32),
            )
            output_shapes[f"l{depth}"] = list(logits.shape)
            finite_outputs = finite_outputs and bool(torch.isfinite(logits).all())
    module_types = tuple(type(module) for module in classifier.modules())
    required_modules_present = all(
        any(issubclass(actual, required) for actual in module_types)
        for required in required_types
    )
    checks_passed = (
        finite_outputs
        and required_modules_present
        and all(shape == [2, 3] for shape in output_shapes.values())
    )
    return {
        "model": model_name,
        "status": "ebs_adaptation_not_exact_reference_replication",
        "parameter_count": model_domain.parameter_count(classifier=classifier),
        "required_modules": [item.__name__ for item in required_types],
        "required_modules_present": required_modules_present,
        "output_shapes": output_shapes,
        "finite_outputs": finite_outputs,
        "checks_passed": checks_passed,
    }
