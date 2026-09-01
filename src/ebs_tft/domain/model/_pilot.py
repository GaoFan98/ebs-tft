"""Provide compact direction classifiers for local feasibility testing."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from typing import cast

import attrs
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data

_DIRECTION_CLASSES: int = 3


@attrs.frozen
class PredictionBatch:
    """Return aligned class labels and probabilities from one classifier."""

    labels: np.ndarray
    probabilities: np.ndarray


class TrainingDivergedError(Exception):
    """Indicate that model optimization produced a non-finite value."""


@attrs.frozen
class EpochMetric:
    """Record training and validation behavior after one complete epoch."""

    epoch: int
    training_loss: float
    validation_log_loss: float
    gradient_norm: float
    improved: bool


@attrs.frozen
class TrainingState:
    """Capture everything required to resume from an epoch boundary."""

    epoch: int
    classifier_state: Mapping[str, torch.Tensor]
    optimizer_state: Mapping[str, object]
    best_classifier_state: Mapping[str, torch.Tensor]
    best_epoch: int
    best_validation_log_loss: float
    stalled_validations: int
    history: tuple[EpochMetric, ...]
    torch_random_state: torch.Tensor


@attrs.frozen
class TrainingResult:
    """Describe a completed fit whose classifier is restored to its best epoch."""

    history: tuple[EpochMetric, ...]
    best_epoch: int
    best_validation_log_loss: float
    epochs_completed: int
    stop_reason: str
    latest_state: TrainingState


class SequenceDataset(torch_data.Dataset[tuple[torch.Tensor, ...]]):
    """Expose lazy overlapping windows over one immutable feature matrix."""

    def __init__(
        self,
        *,
        lob_features: np.ndarray,
        auxiliary_features: np.ndarray,
        labels: np.ndarray,
        target_indices: np.ndarray,
        context_steps: int,
    ) -> None:
        self._lob_features = torch.from_numpy(lob_features).float()
        self._auxiliary_features = torch.from_numpy(auxiliary_features).float()
        self._labels = torch.from_numpy(labels).long()
        self._target_indices = torch.from_numpy(target_indices).long()
        self._context_steps = context_steps

    def __len__(self) -> int:
        """Return the number of selected target windows."""
        return len(self._target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return one context window and its target without copying the corpus."""
        target_index = int(self._target_indices[index])
        start = target_index - self._context_steps + 1
        return (
            self._lob_features[start : target_index + 1],
            self._auxiliary_features[start : target_index + 1],
            self._labels[target_index],
            self._target_indices[index],
        )

    def target_labels(self) -> np.ndarray:
        """Return labels aligned to this dataset's selected target windows."""
        return self._labels[self._target_indices].numpy()


class DeepLobDirectionClassifier(nn.Module):
    """Adapt DeepLOB's spatial-CNN/Inception/LSTM design to EBS classification.

    Transaction and state-quality inputs use a separate temporal branch instead
    of being repeated artificially across order-book levels.
    """

    def __init__(self, *, auxiliary_size: int, hidden_size: int) -> None:
        super().__init__()
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four")
        branch_size = hidden_size // 4
        self._level_encoder = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.GELU(),
        )
        self._spatial_encoder = nn.Sequential(
            nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=(1, 3),
                padding=(0, 1),
            ),
            nn.GELU(),
        )
        self._inception_one = nn.Conv1d(hidden_size, branch_size, kernel_size=1)
        self._inception_three = nn.Conv1d(
            hidden_size, branch_size, kernel_size=3, padding=1
        )
        self._inception_five = nn.Conv1d(
            hidden_size, branch_size, kernel_size=5, padding=2
        )
        self._inception_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, branch_size, kernel_size=1),
        )
        self._inception_activation = nn.Sequential(
            nn.GELU(),
        )
        self._auxiliary_encoder = nn.Sequential(
            nn.Linear(auxiliary_size, hidden_size // 2),
            nn.GELU(),
        )
        self._temporal = nn.LSTM(
            input_size=hidden_size + hidden_size // 2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self._classifier = nn.Linear(hidden_size, _DIRECTION_CLASSES)

    def forward(
        self, lob_features: torch.Tensor, auxiliary_features: torch.Tensor
    ) -> torch.Tensor:
        """Return down/flat/up logits for a batch of context windows."""
        encoded_levels = self._level_encoder(lob_features)
        spatial = self._spatial_encoder(encoded_levels.permute(0, 3, 1, 2))
        temporal_book = spatial.mean(dim=3)
        branches = (
            self._inception_one(temporal_book),
            self._inception_three(temporal_book),
            self._inception_five(temporal_book),
            self._inception_pool(temporal_book),
        )
        encoded_book = self._inception_activation(torch.cat(branches, dim=1)).permute(
            0, 2, 1
        )
        encoded_auxiliary = self._auxiliary_encoder(auxiliary_features)
        sequence = torch.cat((encoded_book, encoded_auxiliary), dim=2)
        temporal, _ = self._temporal(sequence)
        return cast(torch.Tensor, self._classifier(temporal[:, -1]))


class _GatedResidualNetwork(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self._dense = nn.Linear(hidden_size, hidden_size)
        self._gate = nn.Linear(hidden_size, hidden_size * 2)
        self._normalization = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.nn.functional.elu(self._dense(values))
        gated = torch.nn.functional.glu(self._gate(transformed), dim=-1)
        return cast(torch.Tensor, self._normalization(values + gated))


class TftDirectionClassifier(nn.Module):
    """Adapt TFT's selection, recurrence, gating, and attention to one-step classes."""

    def __init__(
        self, *, input_size: int, hidden_size: int, attention_heads: int = 4
    ) -> None:
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self._variable_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.GELU(),
                    _GatedResidualNetwork(hidden_size=hidden_size),
                )
                for _ in range(input_size)
            ]
        )
        self._selection = nn.Linear(hidden_size, 1)
        self._recurrent = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self._recurrent_gate = _GatedResidualNetwork(hidden_size=hidden_size)
        self._attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self._attention_gate = _GatedResidualNetwork(hidden_size=hidden_size)
        self._classifier = nn.Linear(hidden_size, _DIRECTION_CLASSES)

    def forward(
        self, lob_features: torch.Tensor, auxiliary_features: torch.Tensor
    ) -> torch.Tensor:
        """Return down/flat/up logits for a batch of context windows."""
        batch_size, context_steps = lob_features.shape[:2]
        flattened_book = lob_features.reshape(batch_size, context_steps, -1)
        features = torch.cat((flattened_book, auxiliary_features), dim=2)
        embedded = torch.stack(
            [
                encoder(features[:, :, index : index + 1])
                for index, encoder in enumerate(self._variable_encoders)
            ],
            dim=2,
        )
        selection_weights = torch.softmax(self._selection(embedded).squeeze(-1), dim=2)
        selected = (embedded * selection_weights.unsqueeze(-1)).sum(dim=2)
        recurrent, _ = self._recurrent(selected)
        recurrent = self._recurrent_gate(recurrent)
        causal_mask = torch.triu(
            torch.full(
                (context_steps, context_steps),
                float("-inf"),
                device=recurrent.device,
            ),
            diagonal=1,
        )
        attended, _ = self._attention(
            recurrent,
            recurrent,
            recurrent,
            attn_mask=causal_mask,
            need_weights=False,
        )
        attended = self._attention_gate(recurrent + attended)
        return cast(torch.Tensor, self._classifier(attended[:, -1]))


def select_device(*, requested: str) -> torch.device:
    """Resolve one explicit compute device without silently changing a request."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def set_random_seed(*, seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible pilot run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def fit_classifier(
    *,
    classifier: nn.Module,
    training_data: SequenceDataset,
    validation_data: SequenceDataset,
    device: torch.device,
    maximum_epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    early_stopping_minimum_delta: float,
    gradient_clip_norm: float,
    random_seed: int,
    class_weights: np.ndarray | None = None,
    resume_state: TrainingState | None = None,
    epoch_observer: Callable[[EpochMetric], None] | None = None,
    checkpoint_observer: Callable[[TrainingState], None] | None = None,
) -> TrainingResult:
    """Fit one classifier, restore its best epoch, and return resumable state."""
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    weight_tensor = (
        torch.from_numpy(class_weights).float().to(device)
        if class_weights is not None
        else None
    )
    loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
    history: list[EpochMetric] = []
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    best_validation_loss = math.inf
    stalled_validations = 0
    start_epoch = 1
    if resume_state is not None:
        classifier.load_state_dict(resume_state.classifier_state)
        optimizer.load_state_dict(cast(dict[str, object], resume_state.optimizer_state))
        history.extend(resume_state.history)
        best_state = _copy_state(state=resume_state.best_classifier_state)
        best_epoch = resume_state.best_epoch
        best_validation_loss = resume_state.best_validation_log_loss
        stalled_validations = resume_state.stalled_validations
        start_epoch = resume_state.epoch + 1
        torch.random.set_rng_state(resume_state.torch_random_state.cpu())
    latest_state = resume_state
    stop_reason = (
        "early_stopping"
        if resume_state is not None
        and resume_state.stalled_validations >= early_stopping_patience
        else "maximum_epochs"
    )
    epoch_range = (
        range(start_epoch, maximum_epochs + 1)
        if stalled_validations < early_stopping_patience
        else range(0)
    )
    for epoch in epoch_range:
        generator = torch.Generator().manual_seed(random_seed + epoch)
        training_loader = torch_data.DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        classifier.train()
        training_loss = 0.0
        examples = 0
        maximum_observed_gradient_norm = 0.0
        for lob_features, auxiliary_features, labels, _ in training_loader:
            lob_features = lob_features.to(device)
            auxiliary_features = auxiliary_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(lob_features, auxiliary_features)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise TrainingDivergedError("training loss is not finite")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    classifier.parameters(), max_norm=gradient_clip_norm
                )
                .detach()
                .cpu()
            )
            if not math.isfinite(gradient_norm):
                raise TrainingDivergedError("gradient norm is not finite")
            maximum_observed_gradient_norm = max(
                maximum_observed_gradient_norm, gradient_norm
            )
            optimizer.step()
            training_loss += float(loss.detach().cpu()) * len(labels)
            examples += len(labels)
        validation = predict_classifier(
            classifier=classifier,
            dataset=validation_data,
            device=device,
            batch_size=batch_size,
        )
        validation_labels = validation_data.target_labels()
        clipped = np.clip(validation.probabilities, 1e-12, 1.0)
        validation_loss = -float(
            np.log(clipped[np.arange(len(validation_labels)), validation_labels]).mean()
        )
        if not math.isfinite(validation_loss):
            raise TrainingDivergedError("validation log loss is not finite")
        improved = validation_loss < (
            best_validation_loss - early_stopping_minimum_delta
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = _copy_state(state=classifier.state_dict())
            stalled_validations = 0
        else:
            stalled_validations += 1
        epoch_metric = EpochMetric(
            epoch=epoch,
            training_loss=training_loss / max(examples, 1),
            validation_log_loss=validation_loss,
            gradient_norm=maximum_observed_gradient_norm,
            improved=improved,
        )
        history.append(epoch_metric)
        latest_state = TrainingState(
            epoch=epoch,
            classifier_state=_copy_state(state=classifier.state_dict()),
            optimizer_state=cast(Mapping[str, object], optimizer.state_dict()),
            best_classifier_state=_copy_state(state=best_state),
            best_epoch=best_epoch,
            best_validation_log_loss=best_validation_loss,
            stalled_validations=stalled_validations,
            history=tuple(history),
            torch_random_state=torch.random.get_rng_state().clone(),
        )
        if epoch_observer is not None:
            epoch_observer(epoch_metric)
        if checkpoint_observer is not None:
            checkpoint_observer(latest_state)
        if stalled_validations >= early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if latest_state is None or not best_state:
        raise ValueError("maximum_epochs must allow at least one epoch")
    classifier.load_state_dict(best_state)
    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_log_loss=best_validation_loss,
        epochs_completed=latest_state.epoch,
        stop_reason=stop_reason,
        latest_state=latest_state,
    )


def _copy_state(*, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def predict_classifier(
    *,
    classifier: nn.Module,
    dataset: SequenceDataset,
    device: torch.device,
    batch_size: int,
) -> PredictionBatch:
    """Return probabilities and labels in deterministic dataset order."""
    classifier.to(device)
    classifier.eval()
    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for lob_features, auxiliary_features, _, _ in loader:
            logits = classifier(lob_features.to(device), auxiliary_features.to(device))
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(batch_probabilities)
            labels.append(batch_probabilities.argmax(axis=1))
    if not probabilities:
        raise ValueError("prediction dataset must not be empty")
    return PredictionBatch(
        labels=np.concatenate(labels),
        probabilities=np.concatenate(probabilities),
    )


def parameter_count(*, classifier: nn.Module) -> int:
    """Return the number of trainable parameters in one classifier."""
    return sum(
        math.prod(parameter.shape)
        for parameter in classifier.parameters()
        if parameter.requires_grad
    )
