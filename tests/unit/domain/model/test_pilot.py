"""Test local-pilot classifier interfaces."""

from __future__ import annotations

import math
import unittest.mock
from typing import cast

import numpy as np
import torch

from ebs_tft.domain import model


class TestSequenceDataset:
    def test_vectorized_batch_matches_scalar_collation(self) -> None:
        rows = 12
        features = np.arange(rows * 2 * 6, dtype=np.float32).reshape(rows, 2, 6)
        auxiliary = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
        labels = (np.arange(rows) % 3).astype(np.int64)
        dataset = model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            target_indices=np.array([3, 6, 9]),
            context_steps=3,
        )
        indices = torch.tensor([2, 0])

        actual = dataset.batch(indices=indices)
        expected = torch.utils.data.default_collate(
            [dataset[int(index)] for index in indices]
        )

        assert all(
            torch.equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )


class TestDeepLobDirectionClassifier:
    def test_returns_three_logits_for_each_window(self) -> None:
        classifier = model.DeepLobDirectionClassifier(auxiliary_size=10, hidden_size=16)
        lob_features = torch.zeros((4, 32, 10, 6))
        auxiliary_features = torch.zeros((4, 32, 10))

        actual = classifier(lob_features, auxiliary_features)

        assert actual.shape == (4, 3)

    def test_supports_level_one_without_inactive_level_padding(self) -> None:
        classifier = model.DeepLobDirectionClassifier(auxiliary_size=10, hidden_size=16)
        lob_features = torch.zeros((4, 32, 1, 6))
        auxiliary_features = torch.zeros((4, 32, 10))

        actual = classifier(lob_features, auxiliary_features)

        assert actual.shape == (4, 3)

    def test_one_classifier_accepts_depth_one_and_ten_at_fixed_capacity(self) -> None:
        classifier = model.DeepLobDirectionClassifier(auxiliary_size=10, hidden_size=16)
        auxiliary_features = torch.zeros((2, 8, 10))
        parameter_count = model.parameter_count(classifier=classifier)

        classifier(torch.zeros((2, 8, 1, 6)), auxiliary_features)
        classifier(torch.zeros((2, 8, 10, 6)), auxiliary_features)

        assert model.parameter_count(classifier=classifier) == parameter_count


class TestTftDirectionClassifier:
    def test_returns_three_logits_for_each_window(self) -> None:
        classifier = model.TftDirectionClassifier(
            auxiliary_size=10, hidden_size=16, attention_heads=4
        )
        lob_features = torch.zeros((4, 32, 10, 6))
        auxiliary_features = torch.zeros((4, 32, 10))

        actual = classifier(lob_features, auxiliary_features)

        assert actual.shape == (4, 3)

    def test_supports_level_one_without_inactive_level_features(self) -> None:
        classifier = model.TftDirectionClassifier(
            auxiliary_size=10, hidden_size=16, attention_heads=4
        )
        lob_features = torch.zeros((4, 32, 1, 6))
        auxiliary_features = torch.zeros((4, 32, 10))

        actual = classifier(lob_features, auxiliary_features)

        assert actual.shape == (4, 3)

    def test_one_classifier_accepts_depth_one_and_ten_at_fixed_capacity(self) -> None:
        classifier = model.TftDirectionClassifier(
            auxiliary_size=10, hidden_size=16, attention_heads=4
        )
        auxiliary_features = torch.zeros((2, 8, 10))
        parameter_count = model.parameter_count(classifier=classifier)

        classifier(torch.zeros((2, 8, 1, 6)), auxiliary_features)
        classifier(torch.zeros((2, 8, 10, 6)), auxiliary_features)

        assert model.parameter_count(classifier=classifier) == parameter_count


class _BiasClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))

    def forward(
        self, lob_features: torch.Tensor, auxiliary_features: torch.Tensor
    ) -> torch.Tensor:
        del auxiliary_features
        return self.bias.expand(len(lob_features), -1)


class _AuxiliaryClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3, 3)

    def forward(
        self, lob_features: torch.Tensor, auxiliary_features: torch.Tensor
    ) -> torch.Tensor:
        del lob_features
        return cast(torch.Tensor, self.classifier(auxiliary_features[:, -1, :3]))


class TestFitClassifier:
    def test_preserves_the_legacy_seeded_data_loader_order(self) -> None:
        training, validation = _learnable_datasets()
        generator = torch.Generator().manual_seed(18)
        legacy_order = torch.cat(
            [
                target_indices
                for _, _, _, target_indices in torch.utils.data.DataLoader(
                    training,
                    batch_size=15,
                    shuffle=True,
                    num_workers=0,
                    generator=generator,
                )
            ]
        )

        with unittest.mock.patch.object(
            training, "batch", wraps=training.batch
        ) as vectorized_batch:
            _fit(
                classifier=_AuxiliaryClassifier(),
                training=training,
                validation=validation,
                maximum_epochs=1,
            )

        training_batch_count = math.ceil(len(training) / 15)
        vectorized_order = torch.cat(
            [
                call.kwargs["indices"]
                for call in vectorized_batch.call_args_list[:training_batch_count]
            ]
        )
        assert torch.equal(vectorized_order, legacy_order)

    def test_checks_validation_within_one_epoch_when_configured(self) -> None:
        training, validation = _learnable_datasets()

        actual = model.fit_classifier(
            classifier=_AuxiliaryClassifier(),
            training_data=training,
            validation_data=validation,
            device=torch.device("cpu"),
            maximum_epochs=1,
            batch_size=6,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=10,
            early_stopping_minimum_delta=0.0001,
            gradient_clip_norm=1.0,
            random_seed=7,
            validation_checks_per_epoch=3,
        )

        assert len(actual.history) == 3
        assert [item.validation_index for item in actual.history] == [1, 2, 3]
        assert [item.optimizer_step for item in actual.history] == [4, 8, 10]

    def test_stops_after_configured_non_improving_validations(self) -> None:
        features = np.zeros((30, 1, 6), dtype=np.float32)
        auxiliary = np.zeros((30, 10), dtype=np.float32)
        labels = (np.arange(30) % 3).astype(np.int64)
        training = model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            target_indices=np.arange(18),
            context_steps=1,
        )
        validation = model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            target_indices=np.arange(18, 30),
            context_steps=1,
        )
        classifier = _BiasClassifier()

        actual = model.fit_classifier(
            classifier=classifier,
            training_data=training,
            validation_data=validation,
            device=torch.device("cpu"),
            maximum_epochs=10,
            batch_size=6,
            learning_rate=0.0,
            weight_decay=0.0,
            early_stopping_patience=2,
            early_stopping_minimum_delta=0.0001,
            gradient_clip_norm=1.0,
            random_seed=7,
        )

        assert actual.best_epoch == 1
        assert actual.epochs_completed == 3
        assert actual.stop_reason == "early_stopping"
        assert len(actual.history) == 3
        assert torch.equal(
            classifier.state_dict()["bias"],
            actual.latest_state.best_classifier_state["bias"],
        )

    def test_resumed_epoch_boundary_matches_uninterrupted_cpu_training(self) -> None:
        training, validation = _learnable_datasets()
        torch.manual_seed(11)
        partial_classifier = _AuxiliaryClassifier()
        partial = _fit(
            classifier=partial_classifier,
            training=training,
            validation=validation,
            maximum_epochs=2,
        )
        torch.manual_seed(999)
        resumed_classifier = _AuxiliaryClassifier()
        resumed = _fit(
            classifier=resumed_classifier,
            training=training,
            validation=validation,
            maximum_epochs=4,
            resume_state=partial.latest_state,
        )
        torch.manual_seed(11)
        uninterrupted_classifier = _AuxiliaryClassifier()
        uninterrupted = _fit(
            classifier=uninterrupted_classifier,
            training=training,
            validation=validation,
            maximum_epochs=4,
        )

        assert resumed.history == uninterrupted.history
        for name, value in resumed_classifier.state_dict().items():
            assert torch.equal(value, uninterrupted_classifier.state_dict()[name])

    def test_completed_checkpoint_can_finish_interrupted_artifact_write(self) -> None:
        training, validation = _learnable_datasets()
        classifier = _AuxiliaryClassifier()
        completed = _fit(
            classifier=classifier,
            training=training,
            validation=validation,
            maximum_epochs=2,
        )
        reloaded_classifier = _AuxiliaryClassifier()

        resumed = _fit(
            classifier=reloaded_classifier,
            training=training,
            validation=validation,
            maximum_epochs=2,
            resume_state=completed.latest_state,
        )

        assert resumed.history == completed.history
        assert resumed.epochs_completed == 2
        assert resumed.stop_reason == "maximum_epochs"
        for name, value in reloaded_classifier.state_dict().items():
            assert torch.equal(
                value, completed.latest_state.best_classifier_state[name]
            )

    def test_early_stopped_checkpoint_does_not_run_an_extra_epoch(self) -> None:
        features = np.zeros((30, 1, 6), dtype=np.float32)
        auxiliary = np.zeros((30, 10), dtype=np.float32)
        labels = (np.arange(30) % 3).astype(np.int64)
        training = model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            target_indices=np.arange(18),
            context_steps=1,
        )
        validation = model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            target_indices=np.arange(18, 30),
            context_steps=1,
        )
        classifier = _BiasClassifier()
        completed = model.fit_classifier(
            classifier=classifier,
            training_data=training,
            validation_data=validation,
            device=torch.device("cpu"),
            maximum_epochs=10,
            batch_size=6,
            learning_rate=0.0,
            weight_decay=0.0,
            early_stopping_patience=2,
            early_stopping_minimum_delta=0.0001,
            gradient_clip_norm=1.0,
            random_seed=7,
        )
        observer = unittest.mock.Mock()

        resumed = model.fit_classifier(
            classifier=_BiasClassifier(),
            training_data=training,
            validation_data=validation,
            device=torch.device("cpu"),
            maximum_epochs=10,
            batch_size=6,
            learning_rate=0.0,
            weight_decay=0.0,
            early_stopping_patience=2,
            early_stopping_minimum_delta=0.0001,
            gradient_clip_norm=1.0,
            random_seed=7,
            resume_state=completed.latest_state,
            epoch_observer=observer,
        )

        assert resumed.epochs_completed == completed.epochs_completed
        assert resumed.stop_reason == "early_stopping"
        observer.assert_not_called()


class TestPredictClassifier:
    def test_moves_a_fresh_checkpoint_model_to_the_requested_device(self) -> None:
        features = np.zeros((3, 1, 6), dtype=np.float32)
        auxiliary = np.zeros((3, 10), dtype=np.float32)
        dataset = model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=np.arange(3, dtype=np.int64),
            target_indices=np.arange(3),
            context_steps=1,
        )
        classifier = _BiasClassifier()
        requested_device = torch.device("cpu")

        with unittest.mock.patch.object(
            classifier, "to", wraps=classifier.to
        ) as move_to_device:
            actual = model.predict_classifier(
                classifier=classifier,
                dataset=dataset,
                device=requested_device,
                batch_size=2,
            )

        move_to_device.assert_called_once_with(requested_device)
        assert actual.probabilities.shape == (3, 3)

    def test_inference_is_stable_across_evaluation_batch_sizes(self) -> None:
        generator = np.random.default_rng(7)
        rows = 24
        dataset = model.SequenceDataset(
            lob_features=generator.normal(size=(rows, 1, 6)).astype(np.float32),
            auxiliary_features=generator.normal(size=(rows, 10)).astype(np.float32),
            labels=(np.arange(rows) % 3).astype(np.int64),
            target_indices=np.arange(3, rows),
            context_steps=4,
        )
        torch.manual_seed(11)
        classifier = model.DeepLobDirectionClassifier(auxiliary_size=10, hidden_size=8)

        one_at_a_time = model.predict_classifier(
            classifier=classifier,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=1,
        )
        batched = model.predict_classifier(
            classifier=classifier,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=16,
        )

        np.testing.assert_array_equal(one_at_a_time.labels, batched.labels)
        np.testing.assert_allclose(
            one_at_a_time.probabilities,
            batched.probabilities,
            rtol=1e-6,
            atol=1e-7,
        )


def _learnable_datasets() -> tuple[model.SequenceDataset, model.SequenceDataset]:
    rows = 90
    labels = (np.arange(rows) % 3).astype(np.int64)
    features = np.zeros((rows, 1, 6), dtype=np.float32)
    auxiliary = np.zeros((rows, 10), dtype=np.float32)
    auxiliary[np.arange(rows), labels] = 2.0
    return (
        model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            context_steps=1,
            target_indices=np.arange(60),
        ),
        model.SequenceDataset(
            lob_features=features,
            auxiliary_features=auxiliary,
            labels=labels,
            context_steps=1,
            target_indices=np.arange(60, 90),
        ),
    )


def _fit(
    *,
    classifier: torch.nn.Module,
    training: model.SequenceDataset,
    validation: model.SequenceDataset,
    maximum_epochs: int,
    resume_state: model.TrainingState | None = None,
) -> model.TrainingResult:
    return model.fit_classifier(
        classifier=classifier,
        training_data=training,
        validation_data=validation,
        device=torch.device("cpu"),
        maximum_epochs=maximum_epochs,
        batch_size=15,
        learning_rate=0.01,
        weight_decay=0.0,
        early_stopping_patience=10,
        early_stopping_minimum_delta=0.0,
        gradient_clip_norm=1.0,
        random_seed=17,
        resume_state=resume_state,
    )
