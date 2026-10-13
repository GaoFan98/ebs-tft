"""Check the isolated paper-inspired architecture and input ablation."""

from __future__ import annotations

import pytest
import torch

from ebs_tft.domain import model


class TestPaperLstmDirectionClassifier:
    def test_full_input_returns_three_logits(self) -> None:
        classifier = model.PaperLstmDirectionClassifier(
            feature_set="level_1_and_transactions"
        )

        logits = classifier(torch.zeros(2, 100, 1, 6), torch.zeros(2, 100, 10))

        assert logits.shape == (2, 3)
        assert classifier._lstm.input_size == 16
        assert classifier._lstm.hidden_size == 50
        assert classifier._lstm.num_layers == 1

    def test_transactions_only_ignores_quote_and_nontransaction_inputs(self) -> None:
        classifier = model.PaperLstmDirectionClassifier(feature_set="transactions_only")
        auxiliary = torch.zeros(2, 4, 10)
        changed_auxiliary = auxiliary.clone()
        changed_auxiliary[:, :, 5:] = 100

        original = classifier(torch.zeros(2, 4, 1, 6), auxiliary)
        changed = classifier(torch.ones(2, 4, 1, 6), changed_auxiliary)

        torch.testing.assert_close(original, changed)
        assert classifier._lstm.input_size == 5

    def test_rejects_deeper_book_and_unknown_feature_set(self) -> None:
        classifier = model.PaperLstmDirectionClassifier(
            feature_set="level_1_and_transactions"
        )

        with pytest.raises(ValueError, match="Level-1"):
            classifier(torch.zeros(2, 4, 10, 6), torch.zeros(2, 4, 10))
        with pytest.raises(ValueError, match="unsupported"):
            model.PaperLstmDirectionClassifier(feature_set="paper_exact_events")
