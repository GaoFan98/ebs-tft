"""A simple LSTM comparator inspired by Ito, Iima, and Kitamura (2022).

The paper's individual order events cannot be recovered from end-of-slice EBS
snapshots. These adapters therefore compare architectures and available inputs
on this project's native-resolution task; they are not paper reproductions.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ebs_tft.domain.pilot import training as pilot_training

FEATURE_SETS = ("level_1_and_transactions", "transactions_only")
_TRANSACTION_FEATURES = (
    "log_buy_volume",
    "log_sell_volume",
    "log_trade_count",
    "deal_flow_imbalance",
    "deals_observed",
)
_TRANSACTION_INDICES = tuple(
    pilot_training.AUXILIARY_FEATURE_ORDER.index(name) for name in _TRANSACTION_FEATURES
)


class PaperLstmDirectionClassifier(nn.Module):
    """Classify one native 100 ms window with a one-layer, 50-unit LSTM."""

    def __init__(self, *, feature_set: str, hidden_size: int = 50) -> None:
        super().__init__()
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unsupported LSTM feature set: {feature_set}")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self._feature_set = feature_set
        input_size = (
            len(_TRANSACTION_INDICES)
            if feature_set == "transactions_only"
            else len(pilot_training.LOB_FEATURE_ORDER)
            + len(pilot_training.AUXILIARY_FEATURE_ORDER)
        )
        self._lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self._head = nn.Linear(hidden_size, 3)

    def forward(
        self, lob_features: torch.Tensor, auxiliary_features: torch.Tensor
    ) -> torch.Tensor:
        """Return three logits, using only causally available input features."""
        if lob_features.ndim != 4 or lob_features.shape[2:] != (
            1,
            len(pilot_training.LOB_FEATURE_ORDER),
        ):
            raise ValueError("paper-inspired LSTM requires Level-1 book features")
        if (
            auxiliary_features.ndim != 3
            or auxiliary_features.shape[:2] != lob_features.shape[:2]
            or auxiliary_features.shape[2]
            != len(pilot_training.AUXILIARY_FEATURE_ORDER)
        ):
            raise ValueError("auxiliary features have an incompatible shape")
        if self._feature_set == "transactions_only":
            features = auxiliary_features[:, :, _TRANSACTION_INDICES]
        else:
            features = torch.cat((lob_features[:, :, 0, :], auxiliary_features), dim=-1)
        sequence, _ = self._lstm(features)
        return cast(torch.Tensor, self._head(sequence[:, -1, :]))
