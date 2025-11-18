from __future__ import annotations

import torch
from torch import nn


class FailurePredictorModel(nn.Module):
    """
    GRU 기반 시계열 이진 분류 모델.
    입력: (batch, seq_len, input_dim)
    출력: (batch,)  # 장애 발생 확률
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        probs = torch.sigmoid(logits)
        return probs.squeeze(-1)
