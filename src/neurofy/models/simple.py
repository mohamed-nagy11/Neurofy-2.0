"""Placeholder model architecture."""

import torch
from torch import nn

from neurofy.config import BaselineConfig


class SimpleModel(nn.Module):
    """Linear binary classifier that outputs one logit per sample."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def create_model(config: BaselineConfig) -> nn.Module:
    """Create the model described by `config`."""
    return SimpleModel(config.input_dim)
