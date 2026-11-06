"""Model checkpoints."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neurofy.config import ExperimentConfig


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ExperimentConfig,
    epoch: int,
    score: float,
) -> None:
    """Save model weights with the config and the epoch's validation score."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "score": score,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Load weights into `model` and return the full checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
