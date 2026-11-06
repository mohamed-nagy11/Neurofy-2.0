"""Model evaluation."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neurofy.evaluation.metrics import compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and per-sample outputs for one data split."""

    metrics: dict[str, float]
    labels: np.ndarray
    probabilities: np.ndarray


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate `model` over every sample in `dataloader`."""
    model.eval()
    total_loss = 0.0
    labels, probabilities = [], []

    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        labels.append(targets.cpu())
        probabilities.append(torch.sigmoid(logits).cpu())

    labels = torch.cat(labels).numpy()
    probabilities = torch.cat(probabilities).numpy()
    metrics = {
        "loss": total_loss / len(labels),
        **compute_metrics(labels, probabilities),
    }
    return EvaluationResult(metrics, labels, probabilities)
