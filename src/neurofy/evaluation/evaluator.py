"""Model evaluation."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from neurofy.evaluation.metrics import DECISION_THRESHOLD, compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and per-sample outputs for one data split."""

    metrics: dict[str, float]
    labels: np.ndarray
    probabilities: np.ndarray
    threshold: float = DECISION_THRESHOLD


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate `model` over every batch, preserving sample order."""
    model.eval()
    total_loss = 0.0
    labels, probabilities = [], []

    for features, targets in batches:
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
