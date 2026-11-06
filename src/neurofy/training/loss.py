"""Loss functions."""

from torch import nn


def get_loss_fn() -> nn.Module:
    """Return binary cross-entropy computed on logits."""
    return nn.BCEWithLogitsLoss()
