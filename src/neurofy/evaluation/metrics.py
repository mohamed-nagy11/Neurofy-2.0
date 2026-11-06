"""Binary classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DECISION_THRESHOLD = 0.5


def to_predictions(
    probabilities: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> np.ndarray:
    """Convert scores to 0/1 predictions."""
    return (probabilities >= threshold).astype(int)


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Compute metrics over a full split; `auc_roc` requires both classes to be present."""
    labels = labels.astype(int)
    predictions = to_predictions(probabilities, threshold)
    tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
    }
    if len(np.unique(labels)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(labels, probabilities))
    return metrics
