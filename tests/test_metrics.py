import numpy as np
import pytest

from neurofy.evaluation.metrics import compute_metrics


def test_compute_metrics_matches_hand_counts() -> None:
    # TP=1, FN=1, FP=1, TN=2
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    probabilities = np.array([0.9, 0.2, 0.7, 0.1, 0.3])

    metrics = compute_metrics(labels, probabilities)

    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(1 / 3)


def test_compute_metrics_omits_auc_for_single_class() -> None:
    metrics = compute_metrics(np.zeros(4), np.array([0.1, 0.2, 0.6, 0.3]))

    assert "auc_roc" not in metrics
    assert metrics["fpr"] == pytest.approx(0.25)
