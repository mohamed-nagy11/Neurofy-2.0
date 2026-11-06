"""MLflow reporting of evaluation results."""

import mlflow

from neurofy.evaluation.evaluator import EvaluationResult
from neurofy.evaluation.plots import plot_confusion_matrix, plot_roc_curve


def log_evaluation(result: EvaluationResult, split: str) -> None:
    """Log metrics and plots for an evaluated split to the active MLflow run."""
    labels, probabilities = result.labels, result.probabilities

    mlflow.log_metrics({f"{split}/{name}": value for name, value in result.metrics.items()})
    mlflow.log_dict(result.metrics, f"metrics/{split}.json")
    mlflow.log_figure(
        plot_confusion_matrix(labels, probabilities),
        f"plots/{split}_confusion_matrix.png",
    )
    if "auc_roc" in result.metrics:
        mlflow.log_figure(plot_roc_curve(labels, probabilities), f"plots/{split}_roc_curve.png")
