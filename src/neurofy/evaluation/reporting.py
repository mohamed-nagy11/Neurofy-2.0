"""MLflow reporting of evaluation results."""

import mlflow

from neurofy.evaluation.aggregation import (
    FoldResult,
    rows_to_csv,
    summarize_cohort,
    summarize_patients,
)
from neurofy.evaluation.evaluator import EvaluationResult
from neurofy.evaluation.plots import plot_confusion_matrix, plot_patient_metrics, plot_roc_curve
from neurofy.utils.tracking import finished_fold_runs


def log_evaluation(result: EvaluationResult, split: str) -> None:
    """Log metrics and plots for an evaluated split to the active MLflow run."""
    labels, probabilities = result.labels, result.probabilities

    mlflow.log_metrics({f"{split}/{name}": value for name, value in result.metrics.items()})
    mlflow.log_dict(result.metrics, f"metrics/{split}.json")
    mlflow.log_figure(
        plot_confusion_matrix(labels, probabilities, result.threshold),
        f"plots/{split}_confusion_matrix.png",
    )
    if "auc_roc" in result.metrics:
        mlflow.log_figure(plot_roc_curve(labels, probabilities), f"plots/{split}_roc_curve.png")


def fold_results(experiment_name: str, config_name: str) -> list[FoldResult]:
    """Read final metrics of an experiment's finished folds from MLflow."""
    return [
        FoldResult(run.data.tags["patient"], run.data.tags["seizure"], run.data.metrics)
        for run in finished_fold_runs(experiment_name, config_name)
    ]


def log_experiment_report(experiment_name: str, config_name: str) -> dict[str, dict[str, float]]:
    """Log patient and cohort summaries to the active MLflow run and return the cohort summary."""
    rows = summarize_patients(fold_results(experiment_name, config_name))
    if not rows:
        return {}
    cohort = summarize_cohort(rows)

    mlflow.log_text(rows_to_csv(rows), "patient_metrics.csv")
    mlflow.log_dict(cohort, "cohort_metrics.json")
    mlflow.log_metric("cohort/patients", len(rows))
    mlflow.log_metric("cohort/folds", sum(row["folds"] for row in rows))
    for metric, values in cohort.items():
        mlflow.log_metric(f"cohort/{metric}/mean", values["mean"])
        mlflow.log_metric(f"cohort/{metric}/std", values["std"])
    mlflow.log_figure(
        plot_patient_metrics(rows, ("segment/accuracy", "segment/fpr", "event/sensitivity")),
        "plots/patient_metrics.png",
    )
    return cohort
