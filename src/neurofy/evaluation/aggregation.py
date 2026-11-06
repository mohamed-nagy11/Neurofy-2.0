"""Aggregation of fold results into patient and cohort summaries."""

import csv
import io
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

REPORTED_METRICS = (
    "segment/accuracy",
    "segment/precision",
    "segment/recall",
    "segment/f1",
    "segment/fpr",
    "segment/auc_roc",
    "event/sensitivity",
    "event/interictal_alarm",
    "window/accuracy",
    "window/fpr",
)


@dataclass(frozen=True)
class FoldResult:
    """Final metrics of one fold."""

    patient: str
    seizure: str
    metrics: dict[str, float]


def summarize_patients(folds: list[FoldResult]) -> list[dict]:
    """Average each metric over a patient's folds, ordered by patient."""
    by_patient: dict[str, list[FoldResult]] = defaultdict(list)
    for fold in folds:
        by_patient[fold.patient].append(fold)

    rows = []
    for patient in sorted(by_patient):
        patient_folds = by_patient[patient]
        row = {"patient": patient, "folds": len(patient_folds)}
        for metric in REPORTED_METRICS:
            values = [fold.metrics[metric] for fold in patient_folds if metric in fold.metrics]
            row[metric] = float(np.mean(values)) if values else float("nan")
        rows.append(row)
    return rows


def summarize_cohort(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean and standard deviation of each metric across patients."""
    summary = {}
    for metric in REPORTED_METRICS:
        values = np.array([row[metric] for row in rows], dtype=float)
        values = values[~np.isnan(values)]
        if values.size:
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
    return summary


def rows_to_csv(rows: list[dict]) -> str:
    """Serialize summary rows as CSV text."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
