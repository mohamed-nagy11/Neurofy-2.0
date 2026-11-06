"""Comparison of results with the paper and between models."""

import numpy as np
from scipy.stats import wilcoxon

# CHB-MIT Fp1 values from Jang et al. (2026), Supp. Tables 4-5 and Table 4.
PAPER_TARGETS = {
    "segment/accuracy": {"patient_tables": 0.9072, "table4_mean": 0.902, "table4_std": 0.155},
    "segment/precision": {"table4_mean": 0.919, "table4_std": 0.189},
    "segment/recall": {"table4_mean": 0.903, "table4_std": 0.220},
    "segment/f1": {"table4_mean": 0.892, "table4_std": 0.191},
    "segment/fpr": {"patient_tables": 0.092, "table4_mean": 0.099, "table4_std": 0.253},
    "event/sensitivity": {"patient_tables": 0.9792},
}

COMPARED_METRICS = (
    "segment/accuracy",
    "segment/fpr",
    "event/sensitivity",
    "event/interictal_alarm",
    "window/accuracy",
)


def compare_with_paper(cohort: dict[str, dict[str, float]]) -> list[dict]:
    """Place cohort results next to the paper's reported values."""
    rows = []
    for metric, targets in PAPER_TARGETS.items():
        if metric in cohort:
            rows.append(
                {
                    "metric": metric,
                    "mean": cohort[metric]["mean"],
                    "std": cohort[metric]["std"],
                    "paper_patient_tables": targets.get("patient_tables", ""),
                    "paper_table4_mean": targets.get("table4_mean", ""),
                    "paper_table4_std": targets.get("table4_std", ""),
                }
            )
    return rows


def paired_comparison(first: list[dict], second: list[dict]) -> list[dict]:
    """Compare per-patient values of two models with a Wilcoxon signed-rank test.

    Returns `second - first` differences over the patients both models share.
    """
    first_rows = {row["patient"]: row for row in first}
    second_rows = {row["patient"]: row for row in second}
    patients = sorted(first_rows.keys() & second_rows.keys())

    rows = []
    for metric in COMPARED_METRICS:
        pairs = [
            (first_rows[patient][metric], second_rows[patient][metric])
            for patient in patients
            if not np.isnan(first_rows[patient][metric])
            and not np.isnan(second_rows[patient][metric])
        ]
        if not pairs:
            continue
        differences = np.array([right - left for left, right in pairs])
        changed = differences[differences != 0]
        rows.append(
            {
                "metric": metric,
                "patients": len(pairs),
                "first_mean": float(np.mean([left for left, _ in pairs])),
                "second_mean": float(np.mean([right for _, right in pairs])),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "patients_changed": int(changed.size),
                "wilcoxon_p": float(wilcoxon(changed).pvalue) if changed.size else 1.0,
            }
        )
    return rows
