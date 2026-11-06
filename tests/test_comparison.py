import math

import pytest

from neurofy.evaluation.comparison import compare_with_paper, paired_comparison


def _row(patient: str, accuracy: float) -> dict:
    return {
        "patient": patient,
        "segment/accuracy": accuracy,
        "segment/fpr": 0.1,
        "event/sensitivity": 1.0,
        "event/interictal_alarm": float("nan"),
        "window/accuracy": accuracy,
    }


def test_paper_comparison_includes_available_metrics() -> None:
    rows = compare_with_paper({"segment/accuracy": {"mean": 0.8, "std": 0.1}})

    assert len(rows) == 1
    assert rows[0]["paper_patient_tables"] == 0.9072
    assert rows[0]["paper_table4_mean"] == 0.902


def test_paired_comparison_uses_shared_patients_only() -> None:
    first = [_row("chb01", 0.5), _row("chb02", 0.6), _row("chb03", 0.7), _row("chb09", 0.1)]
    second = [_row("chb01", 0.6), _row("chb02", 0.8), _row("chb03", 0.8)]

    rows = {row["metric"]: row for row in paired_comparison(first, second)}
    accuracy = rows["segment/accuracy"]

    assert accuracy["patients"] == 3
    assert accuracy["mean_difference"] == pytest.approx(0.1333, abs=1e-3)
    assert accuracy["patients_changed"] == 3
    assert 0 <= accuracy["wilcoxon_p"] <= 1


def test_paired_comparison_handles_ties_and_missing_values() -> None:
    first = [_row("chb01", 0.5), _row("chb02", 0.6)]
    second = [_row("chb01", 0.5), _row("chb02", 0.6)]

    rows = {row["metric"]: row for row in paired_comparison(first, second)}

    assert rows["event/sensitivity"]["wilcoxon_p"] == 1.0
    assert rows["event/sensitivity"]["patients_changed"] == 0
    assert "event/interictal_alarm" not in rows
    assert not math.isnan(rows["segment/fpr"]["mean_difference"])
