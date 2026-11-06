import math

import pytest

from neurofy.evaluation.aggregation import (
    FoldResult,
    rows_to_csv,
    summarize_cohort,
    summarize_patients,
)


def _fold(patient: str, accuracy: float, sensitivity: float) -> FoldResult:
    return FoldResult(
        patient,
        "0",
        {"segment/accuracy": accuracy, "event/sensitivity": sensitivity},
    )


def test_patient_rows_average_their_folds() -> None:
    rows = summarize_patients(
        [_fold("chb02", 0.5, 1.0), _fold("chb01", 1.0, 1.0), _fold("chb01", 0.5, 0.0)]
    )

    assert [row["patient"] for row in rows] == ["chb01", "chb02"]
    assert rows[0]["folds"] == 2
    assert rows[0]["segment/accuracy"] == pytest.approx(0.75)
    assert rows[0]["event/sensitivity"] == pytest.approx(0.5)
    assert math.isnan(rows[0]["window/accuracy"])


def test_cohort_weighs_patients_equally() -> None:
    rows = summarize_patients(
        [_fold("chb01", 1.0, 1.0), _fold("chb01", 1.0, 1.0), _fold("chb02", 0.5, 1.0)]
    )

    cohort = summarize_cohort(rows)

    assert cohort["segment/accuracy"]["mean"] == pytest.approx(0.75)
    assert cohort["segment/accuracy"]["std"] == pytest.approx(0.3535, abs=1e-3)


def test_rows_serialize_to_csv() -> None:
    text = rows_to_csv([{"patient": "chb01", "folds": 2}])

    assert text == "patient,folds\nchb01,2\n"
