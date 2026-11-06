import numpy as np
import pytest

from neurofy.evaluation.firing_power import evaluate_fold, firing_power, first_alarm


def test_firing_power_uses_only_full_buffers() -> None:
    probabilities = np.array([0.9, 0.1, 0.9, 0.9])

    power = firing_power(probabilities, window=3)

    assert power == pytest.approx([2 / 3, 2 / 3])
    assert firing_power(probabilities, window=5).size == 0


def test_threshold_is_exact_at_60_of_119() -> None:
    at_threshold = np.r_[np.ones(60), np.zeros(59)]
    below = np.r_[np.ones(59), np.zeros(60)]

    assert first_alarm(firing_power(at_threshold, 119), 0.5) == 0
    assert first_alarm(firing_power(below, 119), 0.5) is None


def test_complete_preictal_segment_gives_241_decisions() -> None:
    assert firing_power(np.ones(359), 119).size == 241


def test_all_correct_fold_scores_perfectly() -> None:
    seconds = 1920 - 5 * np.arange(359.0)

    outcome = evaluate_fold(np.ones(359), np.zeros(359), seconds, 119, 0.5, 10.0)

    assert outcome.segment.metrics["accuracy"] == 1.0
    assert outcome.segment.metrics["fpr"] == 0.0
    assert outcome.events["sensitivity"] == 1.0
    assert outcome.events["interictal_alarm"] == 0.0
    # The first alarm can fire once 119 windows exist: 10 min into the -32 min segment.
    assert outcome.events["alarm_lead_seconds"] == 1920 - 5 * 118 - 10


def test_missed_seizure_and_false_alarm() -> None:
    seconds = 1920 - 5 * np.arange(359.0)

    outcome = evaluate_fold(np.zeros(359), np.ones(359), seconds, 119, 0.5, 10.0)

    assert outcome.segment.metrics["accuracy"] == 0.0
    assert outcome.segment.metrics["fpr"] == 1.0
    assert outcome.events["sensitivity"] == 0.0
    assert outcome.events["interictal_alarm"] == 1.0
    assert "alarm_lead_seconds" not in outcome.events
