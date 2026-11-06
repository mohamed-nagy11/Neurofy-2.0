"""Firing power post-processing and event-based metrics."""

from dataclasses import dataclass

import numpy as np

from neurofy.evaluation.evaluator import EvaluationResult
from neurofy.evaluation.metrics import compute_metrics, to_predictions


def firing_power(probabilities: np.ndarray, window: int) -> np.ndarray:
    """Return the fraction of positive windows over each full run of `window` windows.

    The output has one value per window from the first full buffer onward, so it is
    empty when there are fewer than `window` windows.
    """
    if len(probabilities) < window:
        return np.empty(0)
    positives = to_predictions(probabilities).astype(float)
    return np.convolve(positives, np.ones(window), mode="valid") / window


def first_alarm(power: np.ndarray, threshold: float) -> int | None:
    """Return the first decision index at or above `threshold`, if any."""
    alarms = np.flatnonzero(power >= threshold)
    return int(alarms[0]) if alarms.size else None


@dataclass(frozen=True)
class FoldOutcome:
    """Firing-power results for one fold's held-out segments."""

    segment: EvaluationResult
    events: dict[str, float]
    preictal_power: np.ndarray
    interictal_power: np.ndarray


def evaluate_fold(
    preictal_probabilities: np.ndarray,
    interictal_probabilities: np.ndarray,
    seconds_to_onset: np.ndarray,
    window: int,
    threshold: float,
    window_seconds: float,
) -> FoldOutcome:
    """Compute segment metrics on firing-power decisions, and event-based sensitivity."""
    preictal_power = firing_power(preictal_probabilities, window)
    interictal_power = firing_power(interictal_probabilities, window)

    labels = np.concatenate([np.ones(len(preictal_power)), np.zeros(len(interictal_power))])
    scores = np.concatenate([preictal_power, interictal_power])
    segment = EvaluationResult(
        metrics=compute_metrics(labels, scores, threshold),
        labels=labels,
        probabilities=scores,
        threshold=threshold,
    )

    alarm = first_alarm(preictal_power, threshold)
    events = {
        "sensitivity": float(alarm is not None),
        "interictal_alarm": float(first_alarm(interictal_power, threshold) is not None),
    }
    if alarm is not None:
        last_window = alarm + window - 1
        events["alarm_lead_seconds"] = float(seconds_to_onset[last_window] - window_seconds)
    return FoldOutcome(segment, events, preictal_power, interictal_power)
