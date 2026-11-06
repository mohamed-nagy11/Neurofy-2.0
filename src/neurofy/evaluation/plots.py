"""Experiment plots."""

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from neurofy.evaluation.metrics import DECISION_THRESHOLD, to_predictions

CLASS_NAMES = ("interictal", "preictal")


def plot_learning_curves(history: dict[str, list[float]]) -> Figure:
    """Plot train and validation loss and accuracy per epoch."""
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2)
    epochs = range(1, len(history["train/loss"]) + 1)

    for ax, metric in zip(axes, ("loss", "accuracy"), strict=True):
        ax.plot(epochs, history[f"train/{metric}"], label="train")
        ax.plot(epochs, history[f"val/{metric}"], label="val")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.grid(alpha=0.3)
        ax.legend()
    return figure


def plot_confusion_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> Figure:
    """Plot the confusion matrix at the decision threshold."""
    figure = Figure(figsize=(5, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        labels.astype(int),
        to_predictions(probabilities, threshold),
        labels=[0, 1],
        display_labels=CLASS_NAMES,
        ax=figure.subplots(),
    )
    return figure


def plot_roc_curve(labels: np.ndarray, probabilities: np.ndarray) -> Figure:
    """Plot the ROC curve."""
    figure = Figure(figsize=(5, 4), layout="constrained")
    RocCurveDisplay.from_predictions(labels.astype(int), probabilities, ax=figure.subplots())
    return figure


def _plot_segment(
    ax,
    minutes: np.ndarray,
    probabilities: np.ndarray,
    power: np.ndarray,
    window: int,
    threshold: float,
) -> None:
    ax.fill_between(
        minutes,
        0,
        1,
        where=probabilities >= DECISION_THRESHOLD,
        step="mid",
        color="tab:blue",
        alpha=0.2,
        label="window predicted preictal",
    )
    ax.plot(minutes, probabilities, color="tab:blue", linewidth=0.5, label="probability")
    ax.plot(minutes[window - 1 :], power, color="tab:orange", label="firing power")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label="threshold")
    alarms = np.flatnonzero(power >= threshold)
    if alarms.size:
        ax.axvline(minutes[alarms[0] + window - 1], color="red", label="first alarm")
    ax.set(ylim=(0, 1.02), ylabel="Score")
    ax.grid(alpha=0.3)


def plot_prediction_timeline(
    seconds_to_onset: np.ndarray,
    preictal_probabilities: np.ndarray,
    preictal_power: np.ndarray,
    interictal_probabilities: np.ndarray,
    interictal_power: np.ndarray,
    window: int,
    threshold: float,
) -> Figure:
    """Plot held-out predictions and firing power over time, in the style of the paper's Fig. 2."""
    figure = Figure(figsize=(10, 6), layout="constrained")
    preictal_ax, interictal_ax = figure.subplots(2, 1)

    _plot_segment(
        preictal_ax,
        -seconds_to_onset / 60,
        preictal_probabilities,
        preictal_power,
        window,
        threshold,
    )
    preictal_ax.set(title="Held-out preictal", xlabel="Minutes relative to seizure onset")
    preictal_ax.legend(loc="lower right", fontsize=8)

    _plot_segment(
        interictal_ax,
        np.arange(len(interictal_probabilities)) * 5 / 60,
        interictal_probabilities,
        interictal_power,
        window,
        threshold,
    )
    interictal_ax.set(title="Held-out interictal", xlabel="Minutes from segment start")
    return figure


def plot_patient_metrics(rows: list[dict], metrics: tuple[str, ...]) -> Figure:
    """Plot each metric per patient as grouped bars."""
    figure = Figure(figsize=(max(8, len(rows) * 0.7), 4), layout="constrained")
    ax = figure.subplots()
    positions = np.arange(len(rows))
    width = 0.8 / len(metrics)

    for index, metric in enumerate(metrics):
        values = [row[metric] for row in rows]
        ax.bar(positions + index * width, values, width, label=metric)
    ax.set_xticks(positions + 0.4 - width / 2, [row["patient"] for row in rows], rotation=45)
    ax.set(ylim=(0, 1.02), ylabel="Value")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return figure
