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


def plot_window_heatmap(
    spectrogram: np.ndarray,
    cam: np.ndarray,
    frequencies: np.ndarray,
    title: str,
) -> Figure:
    """Plot a window's spectrogram next to its Grad-CAM map."""
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    extent = (0, 10, frequencies[0], frequencies[-1])

    for ax, values, name, colormap in (
        (axes[0], spectrogram, "Log magnitude", "viridis"),
        (axes[1], cam, "Grad-CAM", "magma"),
    ):
        image = ax.imshow(values.T, origin="lower", aspect="auto", extent=extent, cmap=colormap)
        ax.set(title=name, xlabel="Seconds")
        figure.colorbar(image, ax=ax)
    axes[0].set_ylabel("Frequency (Hz)")
    figure.suptitle(title)
    return figure


def plot_band_importance(rows: list[dict]) -> Figure:
    """Plot mean Grad-CAM importance per band for preictal and interictal windows."""
    bands = [key for key in rows[0] if key not in ("patient", "seizure", "class")]
    figure = Figure(figsize=(7, 4), layout="constrained")
    ax = figure.subplots()
    positions = np.arange(len(bands))

    for index, label in enumerate(CLASS_NAMES):
        values = [np.mean([row[band] for row in rows if row["class"] == label]) for band in bands]
        ax.bar(positions + index * 0.4, values, 0.4, label=label)
    ax.set_xticks(positions + 0.2, bands)
    ax.set(ylabel="Mean Grad-CAM")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return figure


def plot_band_importance_by_time(rows: list[dict]) -> Figure:
    """Plot mean preictal Grad-CAM importance per band against time before onset."""
    reserved = ("patient", "seizure", "minutes_before_onset")
    bands = [key for key in rows[0] if key not in reserved]
    times = sorted({row["minutes_before_onset"] for row in rows}, reverse=True)
    figure = Figure(figsize=(8, 4), layout="constrained")
    ax = figure.subplots()

    for band in bands:
        values = [
            np.mean([row[band] for row in rows if row["minutes_before_onset"] == time])
            for time in times
        ]
        ax.plot([-time for time in times], values, marker="o", label=band)
    ax.set(xlabel="Minutes relative to seizure onset", ylabel="Mean Grad-CAM")
    ax.grid(alpha=0.3)
    ax.legend()
    return figure


def plot_deletion_test(rows: list[dict]) -> Figure:
    """Plot the logit drop from masking Grad-CAM regions against random regions."""
    fractions = sorted({row["fraction"] for row in rows})
    figure = Figure(figsize=(6, 4), layout="constrained")
    ax = figure.subplots()

    for key, label in (("cam_logit_drop", "Grad-CAM regions"), ("random_logit_drop", "random")):
        values = [np.mean([row[key] for row in rows if row["fraction"] == f]) for f in fractions]
        ax.plot(fractions, values, marker="o", label=label)
    ax.set(xlabel="Fraction of cells masked", ylabel="Mean logit drop")
    ax.grid(alpha=0.3)
    ax.legend()
    return figure
