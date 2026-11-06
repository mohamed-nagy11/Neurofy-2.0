"""Experiment plots."""

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from neurofy.evaluation.metrics import to_predictions

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


def plot_confusion_matrix(labels: np.ndarray, probabilities: np.ndarray) -> Figure:
    """Plot the confusion matrix at the decision threshold."""
    figure = Figure(figsize=(5, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        labels.astype(int),
        to_predictions(probabilities),
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
