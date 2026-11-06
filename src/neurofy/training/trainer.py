"""Training of one leave-one-seizure-out fold."""

import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import torch
from torch import nn, optim

from neurofy.config import ExperimentConfig
from neurofy.data.dataset import WindowLoader
from neurofy.data.folds import Fold, sample_training_windows
from neurofy.evaluation.evaluator import evaluate
from neurofy.evaluation.firing_power import evaluate_fold
from neurofy.evaluation.metrics import compute_metrics
from neurofy.evaluation.plots import plot_learning_curves, plot_prediction_timeline
from neurofy.evaluation.reporting import log_evaluation
from neurofy.models.mobilenet import create_model
from neurofy.training.checkpoint import load_checkpoint, save_checkpoint
from neurofy.training.early_stopping import EarlyStopping
from neurofy.training.loss import get_loss_fn
from neurofy.utils.runtime import set_seed
from neurofy.utils.tracking import experiment_run

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch and return metrics over all samples."""
    model.train()
    total_loss = 0.0
    labels, probabilities = [], []

    for features, targets in batches:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        labels.append(targets.cpu())
        probabilities.append(torch.sigmoid(logits).detach().cpu())

    labels = torch.cat(labels).numpy()
    probabilities = torch.cat(probabilities).numpy()
    return {
        "loss": total_loss / len(labels),
        **compute_metrics(labels, probabilities),
    }


def _log_final_results(
    model: nn.Module,
    fold: Fold,
    held_out: WindowLoader,
    criterion: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    tmp_dir: Path,
) -> None:
    """Evaluate the restored best model and log every final metric and artifact."""
    window_result = evaluate(model, held_out, criterion, device)
    log_evaluation(window_result, "window")

    preictal_count = len(fold.test_preictal)
    preictal = window_result.probabilities[:preictal_count]
    interictal = window_result.probabilities[preictal_count:]
    window_seconds = config.window_samples / config.sampling_rate
    outcome = evaluate_fold(
        preictal,
        interictal,
        fold.test_seconds_to_onset,
        config.firing_power_window,
        config.firing_power_threshold,
        window_seconds,
    )
    log_evaluation(outcome.segment, "segment")
    mlflow.log_metrics({f"event/{name}": value for name, value in outcome.events.items()})
    mlflow.log_dict(outcome.events, "metrics/event.json")

    predictions_path = tmp_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        preictal_probabilities=preictal,
        interictal_probabilities=interictal,
        preictal_starts=fold.test_preictal,
        interictal_starts=fold.test_interictal,
        seconds_to_onset=fold.test_seconds_to_onset,
    )
    mlflow.log_artifact(str(predictions_path))
    mlflow.log_figure(
        plot_prediction_timeline(
            fold.test_seconds_to_onset,
            preictal,
            outcome.preictal_power,
            interictal,
            outcome.interictal_power,
            config.firing_power_window,
            config.firing_power_threshold,
        ),
        "plots/prediction_timeline.png",
    )
    logger.info(
        "%s | segment_acc=%.4f segment_fpr=%.4f sensitivity=%.0f interictal_alarm=%.0f",
        fold.key,
        outcome.segment.metrics["accuracy"],
        outcome.segment.metrics["fpr"],
        outcome.events["sensitivity"],
        outcome.events["interictal_alarm"],
    )


def train_fold(
    config: ExperimentConfig,
    fold: Fold,
    signal: torch.Tensor,
    parent_run_id: str | None = None,
) -> None:
    """Train one fold in its own MLflow run and log the full result."""
    set_seed(fold.seed)
    rng = np.random.default_rng(fold.seed)
    device = signal.device

    held_out_starts = np.concatenate([fold.test_preictal, fold.test_interictal])
    held_out_labels = np.concatenate(
        [np.ones(len(fold.test_preictal)), np.zeros(len(fold.test_interictal))]
    )
    held_out = WindowLoader(
        signal,
        held_out_starts,
        held_out_labels,
        config.window_samples,
        config.batch_size,
    )

    model = create_model(config).to(device)
    criterion = get_loss_fn()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    stopper = EarlyStopping(config.early_stopping_patience, config.early_stopping_start_epoch)
    history: dict[str, list[float]] = defaultdict(list)

    with (
        experiment_run(
            config,
            "fold",
            run_name=f"{config.name}/{fold.key}",
            parent_run_id=parent_run_id,
            tags={
                "patient": fold.patient_id,
                "seizure": str(fold.seizure_index),
                "fold_key": f"{config.name}/{fold.key}",
                "device": str(device),
            },
        ),
        TemporaryDirectory() as tmp_name,
    ):
        tmp_dir = Path(tmp_name)
        checkpoint_path = tmp_dir / "best_model.pt"
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        mlflow.log_param("num_parameters", num_parameters)
        mlflow.log_param("fold_seed", fold.seed)
        mlflow.log_text(str(model), "model_summary.txt")

        for epoch in range(1, config.max_epochs + 1):
            starts, labels = sample_training_windows(fold, rng)
            batches = WindowLoader(
                signal,
                starts,
                labels,
                config.window_samples,
                config.batch_size,
            )
            train_metrics = train_epoch(model, batches, criterion, optimizer, device)
            val_metrics = evaluate(model, held_out, criterion, device).metrics

            epoch_metrics = {
                **{f"train/{name}": value for name, value in train_metrics.items()},
                **{f"val/{name}": value for name, value in val_metrics.items()},
            }
            mlflow.log_metrics(epoch_metrics, step=epoch, synchronous=False)
            for name, value in epoch_metrics.items():
                history[name].append(value)

            logger.debug(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch,
                config.max_epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

            if stopper.update(val_metrics["accuracy"], epoch):
                save_checkpoint(checkpoint_path, model, config, epoch, stopper.best_score)
            if stopper.should_stop:
                logger.debug("Early stopping at epoch %d", epoch)
                break

        mlflow.log_metrics(
            {"best_epoch": stopper.best_epoch, "epochs_run": epoch, "best_val": stopper.best_score}
        )
        mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        mlflow.log_dict(dict(history), "metrics/history.json")
        mlflow.log_figure(plot_learning_curves(history), "plots/learning_curves.png")

        load_checkpoint(checkpoint_path, model, device)
        _log_final_results(model, fold, held_out, criterion, config, device, tmp_dir)
