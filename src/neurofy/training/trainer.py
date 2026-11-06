"""Model training."""

import logging
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from neurofy.config import BaselineConfig
from neurofy.data.dataset import get_dataloader
from neurofy.evaluation.evaluator import evaluate
from neurofy.evaluation.metrics import compute_metrics
from neurofy.evaluation.plots import plot_learning_curves
from neurofy.evaluation.reporting import log_evaluation
from neurofy.models.simple import create_model
from neurofy.training.checkpoint import load_checkpoint, save_checkpoint
from neurofy.training.early_stopping import EarlyStopping
from neurofy.training.loss import get_loss_fn
from neurofy.utils.runtime import get_device, set_seed
from neurofy.utils.tracking import experiment_run

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch and return metrics over all samples."""
    model.train()
    total_loss = 0.0
    labels, probabilities = [], []

    for features, targets in dataloader:
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


def train(config: BaselineConfig) -> None:
    """Train a model and log the full experiment to MLflow."""
    set_seed(config.seed)
    device = get_device(config.device)

    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")
    test_loader = get_dataloader(config, "test")

    model = create_model(config).to(device)
    criterion = get_loss_fn()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    stopper = EarlyStopping(config.early_stopping_patience, config.early_stopping_start_epoch)
    history: dict[str, list[float]] = defaultdict(list)

    with experiment_run(config, "train"), TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "best_model.pt"
        num_parameters = sum(parameter.numel() for parameter in model.parameters())

        mlflow.set_tag("device", str(device))
        mlflow.log_param("num_parameters", num_parameters)
        mlflow.log_text(str(model), "model_summary.txt")
        logger.info("Training on %s with %d parameters", device, num_parameters)

        for epoch in range(1, config.max_epochs + 1):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device).metrics

            epoch_metrics = {
                **{f"train/{name}": value for name, value in train_metrics.items()},
                **{f"val/{name}": value for name, value in val_metrics.items()},
            }
            mlflow.log_metrics(epoch_metrics, step=epoch)
            for name, value in epoch_metrics.items():
                history[name].append(value)

            logger.info(
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
                logger.info("Early stopping at epoch %d", epoch)
                break

        logger.info("Best epoch %d | val_acc=%.4f", stopper.best_epoch, stopper.best_score)
        mlflow.log_metric("best_epoch", stopper.best_epoch)
        mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        mlflow.log_dict(dict(history), "metrics/history.json")
        mlflow.log_figure(plot_learning_curves(history), "plots/learning_curves.png")

        load_checkpoint(checkpoint_path, model, device)
        log_evaluation(evaluate(model, val_loader, criterion, device), "best_val")

        test_result = evaluate(model, test_loader, criterion, device)
        log_evaluation(test_result, "test")
        logger.info(
            "Test | loss=%.4f acc=%.4f",
            test_result.metrics["loss"],
            test_result.metrics["accuracy"],
        )
