import logging
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
import torch

from conftest import write_synthetic_dataset
from neurofy.config import ExperimentConfig
from neurofy.data.dataset import load_records, load_signal
from neurofy.data.folds import build_folds
from neurofy.evaluation.reporting import log_experiment_report
from neurofy.models.mobilenet import create_model
from neurofy.training.checkpoint import load_checkpoint, save_checkpoint
from neurofy.training.trainer import train_fold
from neurofy.utils.tracking import experiment_run, find_experiment_run_id


def _artifacts(client: mlflow.MlflowClient, run_id: str) -> set[str]:
    return {
        artifact.path
        for directory in (None, "checkpoints", "metrics", "plots")
        for artifact in client.list_artifacts(run_id, directory)
    }


def test_checkpoint_round_trip(config: ExperimentConfig, tmp_path: Path) -> None:
    source = create_model(config)
    target = create_model(config)
    path = tmp_path / "model.pt"

    save_checkpoint(path, source, config, epoch=7, score=0.9)
    checkpoint = load_checkpoint(path, target, torch.device("cpu"))

    assert checkpoint["epoch"] == 7
    assert checkpoint["config"]["channels"] == config.channels
    assert torch.equal(source.classifier.weight, target.classifier.weight)


@pytest.mark.usefixtures("tracking_store")
def test_train_fold_logs_full_experiment(config: ExperimentConfig, tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path / "data")
    record = load_records(tmp_path / "data")["chb01"]
    fold = build_folds(record, config)[0]
    signal = load_signal(record, config.channels, torch.device("cpu"))

    with experiment_run(config, "experiment", run_name=config.name) as parent:
        train_fold(config, fold, signal, parent.info.run_id)
        cohort = log_experiment_report(config.experiment_name, config.name)
    mlflow.flush_async_logging()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    runs = {
        run.data.tags["run_type"]: run for run in client.search_runs([experiment.experiment_id])
    }
    child, parent_run = runs["fold"], runs["experiment"]

    assert child.info.status == "FINISHED"
    assert child.data.tags["mlflow.parentRunId"] == parent_run.info.run_id
    assert child.data.tags["fold_key"] == f"{config.name}/chb01/seizure-0"
    assert child.data.tags["patient"] == "chb01"
    assert {
        "train/loss",
        "val/accuracy",
        "best_epoch",
        "window/accuracy",
        "segment/accuracy",
        "segment/fpr",
        "event/sensitivity",
        "event/interictal_alarm",
    } <= child.data.metrics.keys()
    assert {
        "config.json",
        "model_summary.txt",
        "predictions.npz",
        "run.log",
        "checkpoints/best_model.pt",
        "metrics/history.json",
        "metrics/segment.json",
        "metrics/event.json",
        "plots/learning_curves.png",
        "plots/segment_confusion_matrix.png",
        "plots/segment_roc_curve.png",
        "plots/window_confusion_matrix.png",
        "plots/prediction_timeline.png",
    } <= _artifacts(client, child.info.run_id)
    assert "segment/accuracy" in cohort
    assert {"patient_metrics.csv", "cohort_metrics.json"} <= _artifacts(
        client, parent_run.info.run_id
    )
    assert find_experiment_run_id(config.experiment_name, config.name) == parent_run.info.run_id


@pytest.mark.usefixtures("tracking_store")
def test_train_fold_logs_one_line_per_epoch(
    config: ExperimentConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_synthetic_dataset(tmp_path / "data")
    two_epochs = replace(config, max_epochs=2, early_stopping_start_epoch=10)
    record = load_records(tmp_path / "data")["chb01"]
    signal = load_signal(record, two_epochs.channels, torch.device("cpu"))

    with caplog.at_level(logging.INFO, logger="neurofy.training.trainer"):
        train_fold(two_epochs, build_folds(record, two_epochs)[0], signal)

    lines = [record.getMessage() for record in caplog.records if "| Epoch" in record.getMessage()]
    assert len(lines) == 2
    assert lines[0].startswith("chb01/seizure-0 | Epoch 1/2 | train_loss=")
    assert "train_acc=" in lines[0]
    assert "val_loss=" in lines[0]
    assert "val_acc=" in lines[0]


@pytest.mark.usefixtures("tracking_store")
def test_train_fold_supports_two_channels(config: ExperimentConfig, tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path / "data")
    dual = replace(config, name="fp1f7_t8p8", channels=("FP1-F7", "T8-P8"), max_epochs=1)
    record = load_records(tmp_path / "data")["chb01"]
    signal = load_signal(record, dual.channels, torch.device("cpu"))

    train_fold(dual, build_folds(record, dual)[1], signal)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(dual.experiment_name)
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.data.params["channels"] == "('FP1-F7', 'T8-P8')"
    assert run.info.status == "FINISHED"


@pytest.mark.usefixtures("tracking_store")
def test_training_learns_the_synthetic_signal(config: ExperimentConfig, tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path / "data", seizures=3)
    learned = replace(config, max_epochs=25, early_stopping_start_epoch=25)
    record = load_records(tmp_path / "data")["chb01"]
    signal = load_signal(record, learned.channels, torch.device("cpu"))

    train_fold(learned, build_folds(record, learned)[0], signal)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(learned.experiment_name)
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.data.metrics["window/accuracy"] > 0.9
