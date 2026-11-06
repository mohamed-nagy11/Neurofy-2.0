from pathlib import Path

import mlflow
import pytest
import torch

from neurofy.config import BaselineConfig
from neurofy.models.simple import create_model
from neurofy.training.checkpoint import load_checkpoint, save_checkpoint
from neurofy.training.trainer import train


def test_checkpoint_round_trip(config: BaselineConfig, tmp_path: Path) -> None:
    source = create_model(config)
    target = create_model(config)
    path = tmp_path / "model.pt"

    save_checkpoint(path, source, config, epoch=7, score=0.9)
    checkpoint = load_checkpoint(path, target, torch.device("cpu"))

    assert checkpoint["epoch"] == 7
    assert torch.equal(source.fc.weight, target.fc.weight)


@pytest.mark.usefixtures("tracking_store")
def test_train_logs_full_experiment_to_mlflow(config: BaselineConfig) -> None:
    train(config)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    run = client.search_runs([experiment.experiment_id])[0]
    artifacts = {
        artifact.path
        for directory in (None, "checkpoints", "metrics", "plots")
        for artifact in client.list_artifacts(run.info.run_id, directory)
    }

    assert run.info.status == "FINISHED"
    assert run.data.params["weight_decay"] == str(config.weight_decay)
    assert {"train/loss", "val/accuracy", "best_epoch", "test/fpr"} <= run.data.metrics.keys()
    assert {
        "config.json",
        "model_summary.txt",
        "run.log",
        "checkpoints/best_model.pt",
        "metrics/history.json",
        "metrics/test.json",
        "plots/learning_curves.png",
        "plots/test_confusion_matrix.png",
        "plots/test_roc_curve.png",
    } <= artifacts
