"""Shared test fixtures."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest

from neurofy.config import BaselineConfig


@pytest.fixture
def config() -> BaselineConfig:
    """Return a configuration small enough for fast CPU runs."""
    return replace(
        BaselineConfig(),
        num_samples=32,
        batch_size=8,
        max_epochs=3,
        early_stopping_patience=2,
        early_stopping_start_epoch=1,
        device="cpu",
    )


@pytest.fixture
def tracking_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate MLflow runs and artifacts in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield
    mlflow.set_tracking_uri(previous_uri)
