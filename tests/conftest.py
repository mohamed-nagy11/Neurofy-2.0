"""Shared test fixtures."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pytest

from neurofy.config import ExperimentConfig

SAMPLING_RATE = 256
WINDOW = 256
HOP = 128
HELD_OUT = 2560


@pytest.fixture
def config() -> ExperimentConfig:
    """Return a configuration small enough for fast CPU runs."""
    return replace(
        ExperimentConfig(),
        batch_size=8,
        max_epochs=3,
        early_stopping_patience=2,
        early_stopping_start_epoch=1,
        firing_power_window=3,
        window_samples=WINDOW,
        hop_samples=HOP,
        held_out_interictal_samples=HELD_OUT,
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


def write_synthetic_dataset(
    data_dir: Path,
    patients: tuple[str, ...] = ("chb01",),
    seizures: int = 2,
) -> None:
    """Write a small preprocessed dataset where preictal data has a distinct frequency."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "signals").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    preictal_length = 2560
    time = np.arange(preictal_length) / SAMPLING_RATE

    entries = {}
    for patient in patients:
        parts, seizure_entries, position = [], [], 0
        for _ in range(seizures):
            part = np.sin(2 * np.pi * 30 * time) * 20 + rng.normal(0, 1, preictal_length)
            parts.append(part)
            seizure_entries.append(
                {
                    "file": "x.edf",
                    "onset": 0,
                    "preictal": [
                        {
                            "start": position,
                            "end": position + preictal_length,
                            "seconds_to_onset": 130,
                        }
                    ],
                }
            )
            position += preictal_length
        interictal_length = HELD_OUT * (seizures + 3)
        interictal_time = np.arange(interictal_length) / SAMPLING_RATE
        parts.append(np.sin(2 * np.pi * 5 * interictal_time) * 20)
        signal = np.concatenate(parts, dtype=np.float32)
        stacked = np.stack([signal, signal * 0.5])
        np.save(data_dir / "signals" / f"{patient}.npy", stacked)
        entries[patient] = {
            "signal": f"signals/{patient}.npy",
            "samples": stacked.shape[1],
            "files": {},
            "seizures": seizure_entries,
            "interictal": [{"start": position, "end": position + interictal_length}],
        }

    manifest = {
        "channels": ["FP1-F7", "T8-P8"],
        "sampling_rate": SAMPLING_RATE,
        "patients": entries,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
