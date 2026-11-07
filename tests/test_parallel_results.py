import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest

from conftest import write_synthetic_dataset
from neurofy.config import ExperimentConfig
from neurofy.evaluation.reporting import log_experiment_report
from neurofy.training.runner import run_folds
from neurofy.utils.tracking import (
    RESULTS_ARCHIVE,
    experiment_run,
    restore_results,
    tracking_session,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _archive_with_marker(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True)
    source = path.parent / marker
    source.write_text(marker)
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source, arcname=marker)


def test_restore_selects_results_by_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _archive_with_marker(input_dir / "notebooks" / "train-a" / RESULTS_ARCHIVE, "a.txt")
    _archive_with_marker(input_dir / "notebooks" / "train-b" / RESULTS_ARCHIVE, "b.txt")
    target = tmp_path / "store"
    target.mkdir()

    assert restore_results(target, input_dir, "train-b")
    assert (target / "b.txt").exists()
    assert not (target / "a.txt").exists()
    assert not restore_results(target, input_dir, "train-c")


def test_session_fails_when_required_results_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train-c"):
        with tracking_session(
            tmp_path / "store",
            tmp_path / "out.tar.gz",
            restore_from="train-c",
            search_root=tmp_path,
        ):
            pass


def _train_model_store(config: ExperimentConfig, data_dir: Path, tmp_path: Path, kernel: str):
    """Train one model in its own store, as its own Kaggle kernel would."""
    archive = tmp_path / "input" / kernel / RESULTS_ARCHIVE
    archive.parent.mkdir(parents=True)
    with tracking_session(tmp_path / kernel, archive):
        with experiment_run(config, "experiment", run_name=config.name) as run:
            run_folds(config, data_dir, run.info.run_id)
            log_experiment_report(config.experiment_name, config.name)
    mlflow.set_tracking_uri(None)


def test_comparison_reads_two_separate_stores(config: ExperimentConfig, tmp_path: Path) -> None:
    previous_uri = mlflow.get_tracking_uri()
    data_dir = tmp_path / "data"
    write_synthetic_dataset(data_dir, seizures=2)
    tiny = replace(config, max_epochs=1, experiment_name="parallel-test")
    dual = replace(tiny, name="fp1f7_t8p8", channels=("FP1-F7", "T8-P8"))

    try:
        _train_model_store(tiny, data_dir, tmp_path, "train-first")
        _train_model_store(dual, data_dir, tmp_path, "train-second")
    finally:
        mlflow.set_tracking_uri(previous_uri)

    output = tmp_path / "comparison.tar.gz"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compare_models.py"),
            "--experiment",
            "parallel-test",
            "--tracking-dir",
            str(tmp_path / "comparison"),
            "--archive",
            str(output),
            "--first-results",
            "train-first",
            "--second-results",
            "train-second",
            "--input-dir",
            str(tmp_path / "input"),
        ],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    with tarfile.open(output) as archive:
        names = archive.getnames()
    for artifact in (
        "paper_comparison.csv",
        "model_comparison.csv",
        "patient_metrics_fp1f7.csv",
        "patient_metrics_fp1f7_t8p8.csv",
    ):
        assert any(name.endswith(artifact) for name in names)
