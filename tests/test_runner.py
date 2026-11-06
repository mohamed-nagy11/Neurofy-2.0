import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import mlflow

from conftest import write_synthetic_dataset
from neurofy.config import ExperimentConfig
from neurofy.evaluation.reporting import log_experiment_report
from neurofy.training.runner import run_folds
from neurofy.utils.tracking import (
    RESULTS_ARCHIVE,
    experiment_run,
    find_experiment_run_id,
    restore_results,
    tracking_session,
)


def _resume_session(
    config: ExperimentConfig,
    data_dir: Path,
    tracking_dir: Path,
    archive: Path,
    result_path: Path,
) -> None:
    """Resume in a new process, as a later Kaggle session would."""
    with tracking_session(tracking_dir, archive):
        run_id = find_experiment_run_id(config.experiment_name, config.name)
        with experiment_run(config, "experiment", run_id=run_id) as run:
            resumed = run_folds(config, data_dir, run.info.run_id)

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(config.experiment_name)
        children = client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.run_type = 'fold'",
        )
        artifacts = [
            item.path for item in client.list_artifacts(children[0].info.run_id, "checkpoints")
        ]
    result_path.write_text(
        json.dumps(
            {
                "parent_id": run_id,
                "skipped": resumed.skipped,
                "finished": resumed.finished,
                "child_parents": [c.data.tags["mlflow.parentRunId"] for c in children],
                "artifact_uri": children[0].info.artifact_uri,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def test_parallel_session_resumes_and_restores_from_archive(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    previous_uri = mlflow.get_tracking_uri()
    data_dir = tmp_path / "data"
    write_synthetic_dataset(data_dir, seizures=2)
    tiny = replace(config, max_epochs=1, experiment_name="runner-test")
    archive = tmp_path / RESULTS_ARCHIVE

    try:
        with tracking_session(tmp_path / "first", archive):
            with experiment_run(tiny, "experiment", run_name=tiny.name) as run:
                summary = run_folds(tiny, data_dir, run.info.run_id, workers_per_gpu=2)
                cohort = log_experiment_report(tiny.experiment_name, tiny.name)
    finally:
        mlflow.set_tracking_uri(previous_uri)

    assert (summary.total_folds, summary.finished, summary.skipped) == (2, 2, 0)
    assert cohort
    assert archive.exists()

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    archive.replace(input_dir / RESULTS_ARCHIVE)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    assert restore_results(second_dir, input_dir)

    result_path = tmp_path / "result.json"
    process = mp.get_context("spawn").Process(
        target=_resume_session,
        args=(tiny, data_dir, second_dir, tmp_path / "again.tar.gz", result_path),
    )
    process.start()
    process.join(timeout=180)
    assert process.exitcode == 0

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["parent_id"] == run.info.run_id
    assert (result["skipped"], result["finished"]) == (2, 2)
    assert result["child_parents"] == [run.info.run_id] * 2
    assert result["artifact_uri"].startswith("mlflow-artifacts:")
    assert result["artifacts"] == ["checkpoints/best_model.pt"]
