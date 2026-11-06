"""MLflow run management and a local tracking server."""

import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import torch

from neurofy.config import ExperimentConfig
from neurofy.utils.logger import log_to_file
from neurofy.utils.paths import KAGGLE_INPUT

logger = logging.getLogger(__name__)

RESULTS_ARCHIVE = "mlflow_results.tar.gz"

_SERVER_TIMEOUT_SECONDS = 180


@contextmanager
def experiment_run(
    config: ExperimentConfig,
    run_type: str,
    *,
    run_name: str | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run that records the config, environment and run log."""
    params = asdict(config)
    all_tags = {
        "run_type": run_type,
        "config": config.name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": os.environ.get("NEUROFY_GIT_COMMIT", "unknown"),
        **(tags or {}),
    }

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(
        run_id=run_id,
        run_name=run_name or f"{config.name}-{run_type}",
        parent_run_id=parent_run_id,
        nested=parent_run_id is not None,
        tags=None if run_id else all_tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_dict(params, "config.json")

        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            try:
                with log_to_file(log_path):
                    logger.info("Started MLflow run %s (%s)", run.info.run_id, run.info.run_name)
                    yield run
            except Exception:
                logger.exception("Run %s failed", run.info.run_id)
                raise
            finally:
                mlflow.flush_async_logging()
                mlflow.log_artifact(str(log_path))


def finished_fold_runs(experiment_name: str, config_name: str) -> list:
    """Return an experiment's finished fold runs for one config."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    return mlflow.MlflowClient().search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.run_type = 'fold' AND tags.config = '{config_name}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=5000,
    )


def find_experiment_run_id(experiment_name: str, config_name: str) -> str | None:
    """Return the parent run of a config from an earlier session, if any."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = mlflow.MlflowClient().search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.run_type = 'experiment' AND tags.config = '{config_name}'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_healthy(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + _SERVER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MLflow server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise TimeoutError(f"MLflow server not healthy after {_SERVER_TIMEOUT_SECONDS} s")


def _stop_server(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=60)


def archive_results(tracking_dir: Path, archive_path: Path) -> None:
    """Pack the tracking store into one archive."""
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tracking_dir, arcname=".")
    logger.info("Archived results to %s (%.1f MB)", archive_path, archive_path.stat().st_size / 1e6)


def restore_results(tracking_dir: Path, search_root: Path = KAGGLE_INPUT) -> bool:
    """Unpack the results of an earlier session attached under `search_root`, if any."""
    archives = sorted(search_root.rglob(RESULTS_ARCHIVE)) if search_root.exists() else []
    if not archives:
        return False
    with tarfile.open(archives[0], "r:gz") as archive:
        archive.extractall(tracking_dir, filter="data")
    logger.info("Restored results from %s", archives[0])
    return True


@contextmanager
def tracking_session(tracking_dir: Path, archive_path: Path) -> Iterator[str]:
    """Serve a local tracking store, archiving it on exit.

    A single server process serializes writes from parallel workers, and proxied
    artifacts keep the store portable when the archive moves to another machine.
    """
    tracking_dir.mkdir(parents=True, exist_ok=True)
    database = tracking_dir / "mlflow.db"
    if not database.exists():
        restore_results(tracking_dir)

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{database.as_posix()}",
        "--artifacts-destination",
        str(tracking_dir / "artifacts"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    with (tracking_dir / "server.log").open("a", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
    previous_uri = os.environ.get("MLFLOW_TRACKING_URI")
    try:
        _wait_until_healthy(url, process)
        os.environ["MLFLOW_TRACKING_URI"] = url
        os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
        mlflow.set_tracking_uri(url)
        logger.info("MLflow server running at %s", url)
        yield url
    finally:
        _stop_server(process)
        if previous_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_uri
        archive_results(tracking_dir, archive_path)
