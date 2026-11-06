"""MLflow run management."""

import logging
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import torch

from neurofy.config import BaselineConfig
from neurofy.utils.logger import log_to_file

logger = logging.getLogger(__name__)


@contextmanager
def experiment_run(config: BaselineConfig, run_type: str) -> Iterator[None]:
    """Open an MLflow run that records the config, environment and run log."""
    params = asdict(config)
    tags = {
        "run_type": run_type,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=f"{config.name}-{run_type}", tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_dict(params, "config.json")

        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            try:
                with log_to_file(log_path):
                    logger.info("Started MLflow run %s (%s)", run.info.run_id, run.info.run_name)
                    yield
            except Exception:
                logger.exception("Run %s failed", run.info.run_id)
                raise
            finally:
                mlflow.log_artifact(str(log_path))
