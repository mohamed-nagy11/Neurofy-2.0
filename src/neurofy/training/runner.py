"""Parallel execution of folds with GPU-pinned worker processes."""

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from neurofy.config import ExperimentConfig
from neurofy.data.dataset import load_records, load_signal
from neurofy.data.folds import Fold, build_folds
from neurofy.training.trainer import train_fold
from neurofy.utils.logger import configure_logging
from neurofy.utils.runtime import get_device
from neurofy.utils.tracking import finished_fold_runs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    """Outcome of one session of an experiment."""

    total_folds: int
    skipped: int
    finished: int
    seconds: float

    @property
    def remaining(self) -> int:
        return self.total_folds - self.finished


def _worker(
    config: ExperimentConfig,
    data_dir: Path,
    parent_run_id: str,
    jobs: mp.Queue,
    deadline: float,
) -> None:
    """Take folds from the queue until it is empty or the deadline passes."""
    configure_logging()
    torch.set_num_threads(1)
    device = get_device(config.device)
    records = load_records(data_dir)
    current_patient: str | None = None
    signal: torch.Tensor | None = None
    folds: list[Fold] = []

    while (job := jobs.get()) is not None:
        if time.time() >= deadline:
            continue
        patient_id, seizure_index = job
        started = time.monotonic()
        try:
            if patient_id != current_patient:
                current_patient = None
                signal = None
                signal = load_signal(records[patient_id], config.channels, device)
                folds = build_folds(records[patient_id], config)
                current_patient = patient_id
            fold = folds[seizure_index]
            train_fold(config, fold, signal, parent_run_id)
            logger.info("Finished %s in %.0f s", fold.key, time.monotonic() - started)
        except Exception:
            logger.exception("Failed %s/seizure-%d", patient_id, seizure_index)


def _worker_placement(workers_per_gpu: int) -> list[str | None]:
    """Return the GPU visible to each worker; None means the default device."""
    gpu_count = torch.cuda.device_count()
    logger.info("Visible GPUs: %d", gpu_count)
    if gpu_count == 0:
        return [None] * workers_per_gpu
    return [str(gpu) for gpu in range(gpu_count) for _ in range(workers_per_gpu)]


def run_folds(
    config: ExperimentConfig,
    data_dir: Path,
    parent_run_id: str,
    *,
    patients: list[str] | None = None,
    workers_per_gpu: int = 1,
    max_hours: float = 10.0,
) -> RunSummary:
    """Run all unfinished folds of an experiment in parallel worker processes."""
    records = load_records(data_dir)
    selected = patients or sorted(records)
    finished = {
        run.data.tags["fold_key"] for run in finished_fold_runs(config.experiment_name, config.name)
    }

    all_jobs = [
        (patient_id, index)
        for patient_id in selected
        for index in range(len(records[patient_id].seizures))
    ]
    pending = [
        (patient_id, index)
        for patient_id, index in all_jobs
        if f"{config.name}/{patient_id}/seizure-{index}" not in finished
    ]
    skipped = len(all_jobs) - len(pending)
    logger.info("%d folds to run, %d already finished", len(pending), skipped)

    started = time.monotonic()
    placement = _worker_placement(workers_per_gpu)
    context = mp.get_context("spawn")
    jobs = context.Queue()
    for job in pending:
        jobs.put(job)
    for _ in placement:
        jobs.put(None)

    deadline = time.time() + max_hours * 3600
    environment = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    os.environ["OMP_NUM_THREADS"] = "1"
    processes = []
    for index, gpu in enumerate(placement):
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        process = context.Process(
            target=_worker,
            args=(config, data_dir, parent_run_id, jobs, deadline),
            name=f"gpu{gpu}-worker{index}" if gpu is not None else f"worker{index}",
        )
        process.start()
        processes.append(process)
    for key, value in environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    for process in processes:
        process.join()
        if process.exitcode != 0:
            logger.error("%s exited with code %s", process.name, process.exitcode)

    done = {
        run.data.tags["fold_key"] for run in finished_fold_runs(config.experiment_name, config.name)
    }
    total_finished = sum(
        f"{config.name}/{patient_id}/seizure-{index}" in done for patient_id, index in all_jobs
    )
    summary = RunSummary(len(all_jobs), skipped, total_finished, time.monotonic() - started)
    logger.info(
        "%d of %d folds finished; session took %.0f s",
        summary.finished,
        summary.total_folds,
        summary.seconds,
    )
    return summary
