"""Run a patient-specific experiment across all folds and report the results."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from neurofy.config import CONFIGS, get_config
from neurofy.evaluation.reporting import log_experiment_report
from neurofy.training.runner import run_folds
from neurofy.utils.logger import configure_logging
from neurofy.utils.paths import find_directory_containing
from neurofy.utils.tracking import (
    RESULTS_ARCHIVE,
    experiment_run,
    find_experiment_run_id,
    tracking_session,
)

logger = logging.getLogger("neurofy.scripts.run_experiment")


def _run(config, data_dir: Path, args: argparse.Namespace) -> int:
    run_id = find_experiment_run_id(config.experiment_name, config.name)
    with experiment_run(config, "experiment", run_name=config.name, run_id=run_id) as run:
        summary = run_folds(
            config,
            data_dir,
            run.info.run_id,
            patients=args.patients,
            workers_per_gpu=args.workers_per_gpu,
            max_hours=args.max_hours,
        )
        cohort = log_experiment_report(config.experiment_name, config.name)

    for metric in ("segment/accuracy", "segment/fpr", "event/sensitivity"):
        if metric in cohort:
            logger.info(
                "%s: %.4f ± %.4f",
                metric,
                cohort[metric]["mean"],
                cohort[metric]["std"],
            )
    if summary.remaining:
        logger.warning("%d folds remain; run again to resume", summary.remaining)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--data-dir", type=Path, help="Found automatically on Kaggle")
    parser.add_argument("--tracking-dir", type=Path, help="Serve a local MLflow store from here")
    parser.add_argument("--archive", type=Path, default=Path(RESULTS_ARCHIVE))
    parser.add_argument("--experiment", help="MLflow experiment name")
    parser.add_argument("--patients", nargs="+", help="Run only these patients")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--device", help="Override the device, such as cpu")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=10.0)
    args = parser.parse_args()

    configure_logging()
    overrides = {
        "experiment_name": args.experiment,
        "max_epochs": args.max_epochs,
        "device": args.device,
    }
    config = replace(
        get_config(args.config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    data_dir = args.data_dir or find_directory_containing("manifest.json")

    if args.tracking_dir:
        with tracking_session(args.tracking_dir, args.archive):
            return _run(config, data_dir, args)
    return _run(config, data_dir, args)


if __name__ == "__main__":
    sys.exit(main())
