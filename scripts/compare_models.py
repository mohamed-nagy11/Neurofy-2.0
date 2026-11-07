"""Compare the two models with each other and with the paper's results."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow

from neurofy.config import get_config
from neurofy.evaluation.aggregation import rows_to_csv, summarize_cohort, summarize_patients
from neurofy.evaluation.comparison import compare_with_paper, paired_comparison
from neurofy.evaluation.reporting import fold_results
from neurofy.utils.logger import configure_logging
from neurofy.utils.paths import KAGGLE_INPUT
from neurofy.utils.tracking import RESULTS_ARCHIVE, experiment_run, tracking_session

logger = logging.getLogger("neurofy.scripts.compare_models")

FIRST = "fp1f7"
SECOND = "fp1f7_t8p8"


def _read_rows(experiment: str, config_name: str) -> list[dict]:
    return summarize_patients(fold_results(experiment, config_name))


def _read_rows_from_results(
    experiment: str,
    config_name: str,
    source: str,
    input_dir: Path,
) -> list[dict]:
    """Read a model's patient rows from results archived by another session."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        with tracking_session(
            Path(tmp_dir) / "store",
            Path(tmp_dir) / "unused.tar.gz",
            restore_from=source,
            search_root=input_dir,
        ):
            return _read_rows(experiment, config_name)


def _report(experiment: str, first: list[dict], second: list[dict]) -> None:
    if not first or not second:
        raise ValueError("Both models need finished folds before comparing them")

    config = replace(get_config(FIRST), experiment_name=experiment)
    with experiment_run(config, "comparison", run_name="comparison", tags={"config": "both"}):
        paper = compare_with_paper(summarize_cohort(first))
        paired = paired_comparison(first, second)
        mlflow.log_text(rows_to_csv(first), f"patient_metrics_{FIRST}.csv")
        mlflow.log_text(rows_to_csv(second), f"patient_metrics_{SECOND}.csv")
        mlflow.log_text(rows_to_csv(paper), "paper_comparison.csv")
        mlflow.log_text(rows_to_csv(paired), "model_comparison.csv")

        for row in paper:
            logger.info("Paper | %s: %.4f +/- %.4f", row["metric"], row["mean"], row["std"])
        for row in paired:
            logger.info(
                "Models | %s: %.4f vs %.4f, difference %.4f, p=%.3f, %d patients changed",
                row["metric"],
                row["first_mean"],
                row["second_mean"],
                row["mean_difference"],
                row["wilcoxon_p"],
                row["patients_changed"],
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-dir", type=Path, help="Serve a local MLflow store from here")
    parser.add_argument("--archive", type=Path, default=Path(RESULTS_ARCHIVE))
    parser.add_argument("--experiment", default=get_config(FIRST).experiment_name)
    parser.add_argument("--first-results", help="Attached kernel holding Model 1's results")
    parser.add_argument("--second-results", help="Attached kernel holding Model 2's results")
    parser.add_argument("--input-dir", type=Path, default=KAGGLE_INPUT)
    args = parser.parse_args()

    configure_logging()
    if bool(args.first_results) != bool(args.second_results):
        parser.error("--first-results and --second-results go together")

    if args.first_results:
        first = _read_rows_from_results(args.experiment, FIRST, args.first_results, args.input_dir)
        second = _read_rows_from_results(
            args.experiment, SECOND, args.second_results, args.input_dir
        )
        with tracking_session(args.tracking_dir, args.archive):
            _report(args.experiment, first, second)
    elif args.tracking_dir:
        with tracking_session(args.tracking_dir, args.archive):
            _report(
                args.experiment,
                _read_rows(args.experiment, FIRST),
                _read_rows(args.experiment, SECOND),
            )
    else:
        _report(
            args.experiment,
            _read_rows(args.experiment, FIRST),
            _read_rows(args.experiment, SECOND),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
