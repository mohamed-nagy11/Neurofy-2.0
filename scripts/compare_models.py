"""Compare the two models with each other and with the paper's results."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import mlflow

from neurofy.config import get_config
from neurofy.evaluation.aggregation import rows_to_csv, summarize_cohort, summarize_patients
from neurofy.evaluation.comparison import compare_with_paper, paired_comparison
from neurofy.evaluation.reporting import fold_results
from neurofy.utils.logger import configure_logging
from neurofy.utils.tracking import RESULTS_ARCHIVE, experiment_run, tracking_session

logger = logging.getLogger("neurofy.scripts.compare_models")

FIRST = "fp1f7"
SECOND = "fp1f7_t8p8"


def _compare(experiment: str) -> None:
    first = summarize_patients(fold_results(experiment, FIRST))
    second = summarize_patients(fold_results(experiment, SECOND))
    if not first or not second:
        raise ValueError("Both models need finished folds before comparing them")

    config = replace(get_config(FIRST), experiment_name=experiment)
    with experiment_run(config, "comparison", run_name="comparison", tags={"config": "both"}):
        paper = compare_with_paper(summarize_cohort(first))
        paired = paired_comparison(first, second)
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
    args = parser.parse_args()

    configure_logging()
    if args.tracking_dir:
        with tracking_session(args.tracking_dir, args.archive):
            _compare(args.experiment)
    else:
        _compare(args.experiment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
