"""Filter the selected CHB-MIT cohort and store it with a manifest."""

import argparse
import logging
import sys
from pathlib import Path

from neurofy.data.annotations import load_cases
from neurofy.data.cohort import CohortRules, build_cohort
from neurofy.data.preprocessing import preprocess_cohort
from neurofy.utils.logger import configure_logging
from neurofy.utils.paths import find_directory_containing

logger = logging.getLogger("neurofy.scripts.preprocess")

EXPECTED_PATIENTS = 17
EXPECTED_SEIZURES = 55


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Dataset root; found automatically on Kaggle")
    parser.add_argument("--output", type=Path, default=Path("chbmit_preprocessed"))
    args = parser.parse_args()

    configure_logging()
    root = args.root or find_directory_containing("RECORDS")
    rules = CohortRules()

    cases = load_cases(root)
    cohort = build_cohort(cases, rules)
    seizures = sum(len(patient.seizures) for patient in cohort)
    if (len(cohort), seizures) != (EXPECTED_PATIENTS, EXPECTED_SEIZURES):
        logger.error("Cohort is %d patients and %d seizures", len(cohort), seizures)
        return 1

    preprocess_cohort(cohort, cases, root, args.output, rules)
    size = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    logger.info("Stored %d patients in %s (%.2f GB)", len(cohort), args.output, size / 1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
