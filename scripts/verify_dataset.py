"""Verify a CHB-MIT copy against the checksums published by PhysioNet."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from neurofy.data.annotations import load_cases
from neurofy.data.verification import fetch_published_checksums, parse_checksums, verify_files
from neurofy.utils.logger import configure_logging
from neurofy.utils.paths import find_directory_containing

logger = logging.getLogger("neurofy.scripts.verify_dataset")

EXPECTED_SEIZURES = 198


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Dataset root; found automatically on Kaggle")
    parser.add_argument("--output", type=Path, default=Path("verification_report.json"))
    args = parser.parse_args()

    configure_logging()
    root = args.root or find_directory_containing("RECORDS")
    logger.info("Verifying %s", root)

    published = fetch_published_checksums()
    bundled_list = root / "SHA256SUMS.txt"
    bundled = parse_checksums(bundled_list.read_text(encoding="utf-8"))
    list_matches = bundled == published
    logger.info("Bundled checksum list matches PhysioNet: %s", list_matches)

    report = verify_files(root, published)
    seizures = sum(case.seizure_count for case in load_cases(root))

    summary = {
        **asdict(report),
        "checksum_list_matches_publisher": list_matches,
        "seizures": seizures,
        "expected_seizures": EXPECTED_SEIZURES,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "Matched %d of %d files; %d mismatched, %d missing; %d seizures",
        report.matched,
        report.total,
        len(report.mismatched),
        len(report.missing),
        seizures,
    )

    if not report.passed or not list_matches or seizures != EXPECTED_SEIZURES:
        logger.error("Verification failed; see %s", args.output)
        return 1
    logger.info("Verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
