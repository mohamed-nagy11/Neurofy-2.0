"""Cohort selection and preictal and interictal labeling."""

import logging
from dataclasses import dataclass

from neurofy.data.annotations import Case, RecordingFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohortRules:
    """Selection and labeling rules; durations are in seconds."""

    channels: tuple[str, ...] = ("FP1-F7", "T8-P8")
    sph: int = 120
    sop: int = 1800
    seizure_gap: int = 10_800
    interictal_margin: int = 10_800
    min_preictal_recorded: int = 1790
    min_seizures: int = 2
    held_out_interictal: int = 1800


@dataclass(frozen=True)
class Interval:
    """A time range in seconds from the start of a file."""

    file_name: str
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LabeledSeizure:
    """A qualifying seizure and its preictal intervals."""

    file_name: str
    onset: int
    preictal: tuple[Interval, ...]


@dataclass(frozen=True)
class Patient:
    """A patient in the cohort and their labeled data."""

    case_id: str
    seizures: tuple[LabeledSeizure, ...]
    interictal: tuple[Interval, ...]


def _clip_to_files(
    files: list[RecordingFile],
    start: int,
    end: int,
) -> tuple[Interval, ...]:
    """Intersect an absolute time range with each file."""
    intervals = []
    for file in files:
        clipped_start, clipped_end = max(start, file.start), min(end, file.end)
        if clipped_start < clipped_end:
            intervals.append(
                Interval(file.name, clipped_start - file.start, clipped_end - file.start)
            )
    return tuple(intervals)


def _subtract_zones(
    file: RecordingFile,
    zones: list[tuple[int, int]],
) -> list[Interval]:
    """Return the parts of a file outside the absolute exclusion zones."""
    intervals = []
    cursor = file.start
    for zone_start, zone_end in sorted(zones):
        if zone_start > cursor:
            end = min(zone_start, file.end)
            if cursor < end:
                intervals.append(Interval(file.name, cursor - file.start, end - file.start))
        cursor = max(cursor, zone_end)
    if cursor < file.end:
        intervals.append(Interval(file.name, cursor - file.start, file.end - file.start))
    return intervals


def select_patient(case: Case, rules: CohortRules) -> Patient | None:
    """Apply the cohort rules to a case; return None if the case is excluded."""
    if not case.has_file_times:
        logger.info("Excluding %s: no file times", case.case_id)
        return None

    usable_files = [file for file in case.files if file.has_channels(rules.channels)]
    seizures = sorted(
        (
            (file.start + seizure.onset, file.start + seizure.offset, file.name)
            for file in case.files
            for seizure in file.seizures
        ),
    )

    qualifying = []
    for index, (onset, _, file_name) in enumerate(seizures):
        if index > 0 and onset - seizures[index - 1][1] < rules.seizure_gap:
            continue
        preictal = _clip_to_files(
            usable_files,
            onset - rules.sph - rules.sop,
            onset - rules.sph,
        )
        if sum(interval.duration for interval in preictal) >= rules.min_preictal_recorded:
            qualifying.append(LabeledSeizure(file_name, onset, preictal))

    if len(qualifying) < rules.min_seizures:
        logger.info("Excluding %s: %d qualifying seizures", case.case_id, len(qualifying))
        return None

    zones = [
        (onset - rules.interictal_margin, offset + rules.interictal_margin)
        for onset, offset, _ in seizures
    ]
    interictal = tuple(
        interval for file in usable_files for interval in _subtract_zones(file, zones)
    )

    required = len(qualifying) * rules.held_out_interictal
    available = sum(interval.duration for interval in interictal)
    longest = max((interval.duration for interval in interictal), default=0)
    if available < required or longest < rules.held_out_interictal:
        logger.info(
            "Excluding %s: %.1f h interictal, longest segment %.1f h",
            case.case_id,
            available / 3600,
            longest / 3600,
        )
        return None

    return Patient(case.case_id, tuple(qualifying), interictal)


def build_cohort(cases: list[Case], rules: CohortRules) -> list[Patient]:
    """Select the cohort from all parsed cases."""
    cohort = [patient for case in cases if (patient := select_patient(case, rules)) is not None]
    logger.info(
        "Cohort: %d patients, %d seizures",
        len(cohort),
        sum(len(patient.seizures) for patient in cohort),
    )
    return cohort
