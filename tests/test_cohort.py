from neurofy.data.annotations import Case, RecordingFile, Seizure
from neurofy.data.cohort import CohortRules, Interval, build_cohort, select_patient

HOUR = 3600
CHANNELS = ("FP1-F7", "T8-P8")
RULES = CohortRules()


def _file(
    name: str,
    start_hour: float,
    hours: float = 1,
    seizures: tuple[Seizure, ...] = (),
    channels: tuple[str, ...] = CHANNELS,
) -> RecordingFile:
    start = int(start_hour * HOUR)
    return RecordingFile(name, channels, seizures, start, start + int(hours * HOUR))


def _case(*files: RecordingFile) -> Case:
    return Case("chb99", files)


def _valid_case() -> Case:
    """Two qualifying seizures and interictal data 3 h away from both."""
    return _case(
        _file("a", 0, hours=4),
        _file("b", 4, hours=2, seizures=(Seizure(3600, 3660),)),
        _file("c", 6, hours=8),
        _file("d", 14, hours=2, seizures=(Seizure(3600, 3660),)),
    )


def test_selects_valid_patient() -> None:
    patient = select_patient(_valid_case(), RULES)

    assert patient is not None
    assert len(patient.seizures) == 2
    assert patient.seizures[0].onset == 5 * HOUR


def test_preictal_spans_minus_32_to_minus_2_minutes() -> None:
    patient = select_patient(_valid_case(), RULES)

    assert patient.seizures[0].preictal == (Interval("b", 3600 - 1920, 3600 - 120),)


def test_preictal_is_split_at_file_boundaries() -> None:
    case = _case(
        _file("a", 0, hours=6),
        _file("b", 6, seizures=(Seizure(600, 660),)),
        _file("c", 7, hours=8),
        _file("d", 15, seizures=(Seizure(3000, 3060),)),
    )
    patient = select_patient(case, RULES)

    assert [interval.file_name for interval in patient.seizures[0].preictal] == ["a", "b"]
    assert sum(interval.duration for interval in patient.seizures[0].preictal) == 1800


def test_rejects_seizure_within_gap_of_previous_one() -> None:
    case = _case(
        _file("a", 0, hours=8),
        _file("b", 8, seizures=(Seizure(3000, 3060),)),
        _file("c", 9, seizures=(Seizure(3000, 3060),)),
        _file("d", 10, hours=8),
        _file("e", 18, seizures=(Seizure(3000, 3060),)),
    )
    patient = select_patient(case, RULES)

    assert [seizure.file_name for seizure in patient.seizures] == ["b", "e"]


def test_rejects_seizure_with_incomplete_preictal_recording() -> None:
    case = _case(
        _file("a", 0, hours=8),
        _file("b", 8, seizures=(Seizure(3000, 3060),)),
        _file("c", 18, seizures=(Seizure(1000, 1060),)),
    )

    assert select_patient(case, RULES) is None


def test_allows_short_gap_at_file_boundary() -> None:
    case = _case(
        _file("a", 0, hours=6),
        RecordingFile(
            "b",
            CHANNELS,
            (Seizure(600, 660),),
            6 * HOUR + 10,
            7 * HOUR + 10,
        ),
        _file("c", 8, hours=8),
        _file("d", 16, seizures=(Seizure(3000, 3060),)),
    )
    patient = select_patient(case, RULES)

    assert patient is not None
    assert sum(interval.duration for interval in patient.seizures[0].preictal) == 1790


def test_rejects_patient_with_one_qualifying_seizure() -> None:
    case = _case(_file("a", 0, hours=8), _file("b", 8, seizures=(Seizure(3000, 3060),)))

    assert select_patient(case, RULES) is None


def test_rejects_patient_without_enough_interictal_data() -> None:
    case = _case(
        _file("a", 2, hours=2, seizures=(Seizure(3600, 3660),)),
        _file("b", 4, hours=8),
        _file("c", 12, hours=2, seizures=(Seizure(3600, 3660),)),
    )
    patient_without_interictal = select_patient(
        case,
        CohortRules(interictal_margin=12 * HOUR),
    )

    assert patient_without_interictal is None


def test_rejects_case_without_file_times() -> None:
    file = RecordingFile("a", CHANNELS, (Seizure(1, 2),), None, None)

    assert select_patient(_case(file), RULES) is None


def test_ignores_files_missing_required_channels() -> None:
    files = list(_valid_case().files)
    files[0] = _file("a", 0, hours=4, channels=("FP1-F7",))
    patient = select_patient(_case(*files), RULES)

    assert all(interval.file_name != "a" for interval in patient.interictal)


def test_interictal_stays_outside_margin_of_every_seizure() -> None:
    case = _valid_case()
    patient = select_patient(case, RULES)
    starts = {file.name: file.start for file in case.files}
    seizures = [
        (file.start + seizure.onset, file.start + seizure.offset)
        for file in case.files
        for seizure in file.seizures
    ]

    assert patient.interictal
    for interval in patient.interictal:
        start = starts[interval.file_name] + interval.start
        end = starts[interval.file_name] + interval.end
        for onset, offset in seizures:
            assert end <= onset - RULES.interictal_margin or (
                start >= offset + RULES.interictal_margin
            )


def test_build_cohort_keeps_only_selected_patients() -> None:
    excluded = Case("chb00", (_file("x", 0),))

    cohort = build_cohort([excluded, _valid_case()], RULES)

    assert [patient.case_id for patient in cohort] == ["chb99"]
