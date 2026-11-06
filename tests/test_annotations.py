from neurofy.data.annotations import Seizure, parse_summary

SUMMARY = """
Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: T8-P8
Channel 3: T8-P8

File Name: chb99_01.edf
File Start Time: 23:30:00
File End Time: 24:30:00
Number of Seizures in File: 1
Seizure Start Time: 100 seconds
Seizure End Time: 160 seconds

File Name: chb99_02.edf
File Start Time: 00:30:10
File End Time: 1:30:10
Number of Seizures in File: 2
Seizure 1 Start Time: 10 seconds
Seizure 1 End Time: 20 seconds
Seizure 2 Start Time: 30 seconds
Seizure 2 End Time: 40 seconds

Channels changed:
*****************
Channel 1: fp1-f7
Channel 2: -

File Name: chb99_03.edf
File Start Time: 01:30:20
File End Time: 02:30:20
Number of Seizures in File: 0
"""

NO_TIMES = """
Channels in EDF Files:
Channel 1: FP1-F7

File Name: chb24_01.edf
Number of Seizures in File: 1
Seizure Start Time: 5 seconds
Seizure End Time: 6 seconds
"""


def test_parses_both_seizure_formats() -> None:
    case = parse_summary(SUMMARY, "chb99")

    assert case.seizure_count == 3
    assert case.files[0].seizures == (Seizure(100, 160),)
    assert case.files[1].seizures == (Seizure(10, 20), Seizure(30, 40))


def test_handles_midnight_rollover_and_times_past_24() -> None:
    files = parse_summary(SUMMARY, "chb99").files

    assert (files[0].start, files[0].end) == (84_600, 88_200)
    assert (files[1].start, files[1].end) == (88_210, 91_810)
    assert files[2].start == 91_820


def test_applies_channel_changes_to_later_files() -> None:
    files = parse_summary(SUMMARY, "chb99").files
    both_channels = ("FP1-F7", "T8-P8")

    assert files[0].channels == ("FP1-F7", "T8-P8", "T8-P8")
    assert files[1].has_channels(both_channels)
    assert files[2].has_channels(("FP1-F7",))
    assert not files[2].has_channels(both_channels)


def test_marks_cases_without_file_times() -> None:
    case = parse_summary(NO_TIMES, "chb24")

    assert not case.has_file_times
    assert case.seizure_count == 1
