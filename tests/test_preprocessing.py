import json
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from neurofy.data.annotations import Case, RecordingFile, Seizure
from neurofy.data.cohort import CohortRules, select_patient
from neurofy.data.preprocessing import (
    SAMPLING_RATE,
    _channel_index,
    bandpass,
    design_filter,
    preprocess_cohort,
)

HOUR = 3600
CHANNELS = ("FP1-F7", "T8-P8")


def _sine(frequency: float, seconds: int = 60) -> np.ndarray:
    time = np.arange(seconds * SAMPLING_RATE) / SAMPLING_RATE
    return np.sin(2 * np.pi * frequency * time)


def _rms_after_settling(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal[len(signal) // 2 :] ** 2)))


def test_filter_passes_band_and_attenuates_outside() -> None:
    sos = design_filter()

    passed = _rms_after_settling(bandpass(_sine(10), sos))
    attenuated = _rms_after_settling(bandpass(_sine(120), sos))
    offset = _rms_after_settling(bandpass(np.ones(600 * SAMPLING_RATE), sos))

    assert passed == pytest.approx(1 / np.sqrt(2), rel=0.05)
    assert attenuated < 0.01
    assert offset < 0.01


def test_filter_is_causal() -> None:
    signal = np.zeros(2000)
    signal[1000] = 1.0

    filtered = bandpass(signal, design_filter())

    assert np.allclose(filtered[:1000], 0.0)
    assert filtered[1000] != 0.0


def test_channel_lookup_accepts_duplicate_suffix_and_case() -> None:
    labels = ["FP1-F7", "fp1-f3", "T8-P8-0", "T8-P8-1"]

    assert _channel_index(labels, "T8-P8") == 2
    assert _channel_index(labels, "FP1-F7") == 0
    with pytest.raises(ValueError):
        _channel_index(labels, "O2-P8")


def _write_edf(path: Path, seconds: int, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["FP1-F7", "F7-T7", "T8-P8-0", "T8-P8-1"]
    headers = [
        {
            "label": label,
            "dimension": "uV",
            "sample_frequency": SAMPLING_RATE,
            "physical_max": 1000.0,
            "physical_min": -1000.0,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "",
            "prefilter": "",
        }
        for label in labels
    ]
    signal = 100 * _sine(frequency, seconds)
    with pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDF) as writer:
        writer.setSignalHeaders(headers)
        writer.writeSamples([signal, signal, signal * 2, signal])


def test_preprocessing_stores_labeled_files_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    files = (
        RecordingFile("chb99_01.edf", CHANNELS, (), 0, 3600),
        RecordingFile("chb99_02.edf", CHANNELS, (), 4 * HOUR, 5 * HOUR),
        RecordingFile("chb99_03.edf", CHANNELS, (Seizure(3000, 3060),), 8 * HOUR, 9 * HOUR),
        RecordingFile("chb99_04.edf", CHANNELS, (Seizure(3000, 3060),), 20 * HOUR, 21 * HOUR),
    )
    for file in files:
        _write_edf(root / "chb99" / file.name, 60, 10)
    # Short recordings stand in for the hour-long files listed above.
    rules = CohortRules(
        min_preictal_recorded=1790,
        held_out_interictal=30,
        interictal_margin=3 * HOUR,
    )
    case = Case("chb99", files)
    patient = select_patient(case, rules)
    assert patient is not None

    manifest = preprocess_cohort([patient], [case], root, tmp_path / "out", rules)

    entry = manifest["patients"]["chb99"]
    signal = np.load(tmp_path / "out" / entry["signal"], mmap_mode="r")
    written = json.loads((tmp_path / "out" / "manifest.json").read_text())

    assert written == manifest
    assert signal.dtype == np.float32
    assert signal.shape == (2, entry["samples"])
    assert entry["samples"] == len(entry["files"]) * 60 * SAMPLING_RATE
    # T8-P8 was written with twice the amplitude of FP1-F7.
    assert np.std(signal[1]) == pytest.approx(2 * np.std(signal[0]), rel=0.01)
    assert len(entry["seizures"]) == 2
    for sample_range in entry["interictal"]:
        assert 0 <= sample_range["start"] < sample_range["end"] <= entry["samples"]
