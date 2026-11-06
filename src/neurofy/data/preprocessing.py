"""Signal preprocessing and storage for the selected cohort."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pyedflib
from scipy.signal import butter, sosfilt, sosfilt_zi

from neurofy.data.annotations import Case
from neurofy.data.cohort import CohortRules, Interval, Patient

logger = logging.getLogger(__name__)

SAMPLING_RATE = 256
FILTER_ORDER = 5
FILTER_BAND = (0.1, 50.0)


def design_filter(sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Design the band-pass filter as second-order sections."""
    return butter(FILTER_ORDER, FILTER_BAND, btype="bandpass", fs=sampling_rate, output="sos")


def bandpass(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter causally, starting from the steady state of the first sample."""
    initial_state = sosfilt_zi(sos) * signal[0]
    filtered, _ = sosfilt(sos, signal, zi=initial_state)
    return filtered


def _channel_index(labels: list[str], channel: str) -> int:
    """Find a channel by name; CHB-MIT labels duplicates as `T8-P8-0` and `T8-P8-1`."""
    for index, label in enumerate(labels):
        normalized = label.strip().upper()
        if normalized == channel or normalized.startswith(f"{channel}-"):
            return index
    raise ValueError(f"Channel {channel} not found in {labels}")


def _sample_count(path: Path, channel: str) -> int:
    with pyedflib.EdfReader(str(path)) as reader:
        return int(reader.getNSamples()[_channel_index(reader.getSignalLabels(), channel)])


def read_filtered_channels(path: Path, channels: tuple[str, ...], sos: np.ndarray) -> np.ndarray:
    """Read channels from an EDF file and band-pass filter each one."""
    with pyedflib.EdfReader(str(path)) as reader:
        labels = reader.getSignalLabels()
        signals = []
        for channel in channels:
            index = _channel_index(labels, channel)
            rate = reader.getSampleFrequency(index)
            if rate != SAMPLING_RATE:
                raise ValueError(f"{path.name}: {channel} is sampled at {rate} Hz")
            signals.append(bandpass(reader.readSignal(index), sos))
    return np.stack(signals).astype(np.float32)


def _to_samples(
    intervals: tuple[Interval, ...],
    files: dict[str, dict],
) -> list[dict]:
    """Convert file-relative intervals to positions in the patient's stored signal."""
    ranges = []
    for interval in intervals:
        file = files[interval.file_name]
        start = file["offset"] + interval.start * SAMPLING_RATE
        end = file["offset"] + min(interval.end * SAMPLING_RATE, file["samples"])
        if start < end:
            ranges.append({"start": int(start), "end": int(end)})
    return ranges


def preprocess_patient(
    patient: Patient,
    case: Case,
    root: Path,
    output_dir: Path,
    rules: CohortRules,
) -> dict:
    """Filter a patient's labeled files into one array and return their manifest entry."""
    used_files = sorted(
        {interval.file_name for interval in patient.interictal}
        | {interval.file_name for seizure in patient.seizures for interval in seizure.preictal}
    )
    starts = {file.name: file.start for file in case.files}

    files = {}
    total_samples = 0
    for name in used_files:
        samples = _sample_count(root / case.case_id / name, rules.channels[0])
        files[name] = {"offset": total_samples, "samples": samples}
        total_samples += samples

    signal_path = output_dir / "signals" / f"{case.case_id}.npy"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.lib.format.open_memmap(
        signal_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rules.channels), total_samples),
    )
    sos = design_filter()
    for name, file in files.items():
        filtered = read_filtered_channels(root / case.case_id / name, rules.channels, sos)
        signal[:, file["offset"] : file["offset"] + file["samples"]] = filtered
    signal.flush()
    del signal

    seizures = []
    for seizure in patient.seizures:
        preictal = []
        for interval in seizure.preictal:
            for sample_range in _to_samples((interval,), files):
                seconds_to_onset = seizure.onset - (starts[interval.file_name] + interval.start)
                preictal.append({**sample_range, "seconds_to_onset": seconds_to_onset})
        seizures.append({"file": seizure.file_name, "onset": seizure.onset, "preictal": preictal})

    logger.info(
        "%s: %d files, %.1f h stored, %d seizures",
        case.case_id,
        len(files),
        total_samples / SAMPLING_RATE / 3600,
        len(seizures),
    )
    return {
        "signal": signal_path.relative_to(output_dir).as_posix(),
        "samples": total_samples,
        "files": files,
        "seizures": seizures,
        "interictal": _to_samples(patient.interictal, files),
    }


def preprocess_cohort(
    cohort: list[Patient],
    cases: list[Case],
    root: Path,
    output_dir: Path,
    rules: CohortRules,
) -> dict:
    """Preprocess every patient and write the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_by_id = {case.case_id: case for case in cases}
    manifest = {
        "channels": list(rules.channels),
        "sampling_rate": SAMPLING_RATE,
        "filter": {"order": FILTER_ORDER, "band": list(FILTER_BAND), "causal": True},
        "git_commit": os.environ.get("NEUROFY_GIT_COMMIT", "unknown"),
        "patients": {
            patient.case_id: preprocess_patient(
                patient,
                cases_by_id[patient.case_id],
                root,
                output_dir,
                rules,
            )
            for patient in cohort
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
