"""Patient-specific leave-one-seizure-out folds."""

import zlib
from dataclasses import dataclass

import numpy as np

from neurofy.config import ExperimentConfig
from neurofy.data.dataset import PatientRecord


@dataclass(frozen=True)
class Fold:
    """Window start positions for one held-out seizure."""

    patient_id: str
    seizure_index: int
    seed: int
    train_preictal: np.ndarray
    train_interictal: np.ndarray
    test_preictal: np.ndarray
    test_seconds_to_onset: np.ndarray
    test_interictal: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.patient_id}/seizure-{self.seizure_index}"


def fold_seed(base_seed: int, patient_id: str, seizure_index: int) -> int:
    """Derive a stable seed; excluding the model keeps held-out data identical across models."""
    return zlib.crc32(f"{base_seed}/{patient_id}/{seizure_index}".encode())


def window_starts(start: int, end: int, window: int, hop: int) -> np.ndarray:
    """Return the start of every window fully inside a sample range."""
    return np.arange(start, end - window + 1, hop, dtype=np.int64)


def _choose_held_out_start(
    ranges: tuple[tuple[int, int], ...],
    length: int,
    rng: np.random.Generator,
) -> int:
    """Pick a uniformly random position for a continuous segment of `length` samples."""
    candidates = [(start, end) for start, end in ranges if end - start >= length]
    if not candidates:
        raise ValueError(f"No interictal range holds {length} continuous samples")
    positions = np.array([end - start - length + 1 for start, end in candidates])
    choice = int(rng.integers(positions.sum()))
    for (start, _), count in zip(candidates, positions, strict=True):
        if choice < count:
            return start + choice
        choice -= int(count)
    raise AssertionError("Position choice is always inside the candidates")


def _preictal_windows(
    ranges: tuple[tuple[int, int, int], ...],
    config: ExperimentConfig,
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a seizure's preictal window starts and their seconds to onset."""
    starts, seconds = [], []
    for start, end, seconds_to_onset in ranges:
        positions = window_starts(start, end, config.window_samples, config.hop_samples)
        starts.append(positions)
        seconds.append(seconds_to_onset - (positions - start) / sampling_rate)
    return np.concatenate(starts), np.concatenate(seconds)


def build_folds(record: PatientRecord, config: ExperimentConfig) -> list[Fold]:
    """Create one fold per qualifying seizure of a patient."""
    preictal = [
        _preictal_windows(seizure, config, record.sampling_rate) for seizure in record.seizures
    ]
    interictal = np.concatenate(
        [
            window_starts(start, end, config.window_samples, config.hop_samples)
            for start, end in record.interictal
        ]
    )

    folds = []
    for index, (test_preictal, seconds_to_onset) in enumerate(preictal):
        seed = fold_seed(config.seed, record.patient_id, index)
        held_out = _choose_held_out_start(
            record.interictal,
            config.held_out_interictal_samples,
            np.random.default_rng(seed),
        )
        held_out_end = held_out + config.held_out_interictal_samples
        outside = (interictal + config.window_samples <= held_out) | (interictal >= held_out_end)
        folds.append(
            Fold(
                patient_id=record.patient_id,
                seizure_index=index,
                seed=seed,
                train_preictal=np.concatenate(
                    [starts for other, (starts, _) in enumerate(preictal) if other != index]
                ),
                train_interictal=interictal[outside],
                test_preictal=test_preictal,
                test_seconds_to_onset=seconds_to_onset,
                test_interictal=window_starts(
                    held_out,
                    held_out_end,
                    config.window_samples,
                    config.hop_samples,
                ),
            )
        )
    return folds


def sample_training_windows(
    fold: Fold,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw balanced, shuffled training windows for one epoch."""
    count = min(len(fold.train_preictal), len(fold.train_interictal))
    interictal = rng.choice(fold.train_interictal, size=count, replace=False)
    starts = np.concatenate([fold.train_preictal, interictal])
    labels = np.concatenate([np.ones(len(fold.train_preictal)), np.zeros(count)])
    order = rng.permutation(len(starts))
    return starts[order], labels[order]
