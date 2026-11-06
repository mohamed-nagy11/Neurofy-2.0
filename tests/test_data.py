from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from conftest import write_synthetic_dataset
from neurofy.config import ExperimentConfig
from neurofy.data.dataset import PatientRecord, WindowLoader, load_records, load_signal
from neurofy.data.folds import build_folds, fold_seed, sample_training_windows, window_starts


def _record() -> PatientRecord:
    return PatientRecord(
        patient_id="chb01",
        signal_path=Path("unused"),
        stored_channels=("FP1-F7", "T8-P8"),
        sampling_rate=256,
        seizures=(
            ((0, 2560, 130),),
            ((10_000, 11_280, 130), (11_300, 12_580, 65)),
            ((20_000, 22_560, 130),),
        ),
        interictal=((30_000, 40_000), (50_000, 70_000)),
    )


def test_window_starts_stay_inside_the_range() -> None:
    assert window_starts(100, 700, 256, 128).tolist() == [100, 228, 356]
    assert window_starts(0, 100, 256, 128).size == 0


def test_one_fold_per_seizure_holds_out_that_seizure(config: ExperimentConfig) -> None:
    record = _record()
    folds = build_folds(record, config)

    assert len(folds) == 3
    for index, fold in enumerate(folds):
        assert fold.seizure_index == index
        held_out = set(fold.test_preictal.tolist())
        assert held_out
        assert held_out.isdisjoint(fold.train_preictal.tolist())
    assert set(folds[0].train_preictal.tolist()) == set(folds[1].test_preictal.tolist()) | set(
        folds[2].test_preictal.tolist()
    )


def test_windows_do_not_cross_preictal_gaps(config: ExperimentConfig) -> None:
    fold = build_folds(_record(), config)[1]

    assert np.all((fold.test_preictal + 256 <= 11_280) | (fold.test_preictal >= 11_300))


def test_held_out_interictal_never_overlaps_training(config: ExperimentConfig) -> None:
    for fold in build_folds(_record(), config):
        start = fold.test_interictal[0]
        end = fold.test_interictal[-1] + config.window_samples

        assert end - start <= config.held_out_interictal_samples
        train = fold.train_interictal
        assert np.all((train + config.window_samples <= start) | (train >= end))
        assert train.size > 0


def test_seconds_to_onset_decreases_across_a_seizure(config: ExperimentConfig) -> None:
    fold = build_folds(_record(), config)[0]

    assert fold.test_seconds_to_onset[0] == 130
    assert fold.test_seconds_to_onset[1] == 130 - config.hop_samples / 256
    assert np.all(np.diff(fold.test_seconds_to_onset) < 0)


def test_folds_are_deterministic_and_shared_across_models(config: ExperimentConfig) -> None:
    first = build_folds(_record(), config)
    other_model = replace(config, name="fp1f7_t8p8", channels=("FP1-F7", "T8-P8"))
    second = build_folds(_record(), other_model)

    for left, right in zip(first, second, strict=True):
        assert left.seed == right.seed
        assert np.array_equal(left.test_interictal, right.test_interictal)
    assert fold_seed(42, "chb01", 0) != fold_seed(42, "chb01", 1)


def test_training_windows_are_balanced_and_redrawn(config: ExperimentConfig) -> None:
    fold = build_folds(_record(), config)[0]
    rng = np.random.default_rng(0)

    first_starts, first_labels = sample_training_windows(fold, rng)
    second_starts, _ = sample_training_windows(fold, rng)

    assert first_labels.sum() == len(first_labels) - first_labels.sum()
    first_interictal = set(first_starts[first_labels == 0].tolist())
    assert first_interictal <= set(fold.train_interictal.tolist())
    assert not np.array_equal(first_starts, second_starts)


def test_loader_slices_expected_samples() -> None:
    signal = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    loader = WindowLoader(signal, np.array([0, 5, 10]), np.array([1, 0, 1]), 4, 2)

    batches = list(loader)

    assert len(loader) == 2
    assert batches[0][0].shape == (2, 2, 4)
    assert batches[0][0][1, 0].tolist() == [5, 6, 7, 8]
    assert batches[0][0][1, 1].tolist() == [25, 26, 27, 28]
    assert batches[1][1].tolist() == [1.0]


def test_loads_records_and_selects_channels(tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path)
    record = load_records(tmp_path)["chb01"]

    single = load_signal(record, ("FP1-F7",), torch.device("cpu"))
    swapped = load_signal(record, ("T8-P8", "FP1-F7"), torch.device("cpu"))

    assert single.shape[0] == 1
    assert swapped.shape[0] == 2
    assert torch.equal(swapped[1], single[0])
    assert len(record.seizures) == 2
