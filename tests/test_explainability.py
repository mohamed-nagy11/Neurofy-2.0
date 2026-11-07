from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from neurofy.config import CONFIGS
from neurofy.data.folds import Fold
from neurofy.evaluation.explainability import (
    BANDS,
    _shifted_maps,
    band_importance,
    band_importance_by_time,
    bin_frequencies,
    channel_ablation,
    deletion_test,
    explain_fold,
    grad_cam,
    map_similarity,
    sample_windows_by_outcome,
)
from neurofy.evaluation.plots import (
    plot_band_importance,
    plot_band_importance_by_time,
    plot_deletion_test,
    plot_window_heatmap,
)
from neurofy.models.mobilenet import create_model

SINGLE = CONFIGS["fp1f7"]
DUAL = CONFIGS["fp1f7_t8p8"]
RATE = SINGLE.sampling_rate
SAMPLES = SINGLE.window_samples


def _sine(frequency: float, count: int) -> torch.Tensor:
    time = torch.arange(SAMPLES) / RATE
    phases = torch.linspace(0, 3, count)
    waves = [torch.sin(2 * torch.pi * frequency * time + phase) for phase in phases]
    return torch.stack(waves).unsqueeze(1) * 20


@pytest.mark.parametrize("config", [SINGLE, DUAL])
def test_grad_cam_matches_input_size_and_range(config) -> None:
    model = create_model(config).eval()
    spectrograms = torch.rand(2, len(config.channels), 80, 63)

    maps = grad_cam(model, spectrograms)

    assert maps.shape == (2, 80, 63)
    assert maps.min() >= 0
    assert maps.max() <= 1


def test_grad_cam_removes_hooks_and_keeps_outputs() -> None:
    model = create_model(SINGLE).eval()
    spectrograms = torch.rand(2, 1, 80, 63)
    before = model.forward_spectrogram(spectrograms).detach()

    grad_cam(model, spectrograms)

    assert not model.features[-1]._forward_hooks
    assert torch.equal(model.forward_spectrogram(spectrograms).detach(), before)


def test_grad_cam_highlights_the_trained_frequency_band() -> None:
    torch.manual_seed(0)
    # Both classes share low-frequency content, so only the 40 Hz tone separates them.
    negative = _sine(3, 16)
    positive = negative + _sine(40, 16)
    windows = torch.cat([positive, negative])
    labels = torch.tensor([1.0] * 16 + [0.0] * 16)
    model = create_model(SINGLE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(60):
        batch = torch.randperm(len(windows))[:16]
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(windows[batch]), labels[batch])
        loss.backward()
        optimizer.step()
    model.eval()

    assert torch.equal((model(windows) > 0).float(), labels)
    maps = grad_cam(model, model.spectrogram(positive)).numpy()
    importance = band_importance(maps, bin_frequencies(SINGLE))

    assert importance["gamma"] > 0.3
    assert importance["delta"] < 0.1
    assert importance["theta"] < 0.1


def test_band_importance_averages_known_bins() -> None:
    frequencies = bin_frequencies(SINGLE)
    maps = np.zeros((2, 80, 63))
    maps[:, :, (frequencies >= 13) & (frequencies < 30)] = 1.0

    importance = band_importance(maps, frequencies)

    assert importance["beta"] == pytest.approx(1.0)
    assert importance["delta"] == 0.0
    assert set(importance) == set(BANDS)


def test_bands_cover_every_bin() -> None:
    frequencies = bin_frequencies(SINGLE)

    assert frequencies[0] == pytest.approx(0.8)
    assert frequencies[-1] == pytest.approx(50.4)
    covered = sum(
        ((frequencies >= low) & (frequencies < high)).sum() for low, high in BANDS.values()
    )
    assert covered == 63


def test_band_importance_by_time_bins_by_minutes() -> None:
    frequencies = bin_frequencies(SINGLE)
    maps = np.zeros((4, 80, 63))
    maps[0:2] = 1.0
    seconds = np.array([590.0, 500.0, 200.0, 130.0])

    rows = band_importance_by_time(maps, seconds, frequencies)

    assert [row["minutes_before_onset"] for row in rows] == [2.5, 7.5]
    assert rows[0]["alpha"] == 0.0
    assert rows[1]["alpha"] == 1.0


def test_deletion_test_reports_every_fraction() -> None:
    model = create_model(SINGLE).eval()
    spectrograms = torch.rand(4, 1, 80, 63)
    maps = grad_cam(model, spectrograms)

    rows = deletion_test(model, spectrograms, maps, torch.Generator().manual_seed(0))

    assert [row["fraction"] for row in rows] == [0.1, 0.2, 0.3]
    assert all(np.isfinite(row["cam_logit_drop"]) for row in rows)


def test_shifted_maps_keep_each_map_values() -> None:
    maps = torch.rand(3, 80, 63)

    moved = _shifted_maps(maps, torch.Generator().manual_seed(0))

    assert moved.shape == maps.shape
    assert not torch.equal(moved, maps)
    for original, shifted in zip(maps, moved, strict=True):
        assert torch.equal(original.flatten().sort().values, shifted.flatten().sort().values)


def test_map_similarity_detects_identical_and_opposite_maps() -> None:
    maps = torch.rand(3, 80, 63)

    assert map_similarity(maps, maps) == pytest.approx(1.0)
    assert map_similarity(maps, -maps) == pytest.approx(-1.0)


def _fold_and_signal(channels: int) -> tuple[Fold, torch.Tensor, object]:
    config = replace(DUAL if channels == 2 else SINGLE, firing_power_window=3)
    signal = torch.randn(channels, 45_000)
    starts = np.arange(0, 12_800, 1280)
    fold = Fold(
        patient_id="chb01",
        seizure_index=0,
        seed=1,
        train_preictal=starts + 13_000,
        train_interictal=starts + 26_000,
        test_preictal=starts,
        test_seconds_to_onset=np.arange(100.0, 0.0, -10.0),
        test_interictal=starts + 13_000,
    )
    return fold, signal, config


@pytest.mark.parametrize("channels", [1, 2])
def test_channel_ablation_reports_each_input(channels: int) -> None:
    fold, signal, config = _fold_and_signal(channels)
    model = create_model(config).eval()
    explanation = explain_fold(model, fold, signal, config)

    rows = channel_ablation(model, fold, signal, explanation, config)

    assert explanation.preictal_maps.shape[0] == len(fold.test_preictal)
    expected = ["all channels"]
    if channels == 2:
        expected += ["FP1-F7 only", "T8-P8 only"]
    assert [row["input"] for row in rows] == expected
    assert all(0 <= row["segment_accuracy"] <= 1 for row in rows)


def test_sample_windows_picks_most_confident_of_each_outcome() -> None:
    probabilities = np.array([0.9, 0.6, 0.4, 0.05])

    assert sample_windows_by_outcome(probabilities, True) == {
        "preictal_correct": 0,
        "preictal_incorrect": 3,
    }
    assert sample_windows_by_outcome(probabilities, False) == {
        "interictal_correct": 3,
        "interictal_incorrect": 0,
    }


def test_explanation_plots_render() -> None:
    frequencies = bin_frequencies(SINGLE)
    band_values = dict.fromkeys(BANDS, 0.5)
    identity = {"patient": "chb01", "seizure": 0}

    plots = [
        plot_window_heatmap(np.zeros((80, 63)), np.zeros((80, 63)), frequencies, "title"),
        plot_band_importance(
            [
                {**identity, "class": "preictal", **band_values},
                {**identity, "class": "interictal", **band_values},
            ]
        ),
        plot_band_importance_by_time([{**identity, "minutes_before_onset": 2.5, **band_values}]),
        plot_deletion_test(
            [{**identity, "fraction": 0.1, "cam_logit_drop": 1.0, "random_logit_drop": 0.1}]
        ),
    ]

    assert all(plot.axes for plot in plots)
