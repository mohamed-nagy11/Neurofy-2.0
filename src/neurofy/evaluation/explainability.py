"""Grad-CAM, reliability checks and channel ablation."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from neurofy.config import ExperimentConfig
from neurofy.data.folds import Fold
from neurofy.evaluation.firing_power import evaluate_fold
from neurofy.models.mobilenet import SeizureNet

BANDS = {
    "delta": (0.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 51.0),
}
DELETION_FRACTIONS = (0.1, 0.2, 0.3)

_BATCH_SIZE = 64
_NEUTRAL_WINDOWS = 256


def bin_frequencies(config: ExperimentConfig) -> np.ndarray:
    """Return the frequency in Hz of each kept spectrogram bin."""
    return (np.arange(config.stft_bins) + 1) * config.sampling_rate / config.stft_fft


def band_importance(maps: np.ndarray, frequencies: np.ndarray) -> dict[str, float]:
    """Average maps shaped `(..., frames, bins)` within each frequency band."""
    by_bin = maps.reshape(-1, maps.shape[-2], maps.shape[-1]).mean(axis=(0, 1))
    return {
        band: float(by_bin[(frequencies >= low) & (frequencies < high)].mean())
        for band, (low, high) in BANDS.items()
    }


def grad_cam(model: SeizureNet, spectrograms: torch.Tensor) -> torch.Tensor:
    """Compute normalized Grad-CAM maps shaped `(batch, frames, bins)`."""
    captured = {}

    def capture(_module, _inputs, output: torch.Tensor) -> None:
        output.retain_grad()
        captured["activations"] = output

    handle = model.features[-1].register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model.forward_spectrogram(spectrograms.detach())
            model.zero_grad(set_to_none=True)
            logits.sum().backward()
    finally:
        handle.remove()

    activations = captured["activations"]
    weights = activations.grad.mean(dim=(2, 3), keepdim=True)
    maps = F.relu((weights * activations).sum(dim=1, keepdim=True))
    maps = F.interpolate(maps, size=spectrograms.shape[-2:], mode="bilinear", align_corners=False)
    maps = maps.squeeze(1).detach()
    peak = maps.flatten(1).max(dim=1).values.clamp_min(1e-8)
    return maps / peak[:, None, None]


def _slice_windows(signal: torch.Tensor, starts: np.ndarray, window: int) -> torch.Tensor:
    offsets = torch.arange(window, device=signal.device)
    positions = torch.as_tensor(starts, dtype=torch.long, device=signal.device)
    return signal[:, positions[:, None] + offsets].permute(1, 0, 2)


def _spectrograms(
    model: SeizureNet,
    signal: torch.Tensor,
    starts: np.ndarray,
    window: int,
) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            [
                model.spectrogram(_slice_windows(signal, batch, window))
                for batch in np.array_split(starts, max(1, len(starts) // _BATCH_SIZE))
            ]
        )


def _in_batches(model: SeizureNet, spectrograms: torch.Tensor, operation) -> torch.Tensor:
    return torch.cat([operation(model, batch) for batch in spectrograms.split(_BATCH_SIZE)])


def _logits(model: SeizureNet, spectrograms: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model.forward_spectrogram(spectrograms)


def _masked(
    spectrograms: torch.Tensor,
    scores: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Replace the highest-scoring cells of each window with its channel means."""
    flat = scores.flatten(1)
    count = max(1, int(flat.shape[1] * fraction))
    threshold = flat.topk(count, dim=1).values[:, -1, None, None]
    mask = (scores >= threshold).unsqueeze(1)
    fill = spectrograms.mean(dim=(2, 3), keepdim=True)
    return torch.where(mask, fill, spectrograms)


def _shifted_maps(maps: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Move each map to a random position, keeping the shape of its highlighted regions."""
    frames, bins = maps.shape[1:]
    shifted = []
    for single in maps:
        offsets = (
            int(torch.randint(frames, (1,), generator=generator, device=generator.device)),
            int(torch.randint(bins, (1,), generator=generator, device=generator.device)),
        )
        shifted.append(torch.roll(single, offsets, dims=(0, 1)))
    return torch.stack(shifted)


def deletion_test(
    model: SeizureNet,
    spectrograms: torch.Tensor,
    maps: torch.Tensor,
    generator: torch.Generator,
) -> list[dict[str, float]]:
    """Compare the logit drop from masking CAM regions with masking those regions moved elsewhere.

    Moved regions keep the same size and shape, so the baseline differs only in location.
    """
    baseline = _in_batches(model, spectrograms, _logits)
    moved = _shifted_maps(maps, generator)
    rows = []
    for fraction in DELETION_FRACTIONS:
        cam_logits = _in_batches(model, _masked(spectrograms, maps, fraction), _logits)
        moved_logits = _in_batches(model, _masked(spectrograms, moved, fraction), _logits)
        rows.append(
            {
                "fraction": fraction,
                "cam_logit_drop": float((baseline - cam_logits).mean()),
                "random_logit_drop": float((baseline - moved_logits).mean()),
            }
        )
    return rows


def map_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the mean per-window Pearson correlation between two sets of maps."""
    left = first.flatten(1)
    right = second.flatten(1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    denominator = (left.norm(dim=1) * right.norm(dim=1)).clamp_min(1e-8)
    return float(((left * right).sum(dim=1) / denominator).mean())


@dataclass(frozen=True)
class FoldExplanation:
    """Explanation data for one fold's held-out windows."""

    preictal_spectrograms: torch.Tensor
    preictal_maps: torch.Tensor
    interictal_spectrograms: torch.Tensor
    interictal_maps: torch.Tensor


def explain_fold(
    model: SeizureNet,
    fold: Fold,
    signal: torch.Tensor,
    config: ExperimentConfig,
) -> FoldExplanation:
    """Compute spectrograms and Grad-CAM maps for a fold's held-out windows."""
    preictal = _spectrograms(model, signal, fold.test_preictal, config.window_samples)
    interictal = _spectrograms(model, signal, fold.test_interictal, config.window_samples)
    return FoldExplanation(
        preictal,
        _in_batches(model, preictal, grad_cam),
        interictal,
        _in_batches(model, interictal, grad_cam),
    )


def band_importance_by_time(
    maps: np.ndarray,
    seconds_to_onset: np.ndarray,
    frequencies: np.ndarray,
    bin_minutes: int = 5,
) -> list[dict]:
    """Average band importance in bins of time before seizure onset."""
    minutes = seconds_to_onset / 60
    rows = []
    for low in range(0, int(np.ceil(minutes.max())), bin_minutes):
        selected = (minutes >= low) & (minutes < low + bin_minutes)
        if selected.any():
            rows.append(
                {
                    "minutes_before_onset": low + bin_minutes / 2,
                    **band_importance(maps[selected], frequencies),
                }
            )
    return rows


def channel_ablation(
    model: SeizureNet,
    fold: Fold,
    signal: torch.Tensor,
    explanation: FoldExplanation,
    config: ExperimentConfig,
) -> list[dict]:
    """Measure performance when each channel is replaced by its training mean spectrogram."""
    rng = np.random.default_rng(fold.seed)
    pool = np.concatenate([fold.train_preictal, fold.train_interictal])
    sample = rng.choice(pool, size=min(_NEUTRAL_WINDOWS, len(pool)), replace=False)
    neutral = _spectrograms(model, signal, sample, config.window_samples).mean(dim=0)

    def evaluate(kept_channel: int | None, name: str) -> dict:
        def ablate(spectrograms: torch.Tensor) -> torch.Tensor:
            if kept_channel is None:
                return spectrograms
            output = spectrograms.clone()
            for channel in range(output.shape[1]):
                if channel != kept_channel:
                    output[:, channel] = neutral[channel]
            return output

        preictal = _in_batches(model, ablate(explanation.preictal_spectrograms), _logits)
        interictal = _in_batches(model, ablate(explanation.interictal_spectrograms), _logits)
        outcome = evaluate_fold(
            torch.sigmoid(preictal).cpu().numpy(),
            torch.sigmoid(interictal).cpu().numpy(),
            fold.test_seconds_to_onset,
            config.firing_power_window,
            config.firing_power_threshold,
            config.window_samples / config.sampling_rate,
        )
        return {
            "input": name,
            "segment_accuracy": outcome.segment.metrics["accuracy"],
            "segment_fpr": outcome.segment.metrics["fpr"],
            "sensitivity": outcome.events["sensitivity"],
            "interictal_alarm": outcome.events["interictal_alarm"],
        }

    rows = [evaluate(None, "all channels")]
    if len(config.channels) > 1:
        rows.extend(
            evaluate(index, f"{channel} only") for index, channel in enumerate(config.channels)
        )
    return rows


def sample_windows_by_outcome(
    probabilities: np.ndarray,
    is_preictal: bool,
) -> dict[str, int]:
    """Pick the most confident correct and incorrect window, for example heatmaps."""
    positive = probabilities >= 0.5
    correct = positive if is_preictal else ~positive
    label = "preictal" if is_preictal else "interictal"
    confidence = np.abs(probabilities - 0.5)

    samples = {}
    for name, selected in (("correct", correct), ("incorrect", ~correct)):
        if selected.any():
            candidates = np.flatnonzero(selected)
            samples[f"{label}_{name}"] = int(candidates[np.argmax(confidence[candidates])])
    return samples
