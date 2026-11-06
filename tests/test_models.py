import numpy as np
import pytest
import torch
from torch import nn

from neurofy.config import CONFIGS
from neurofy.models.mobilenet import SeizureNet, create_model
from neurofy.models.spectrogram import SpectrogramTransform

SAMPLES = 2560
RATE = 256
SINGLE = CONFIGS["fp1f7"]
DUAL = CONFIGS["fp1f7_t8p8"]


def _transform() -> SpectrogramTransform:
    return SpectrogramTransform(
        SINGLE.stft_frame, SINGLE.stft_hop, SINGLE.stft_fft, SINGLE.stft_bins
    )


def _table_count(module: nn.Module) -> int:
    """Count parameters plus BatchNorm running statistics, as Supp. Table 3 does."""
    buffers = sum(
        buffer.numel()
        for name, buffer in module.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    )
    return sum(parameter.numel() for parameter in module.parameters()) + buffers


@pytest.mark.parametrize("channels", [1, 2])
def test_spectrogram_output_shape(channels: int) -> None:
    output = _transform()(torch.zeros(3, channels, SAMPLES))

    assert output.shape == (3, channels, 80, 63)


def test_spectrogram_peaks_at_input_frequency() -> None:
    time = torch.arange(SAMPLES) / RATE
    sine = torch.sin(2 * torch.pi * 12.0 * time).reshape(1, 1, -1)

    spectrum = _transform()(sine)[0, 0].mean(dim=0)
    # Bin k holds k * 0.8 Hz, and index i holds bin i + 1.
    peak_hz = (spectrum.argmax().item() + 1) * 0.8

    assert peak_hz == pytest.approx(12.0, abs=0.8)


def test_spectrogram_frames_do_not_use_later_samples() -> None:
    windows = torch.randn(1, 1, SAMPLES)
    changed = windows.clone()
    changed[..., -32:] += 100.0

    original = _transform()(windows)
    updated = _transform()(changed)

    assert torch.equal(original[..., :78, :], updated[..., :78, :])
    assert not torch.equal(original[..., 79, :], updated[..., 79, :])


def test_layer_counts_match_supplementary_table() -> None:
    blocks = create_model(SINGLE).features

    assert [_table_count(block) for block in blocks] == [224, 896, 2816, 5312, 9728, 18816]


def test_total_parameter_counts() -> None:
    assert _table_count(create_model(SINGLE)) == 37_921
    assert _table_count(create_model(DUAL)) == 38_065
    assert sum(p.numel() for p in create_model(SINGLE).parameters()) == 36_449
    assert sum(p.numel() for p in create_model(DUAL).parameters()) == 36_593


def test_feature_shapes_match_supplementary_table() -> None:
    model = create_model(SINGLE)
    x = torch.zeros(2, 1, 80, 63)
    shapes = []
    for block in model.features:
        x = block(x)
        shapes.append(tuple(x.shape[1:]))

    assert shapes == [
        (16, 80, 63),
        (32, 80, 63),
        (64, 40, 32),
        (64, 40, 32),
        (128, 20, 16),
        (128, 20, 16),
    ]


@pytest.mark.parametrize("config", [SINGLE, DUAL])
def test_forward_returns_one_logit_per_window(config) -> None:
    logits = SeizureNet(config)(torch.randn(4, len(config.channels), SAMPLES))

    assert logits.shape == (4,)


def test_model_can_overfit_a_small_batch() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    time = np.arange(SAMPLES) / RATE
    low = np.stack([np.sin(2 * np.pi * 5 * time + rng.random()) for _ in range(4)])
    high = np.stack([np.sin(2 * np.pi * 30 * time + rng.random()) for _ in range(4)])
    windows = torch.tensor(np.concatenate([low, high]), dtype=torch.float32).unsqueeze(1) * 20
    labels = torch.tensor([0.0] * 4 + [1.0] * 4)
    model = create_model(SINGLE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(40):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(windows), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    predictions = (model(windows) > 0).float()
    assert torch.equal(predictions, labels)
