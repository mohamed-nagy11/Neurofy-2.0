"""Lightweight MobileNet from Jang et al. (2026), Supplementary Table 3."""

import torch
from torch import nn

from neurofy.config import ExperimentConfig
from neurofy.models.spectrogram import SpectrogramTransform


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise then pointwise convolution, each with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SeizureNet(nn.Module):
    """Predict one preictal logit per raw EEG window."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.spectrogram = SpectrogramTransform(
            config.stft_frame,
            config.stft_hop,
            config.stft_fft,
            config.stft_bins,
        )
        self.features = nn.Sequential(
            _conv_block(len(config.channels), 16),
            DepthwiseSeparableBlock(16, 32, stride=1),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64, stride=1),
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 1)

    def forward_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Classify precomputed spectrograms."""
        features = self.pool(self.features(spectrogram)).flatten(1)
        return self.classifier(features).squeeze(-1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.forward_spectrogram(self.spectrogram(windows))


def create_model(config: ExperimentConfig) -> SeizureNet:
    """Create the model described by `config`."""
    return SeizureNet(config)
