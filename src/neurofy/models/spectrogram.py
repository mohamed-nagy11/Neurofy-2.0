"""Spectrogram input transform."""

import torch
import torch.nn.functional as F
from torch import nn


class SpectrogramTransform(nn.Module):
    """Convert raw windows to log-magnitude spectrograms.

    Frames are taken directly rather than with `torch.stft`, which frames by the FFT
    length when `center=False`. Right-padding yields the last frame without using
    future samples.
    """

    def __init__(self, frame: int, hop: int, fft: int, bins: int) -> None:
        super().__init__()
        self.frame = frame
        self.hop = hop
        self.fft = fft
        self.bins = bins
        self.register_buffer("window", torch.hann_window(frame), persistent=False)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Map `(batch, channels, samples)` to `(batch, channels, frames, bins)`."""
        padded = F.pad(windows, (0, self.hop))
        frames = padded.unfold(-1, self.frame, self.hop)
        spectrum = torch.fft.rfft(frames * self.window, n=self.fft)
        magnitude = spectrum.abs()[..., 1 : self.bins + 1]
        return torch.log1p(magnitude)
