"""Experiment configurations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings for one experiment; defaults follow Jang et al. (2026) where stated."""

    name: str = "fp1f7"
    experiment_name: str = "neurofy"
    channels: tuple[str, ...] = ("FP1-F7",)

    # Windows, in samples
    sampling_rate: int = 256
    window_samples: int = 2560
    hop_samples: int = 1280
    held_out_interictal_samples: int = 460_800

    # Spectrogram
    stft_frame: int = 64
    stft_hop: int = 32
    stft_fft: int = 320
    stft_bins: int = 63

    # Training
    batch_size: int = 16
    max_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    early_stopping_patience: int = 30
    early_stopping_start_epoch: int = 100

    # Post-processing
    firing_power_window: int = 119
    firing_power_threshold: float = 0.5

    # Runtime
    seed: int = 42
    device: str = "cuda"


CONFIGS: dict[str, ExperimentConfig] = {
    "fp1f7": ExperimentConfig(),
    "fp1f7_t8p8": ExperimentConfig(name="fp1f7_t8p8", channels=("FP1-F7", "T8-P8")),
}


def get_config(name: str) -> ExperimentConfig:
    """Return the configuration registered under `name`."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {sorted(CONFIGS)}")
    return CONFIGS[name]
