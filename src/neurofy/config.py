"""Experiment configurations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    """Baseline experiment."""

    name: str = "baseline"
    experiment_name: str = "neurofy"

    # Data
    num_samples: int = 1000
    input_dim: int = 10
    num_workers: int = 0

    # Training
    batch_size: int = 16
    max_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    early_stopping_patience: int = 30
    early_stopping_start_epoch: int = 100

    # Runtime
    seed: int = 42
    device: str = "cuda"


@dataclass(frozen=True)
class Ablation1Config(BaselineConfig):
    """Lower learning rate."""

    name: str = "ablation1"
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class Ablation2Config(BaselineConfig):
    """Larger batch size."""

    name: str = "ablation2"
    batch_size: int = 64


@dataclass(frozen=True)
class Ablation3Config(BaselineConfig):
    """No weight decay."""

    name: str = "ablation3"
    weight_decay: float = 0.0


CONFIGS: dict[str, type[BaselineConfig]] = {
    "baseline": BaselineConfig,
    "ablation1": Ablation1Config,
    "ablation2": Ablation2Config,
    "ablation3": Ablation3Config,
}


def get_config(name: str) -> BaselineConfig:
    """Return the configuration registered under `name`."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {sorted(CONFIGS)}")
    return CONFIGS[name]()
