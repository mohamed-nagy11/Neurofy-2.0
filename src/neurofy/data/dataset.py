"""Datasets and data loaders."""

import torch
from torch.utils.data import DataLoader, Dataset

from neurofy.config import BaselineConfig
from neurofy.utils.runtime import seed_worker

_SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}


class SyntheticDataset(Dataset):
    """Placeholder binary dataset labeled by the sign of the feature sum."""

    def __init__(self, num_samples: int, input_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, input_dim, generator=generator)
        self.labels = (self.features.sum(dim=1) > 0).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def get_dataloader(config: BaselineConfig, split: str) -> DataLoader:
    """Create the data loader for the `train`, `val` or `test` split."""
    seed = config.seed + _SPLIT_SEED_OFFSETS[split]
    dataset = SyntheticDataset(config.num_samples, config.input_dim, seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=split == "train",
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
