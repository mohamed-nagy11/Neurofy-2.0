import torch

from neurofy.config import BaselineConfig
from neurofy.data.dataset import get_dataloader


def test_splits_contain_different_data(config: BaselineConfig) -> None:
    features = {
        split: get_dataloader(config, split).dataset.features for split in ("train", "val", "test")
    }

    assert not torch.equal(features["train"], features["val"])
    assert not torch.equal(features["val"], features["test"])


def test_dataset_creation_does_not_change_global_rng(config: BaselineConfig) -> None:
    torch.manual_seed(0)
    expected = torch.rand(1)

    torch.manual_seed(0)
    get_dataloader(config, "train")

    assert torch.equal(torch.rand(1), expected)
