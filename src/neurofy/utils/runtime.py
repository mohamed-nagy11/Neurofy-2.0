"""Seeding and device selection."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed all random number generators and make CUDA operations deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and `random` inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(name: str) -> torch.device:
    """Resolve a device name, falling back to CPU when CUDA is unavailable."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(name)
