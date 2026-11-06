"""Preprocessed patient data and window batches."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class PatientRecord:
    """A patient's stored signal and labeled ranges, in samples."""

    patient_id: str
    signal_path: Path
    stored_channels: tuple[str, ...]
    sampling_rate: int
    seizures: tuple[tuple[tuple[int, int, int], ...], ...]
    interictal: tuple[tuple[int, int], ...]


def load_records(data_dir: Path) -> dict[str, PatientRecord]:
    """Read the preprocessing manifest into patient records."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        patient_id: PatientRecord(
            patient_id=patient_id,
            signal_path=data_dir / entry["signal"],
            stored_channels=tuple(manifest["channels"]),
            sampling_rate=manifest["sampling_rate"],
            seizures=tuple(
                tuple(
                    (item["start"], item["end"], item["seconds_to_onset"])
                    for item in seizure["preictal"]
                )
                for seizure in entry["seizures"]
            ),
            interictal=tuple((item["start"], item["end"]) for item in entry["interictal"]),
        )
        for patient_id, entry in manifest["patients"].items()
    }


def load_signal(
    record: PatientRecord,
    channels: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """Load selected channels into memory on `device`, shaped `(channels, samples)`."""
    indices = [record.stored_channels.index(channel) for channel in channels]
    signal = np.load(record.signal_path, mmap_mode="r")[indices]
    return torch.from_numpy(np.array(signal, dtype=np.float32)).to(device)


class WindowLoader:
    """Yield batches of windows sliced from an in-memory signal, in the given order."""

    def __init__(
        self,
        signal: torch.Tensor,
        starts: np.ndarray,
        labels: np.ndarray,
        window_samples: int,
        batch_size: int,
    ) -> None:
        self.signal = signal
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=signal.device)
        self.labels = torch.as_tensor(labels, dtype=torch.float32, device=signal.device)
        self.offsets = torch.arange(window_samples, device=signal.device)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.starts) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for first in range(0, len(self.starts), self.batch_size):
            last = first + self.batch_size
            indices = self.starts[first:last, None] + self.offsets
            yield self.signal[:, indices].permute(1, 0, 2), self.labels[first:last]
