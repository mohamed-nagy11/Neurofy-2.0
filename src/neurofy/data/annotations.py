"""Parser for CHB-MIT summary files."""

import re
from dataclasses import dataclass
from pathlib import Path

_SECONDS_PER_DAY = 86_400

_CHANNEL_BLOCK = re.compile(r"Channels (?:in EDF|changed)")
_CHANNEL = re.compile(r"Channel \d+:\s*(\S+)")
_FILE_NAME = re.compile(r"File Name:\s*(\S+)")
_FILE_START = re.compile(r"File Start Time:\s*(\d+):(\d+):(\d+)")
_FILE_END = re.compile(r"File End Time:\s*(\d+):(\d+):(\d+)")
_SEIZURE_START = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)")
_SEIZURE_END = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)")


@dataclass(frozen=True)
class Seizure:
    """Seizure times in seconds from the start of its file."""

    onset: int
    offset: int


@dataclass(frozen=True)
class RecordingFile:
    """One EDF file; `start` and `end` are seconds from midnight of the case's first day."""

    name: str
    channels: tuple[str, ...]
    seizures: tuple[Seizure, ...]
    start: int | None
    end: int | None

    def has_channels(self, names: tuple[str, ...]) -> bool:
        return all(any(channel.upper() == name for channel in self.channels) for name in names)


@dataclass(frozen=True)
class Case:
    """All recordings of one CHB-MIT case, in summary order."""

    case_id: str
    files: tuple[RecordingFile, ...]

    @property
    def has_file_times(self) -> bool:
        return all(file.start is not None for file in self.files)

    @property
    def seizure_count(self) -> int:
        return sum(len(file.seizures) for file in self.files)


def _seconds(match: re.Match) -> int:
    hours, minutes, seconds = (int(group) for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _absolute_times(files: list[dict]) -> None:
    """Convert clock times to absolute seconds, adding a day when the clock goes backward."""
    day_offset = 0
    previous_end = None
    for file in files:
        if file["start"] is None or file["end"] is None:
            continue
        duration = (file["end"] - file["start"]) % _SECONDS_PER_DAY
        if previous_end is not None and file["start"] + day_offset < previous_end:
            day_offset += _SECONDS_PER_DAY
        file["start"] += day_offset
        file["end"] = file["start"] + duration
        previous_end = file["end"]


def parse_summary(text: str, case_id: str) -> Case:
    """Parse the contents of a `chbNN-summary.txt` file."""
    files: list[dict] = []
    channels: tuple[str, ...] = ()
    pending_channels: list[str] | None = None

    for line in text.splitlines():
        line = line.strip()
        if _CHANNEL_BLOCK.match(line):
            pending_channels = []
        elif match := _CHANNEL.match(line):
            if pending_channels is not None:
                pending_channels.append(match.group(1))
        elif match := _FILE_NAME.match(line):
            if pending_channels:
                channels = tuple(pending_channels)
            pending_channels = None
            files.append(
                {
                    "name": match.group(1),
                    "channels": channels,
                    "seizures": [],
                    "start": None,
                    "end": None,
                }
            )
        elif match := _FILE_START.match(line):
            files[-1]["start"] = _seconds(match)
        elif match := _FILE_END.match(line):
            files[-1]["end"] = _seconds(match)
        elif match := _SEIZURE_START.match(line):
            files[-1]["seizures"].append([int(match.group(1)), None])
        elif match := _SEIZURE_END.match(line):
            files[-1]["seizures"][-1][1] = int(match.group(1))

    _absolute_times(files)
    return Case(
        case_id=case_id,
        files=tuple(
            RecordingFile(
                name=file["name"],
                channels=file["channels"],
                seizures=tuple(Seizure(onset, offset) for onset, offset in file["seizures"]),
                start=file["start"],
                end=file["end"],
            )
            for file in files
        ),
    )


def load_cases(root: Path) -> list[Case]:
    """Parse every case summary under a CHB-MIT dataset root, ordered by case ID."""
    return [
        parse_summary(path.read_text(encoding="utf-8", errors="ignore"), path.parent.name)
        for path in sorted(root.glob("chb*/chb*-summary.txt"))
    ]
