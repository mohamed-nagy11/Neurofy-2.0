"""Locating data attached to a Kaggle kernel."""

from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")


def find_directory_containing(file_name: str, search_root: Path = KAGGLE_INPUT) -> Path:
    """Return the shallowest directory under `search_root` that holds `file_name`."""
    matches = sorted(search_root.rglob(file_name), key=lambda path: len(path.parts))
    if not matches:
        raise FileNotFoundError(f"No {file_name} found under {search_root}")
    return matches[0].parent
