"""Application logging setup."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(processName)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_package_logger = logging.getLogger("neurofy")


def configure_logging(level: int = logging.INFO) -> None:
    """Send package logs at `level` to the console; files may capture more detail."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_FORMATTER)
    _package_logger.handlers.clear()
    _package_logger.addHandler(handler)
    _package_logger.setLevel(logging.DEBUG)
    _package_logger.propagate = False


@contextmanager
def log_to_file(path: Path) -> Iterator[None]:
    """Mirror package logs, including debug detail, to `path` for the duration of the context."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_FORMATTER)
    previous_level = _package_logger.level
    _package_logger.addHandler(handler)
    _package_logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        _package_logger.removeHandler(handler)
        _package_logger.setLevel(previous_level)
        handler.close()
