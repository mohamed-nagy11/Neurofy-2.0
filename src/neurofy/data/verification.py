"""Dataset integrity checks against published checksums."""

import hashlib
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PHYSIONET_CHECKSUMS_URL = "https://physionet.org/files/chbmit/1.0.0/SHA256SUMS.txt"

_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of comparing a dataset with its published checksums."""

    total: int
    matched: int
    mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.matched == self.total


def parse_checksums(text: str) -> dict[str, str]:
    """Parse `sha256sum` output into a mapping of relative path to hash."""
    checksums = {}
    for line in text.splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            checksums[path.strip().lstrip("*")] = digest
    return checksums


def fetch_published_checksums(url: str = PHYSIONET_CHECKSUMS_URL) -> dict[str, str]:
    """Download the checksum list from the original publisher."""
    with urllib.request.urlopen(url, timeout=120) as response:
        return parse_checksums(response.read().decode("utf-8"))


def hash_file(path: Path) -> str:
    """Compute a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, checksums: dict[str, str], workers: int = 4) -> VerificationReport:
    """Hash every listed file under `root` and compare it with its published digest."""
    missing = sorted(path for path in checksums if not (root / path).is_file())
    present = [path for path in checksums if path not in set(missing)]

    def check(path: str) -> tuple[str, bool]:
        return path, hash_file(root / path) == checksums[path]

    mismatched = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (path, is_match) in enumerate(executor.map(check, present), start=1):
            if not is_match:
                mismatched.append(path)
                logger.error("Checksum mismatch: %s", path)
            if index % 100 == 0:
                logger.info("Hashed %d of %d files", index, len(present))

    return VerificationReport(
        total=len(checksums),
        matched=len(present) - len(mismatched),
        mismatched=sorted(mismatched),
        missing=missing,
    )
