import hashlib
from pathlib import Path

from neurofy.data.verification import parse_checksums, verify_files
from neurofy.utils.paths import find_directory_containing


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_parse_checksums_handles_special_file_names() -> None:
    checksums = parse_checksums("abc chb02/chb02_16+.edf\n\ndef RECORDS\n")

    assert checksums == {"chb02/chb02_16+.edf": "abc", "RECORDS": "def"}


def test_verify_reports_matches_mismatches_and_missing(tmp_path: Path) -> None:
    (tmp_path / "good.edf").write_bytes(b"good")
    (tmp_path / "bad.edf").write_bytes(b"tampered")
    checksums = {
        "good.edf": _sha(b"good"),
        "bad.edf": _sha(b"original"),
        "gone.edf": _sha(b"gone"),
    }

    report = verify_files(tmp_path, checksums)

    assert report.total == 3
    assert report.matched == 1
    assert report.mismatched == ["bad.edf"]
    assert report.missing == ["gone.edf"]
    assert not report.passed


def test_verify_passes_when_all_files_match(tmp_path: Path) -> None:
    (tmp_path / "good.edf").write_bytes(b"good")

    assert verify_files(tmp_path, {"good.edf": _sha(b"good")}).passed


def test_find_directory_prefers_shallowest_match(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "RECORDS").write_text("x")
    (tmp_path / "a" / "RECORDS").write_text("x")

    assert find_directory_containing("RECORDS", tmp_path) == tmp_path / "a"
