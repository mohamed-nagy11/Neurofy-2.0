import base64
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("kaggle_push", REPO_ROOT / "scripts/kaggle_push.py")
kaggle_push = importlib.util.module_from_spec(_spec)
sys.modules["kaggle_push"] = kaggle_push
_spec.loader.exec_module(kaggle_push)


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _source(notebook: dict) -> str:
    return "".join("".join(cell["source"]) for cell in notebook["cells"])


def test_load_env_ignores_comments_and_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nKAGGLE_API_TOKEN="value"\nOTHER = x\n')

    assert kaggle_push.load_env(env_file) == {"KAGGLE_API_TOKEN": "value", "OTHER": "x"}
    assert kaggle_push.load_env(tmp_path / "missing") == {}


def test_notebook_embeds_archive_commit_and_command() -> None:
    spec = kaggle_push.KERNELS["verify"]
    archive = _archive({"scripts/verify_dataset.py": b"content"})

    notebook = kaggle_push.build_notebook(archive, "abc123", spec, "python {code}/run.py")
    source = _source(notebook)

    assert base64.b64encode(archive).decode() in source
    assert "abc123" in source
    assert "python /kaggle/working/.neurofy_code/run.py" in source


def test_notebook_bootstrap_compiles() -> None:
    spec = kaggle_push.KERNELS["verify"]
    notebook = kaggle_push.build_notebook(_archive({"x": b"y"}), "abc", spec, "python x")

    for cell in notebook["cells"]:
        compile("".join(cell["source"]), "cell", "exec")
    json.dumps(notebook)


def test_metadata_is_private_and_selects_t4_for_gpu_kernels() -> None:
    cpu = kaggle_push.build_metadata(kaggle_push.KERNELS["verify"], "x.ipynb")
    gpu_spec = kaggle_push.KernelSpec(slug="s", title="T", command="c", gpu=True)
    gpu = kaggle_push.build_metadata(gpu_spec, "x.ipynb")

    assert cpu["is_private"] == "true"
    assert cpu["enable_gpu"] == "false"
    assert "machine_shape" not in cpu
    assert gpu["is_private"] == "true"
    assert gpu["machine_shape"] == "NvidiaTeslaT4"
    assert gpu["enable_gpu"] == "true"


def test_tracked_files_exclude_secrets_and_plans() -> None:
    tracked = kaggle_push.git("ls-files").decode().splitlines()

    assert ".env" not in tracked
    assert not any(path.startswith(("plans/", "reaserch papers/")) for path in tracked)
