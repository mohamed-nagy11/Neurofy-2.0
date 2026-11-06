import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("kaggle_push", REPO_ROOT / "scripts/kaggle_push.py")
kaggle_push = importlib.util.module_from_spec(_spec)
sys.modules["kaggle_push"] = kaggle_push
_spec.loader.exec_module(kaggle_push)


def _source(notebook: dict) -> str:
    return "".join("".join(cell["source"]) for cell in notebook["cells"])


def test_load_env_ignores_comments_and_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nKAGGLE_API_TOKEN="value"\nOTHER = x\n')

    assert kaggle_push.load_env(env_file) == {"KAGGLE_API_TOKEN": "value", "OTHER": "x"}
    assert kaggle_push.load_env(tmp_path / "missing") == {}


def test_notebooks_stay_short_and_readable() -> None:
    for spec in kaggle_push.KERNELS.values():
        notebook = kaggle_push.build_notebook(spec, "a" * 40)
        lines = _source(notebook).splitlines()

        assert notebook["cells"][0]["cell_type"] == "markdown"
        assert max(len(line) for line in lines) < 120
        assert len(lines) < 100


def test_notebook_checks_commit_and_runs_each_command_in_its_own_cell() -> None:
    spec = kaggle_push.KERNELS["train-fp1f7-t8p8"]

    notebook = kaggle_push.build_notebook(spec, "abc123", "--workers-per-gpu 2")
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    source = "".join(sources)

    assert 'assert commit == "abc123"' in source
    assert "pip install -q mlflow" in source
    assert sum("run_experiment.py" in cell for cell in sources) == 1
    assert sum("compare_models.py" in cell for cell in sources) == 1
    assert "--workers-per-gpu 2" in next(cell for cell in sources if "run_experiment" in cell)
    assert "--workers-per-gpu" not in next(cell for cell in sources if "compare_models" in cell)


def test_command_cells_stop_on_failure() -> None:
    spec = kaggle_push.KERNELS["verify"]

    notebook = kaggle_push.build_notebook(spec, "abc")
    commands = [cell for cell in notebook["cells"] if "%%bash\n" in cell["source"]]

    assert commands
    assert all(cell["source"][1] == "set -e\n" for cell in commands)


def test_setup_cell_compiles_and_notebook_serializes() -> None:
    notebook = kaggle_push.build_notebook(kaggle_push.KERNELS["verify"], "abc")

    compile("".join(notebook["cells"][1]["source"]), "setup", "exec")
    json.dumps(notebook)


def test_metadata_is_private_attaches_code_and_selects_t4() -> None:
    cpu = kaggle_push.build_metadata(kaggle_push.KERNELS["verify"], "x.ipynb")
    gpu = kaggle_push.build_metadata(kaggle_push.KERNELS["train-fp1f7"], "x.ipynb")

    assert cpu["is_private"] == "true"
    assert cpu["enable_gpu"] == "false"
    assert "machine_shape" not in cpu
    assert kaggle_push.CODE_DATASET in cpu["dataset_sources"]
    assert gpu["is_private"] == "true"
    assert gpu["machine_shape"] == "NvidiaTeslaT4"
    assert kaggle_push.PREPROCESSED_DATASET in gpu["dataset_sources"]


def test_code_dataset_does_not_overwrite_existing_dataset() -> None:
    assert kaggle_push.CODE_DATASET_SLUG != "neurofy-code"


def test_publish_command_never_makes_the_dataset_public(tmp_path: Path) -> None:
    create = kaggle_push.dataset_publish_command(False, tmp_path, "abcdef123")
    version = kaggle_push.dataset_publish_command(True, tmp_path, "abcdef123")

    assert create[:3] == ["kaggle", "datasets", "create"]
    assert version[:3] == ["kaggle", "datasets", "version"]
    assert "Commit abcdef1" in version
    for command in (create, version):
        assert "-u" not in command
        assert "--public" not in command


def test_code_package_holds_only_code_and_the_commit(tmp_path: Path) -> None:
    commit = kaggle_push.git("rev-parse", "HEAD").decode().strip()
    output = tmp_path / "code"

    kaggle_push.build_code_package(commit, output)

    assert (output / kaggle_push.COMMIT_FILE).read_text() == commit
    assert (output / "src" / "neurofy" / "config.py").is_file()
    assert (output / "scripts" / "run_experiment.py").is_file()
    assert json.loads((output / "dataset-metadata.json").read_text())["id"] == (
        kaggle_push.CODE_DATASET
    )
    names = {path.name for path in output.rglob("*")}
    assert ".env" not in names
    assert not any((output / folder).exists() for folder in ("plans", "reaserch papers", "tests"))


def test_training_kernels_use_gpu_and_keep_earlier_results() -> None:
    first = kaggle_push.KERNELS["train-fp1f7"]
    second = kaggle_push.KERNELS["train-fp1f7-t8p8"]

    assert first.gpu and second.gpu
    assert "mlflow" in first.packages
    assert f"{kaggle_push.KAGGLE_USER}/neurofy-train-fp1f7" in second.kernel_sources
