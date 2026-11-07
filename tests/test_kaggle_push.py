import importlib.util
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("kaggle_push", REPO_ROOT / "scripts/kaggle_push.py")
kaggle_push = importlib.util.module_from_spec(_spec)
sys.modules["kaggle_push"] = kaggle_push
_spec.loader.exec_module(kaggle_push)

COMMIT = "a" * 40


def _sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_load_env_ignores_comments_and_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nKAGGLE_API_TOKEN="value"\nOTHER = x\n')

    assert kaggle_push.load_env(env_file) == {"KAGGLE_API_TOKEN": "value", "OTHER": "x"}
    assert kaggle_push.load_env(tmp_path / "missing") == {}


def test_notebooks_stay_short_and_readable() -> None:
    for spec in kaggle_push.KERNELS.values():
        notebook = kaggle_push.build_notebook(spec, COMMIT)
        lines = "".join(_sources(notebook)).splitlines()

        assert notebook["cells"][0]["cell_type"] == "markdown"
        assert max(len(line) for line in lines) < 140
        assert len(lines) < 120


def test_notebook_clones_the_exact_commit_and_installs_it() -> None:
    notebook = kaggle_push.build_notebook(kaggle_push.KERNELS["verify"], COMMIT)
    source = "".join(_sources(notebook))

    assert f'run("git clone --quiet {kaggle_push.REPO_URL}.git' in source
    assert f"checkout --quiet {COMMIT}" in source
    assert f'run("pip install --quiet {kaggle_push.CODE_DIR}")' in source
    assert f"%env NEUROFY_GIT_COMMIT={COMMIT}" in source
    assert f"tree/{COMMIT}" in source


def test_each_step_has_a_title_and_extra_args_go_to_the_first() -> None:
    spec = kaggle_push.KERNELS["analysis"]

    sources = _sources(kaggle_push.build_notebook(spec, COMMIT, "--example 1"))
    compare = next(cell for cell in sources if "compare_models" in cell)
    explain = next(cell for cell in sources if "explain.py" in cell)

    assert any(cell.startswith("## Compare the models") for cell in sources)
    assert any(cell.startswith("## Explain Model 1") for cell in sources)
    assert "--example 1" in compare
    assert "--example 1" not in explain


def test_note_appears_in_the_header() -> None:
    spec = kaggle_push.KERNELS["verify"]

    notebook = kaggle_push.build_notebook(spec, COMMIT, note="**Result:** passed.")

    assert "**Result:** passed." in _sources(notebook)[0]


def test_notebooks_avoid_cell_types_that_hide_live_logs() -> None:
    for spec in kaggle_push.KERNELS.values():
        source = "".join(_sources(kaggle_push.build_notebook(spec, COMMIT)))

        assert "%%bash" not in source
        assert "
!" not in source


def test_code_cells_compile() -> None:
    for spec in kaggle_push.KERNELS.values():
        for cell in kaggle_push.build_notebook(spec, COMMIT)["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and not source.startswith("%"):
                compile(source, "cell", "exec")


def test_multiline_commands_parse_to_the_original_arguments() -> None:
    spec = kaggle_push.KERNELS["train-fp1f7"]
    namespace: dict = {}

    def capture(command: str) -> None:
        namespace["command"] = command

    cell = _sources(kaggle_push.build_notebook(spec, COMMIT))[-1]
    exec(cell, {"run": capture})

    assert shlex.split(namespace["command"]) == shlex.split(spec.steps[0].command)
    assert "
  --config fp1f7" in namespace["command"]


def test_run_helper_streams_output_and_raises_on_failure(capsys) -> None:
    namespace: dict = {}
    exec(kaggle_push.RUN_HELPER, namespace)

    namespace["run"](f'"{sys.executable}" -c "print(123)"')
    assert "123" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="Command failed"):
        namespace["run"](f'"{sys.executable}" -c "import sys; sys.exit(3)"')


def test_metadata_is_private_and_selects_t4_for_gpu_kernels() -> None:
    cpu = kaggle_push.build_metadata(kaggle_push.KERNELS["verify"], "x.ipynb")
    gpu = kaggle_push.build_metadata(kaggle_push.KERNELS["train-fp1f7"], "x.ipynb")

    assert cpu["is_private"] == "true"
    assert cpu["enable_gpu"] == "false"
    assert "machine_shape" not in cpu
    assert gpu["is_private"] == "true"
    assert gpu["machine_shape"] == "NvidiaTeslaT4"
    assert kaggle_push.PREPROCESSED_DATASET in gpu["dataset_sources"]


def test_tracked_files_exclude_secrets_and_plans() -> None:
    tracked = kaggle_push.git("ls-files").decode().splitlines()

    assert ".env" not in tracked
    assert not any(path.startswith(("plans/", "reaserch papers/")) for path in tracked)


def test_training_kernels_run_independently_with_two_workers_per_gpu() -> None:
    first = kaggle_push.KERNELS["train-fp1f7"]
    second = kaggle_push.KERNELS["train-fp1f7-t8p8"]

    for spec in (first, second):
        assert spec.gpu
        assert not spec.kernel_sources
        assert "--workers-per-gpu 2" in spec.steps[0].command


def test_analysis_attaches_both_training_kernels() -> None:
    spec = kaggle_push.KERNELS["analysis"]
    commands = " ".join(step.command for step in spec.steps)

    assert f"{kaggle_push.KAGGLE_USER}/{kaggle_push.MODEL_1_KERNEL}" in spec.kernel_sources
    assert f"{kaggle_push.KAGGLE_USER}/{kaggle_push.MODEL_2_KERNEL}" in spec.kernel_sources
    assert f"--first-results {kaggle_push.MODEL_1_KERNEL}" in commands
    assert f"--second-results {kaggle_push.MODEL_2_KERNEL}" in commands
    assert f"--restore-from {kaggle_push.MODEL_1_KERNEL}" in commands
    assert f"--restore-from {kaggle_push.MODEL_2_KERNEL}" in commands
