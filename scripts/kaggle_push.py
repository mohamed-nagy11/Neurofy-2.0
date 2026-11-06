"""Publish the current commit as a private code dataset and push a Kaggle notebook that runs it."""

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from neurofy.utils.logger import configure_logging

logger = logging.getLogger("neurofy.scripts.kaggle_push")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = REPO_ROOT / "build" / "kaggle"
KAGGLE_USER = "mohamed11nagy11"
CHBMIT_DATASET = "abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric"
PREPROCESSED_DATASET = f"{KAGGLE_USER}/neurofy-chbmit-preprocessed"
CODE_DATASET_SLUG = "neurofy-2-code"
CODE_DATASET = f"{KAGGLE_USER}/{CODE_DATASET_SLUG}"
COMMIT_FILE = "NEUROFY_COMMIT"
CODE_PATHS = ("src", "scripts", "pyproject.toml", "README.md")
T4_X2 = "NvidiaTeslaT4"
RESULTS = "--tracking-dir /tmp/mlflow --archive /kaggle/working/mlflow_results.tar.gz"
EXPERIMENT = f'python "$CODE/scripts/run_experiment.py" {RESULTS}'
COMPARE = f'python "$CODE/scripts/compare_models.py" {RESULTS}'
EXPLAIN = f'python "$CODE/scripts/explain.py" {RESULTS}'
SMOKE_PATIENTS = "chb04 chb02 chb16 chb17 chb19 chb22"

_DATASET_READY_TIMEOUT_SECONDS = 600

_SETUP_CELL = """\
import os
from pathlib import Path

code_dir = next(Path("/kaggle/input").rglob("{commit_file}")).parent
commit = (code_dir / "{commit_file}").read_text().strip()
assert commit == "{commit}", f"Code dataset holds {{commit}}; wait for its new version"

os.environ["CODE"] = str(code_dir)
os.environ["PYTHONPATH"] = str(code_dir / "src")
os.environ["NEUROFY_GIT_COMMIT"] = commit
"""


@dataclass(frozen=True)
class KernelSpec:
    """Settings for one Kaggle kernel."""

    slug: str
    title: str
    description: str
    commands: tuple[str, ...]
    packages: tuple[str, ...] = ()
    gpu: bool = False
    dataset_sources: tuple[str, ...] = ()
    kernel_sources: tuple[str, ...] = ()


def _benchmark(workers: int) -> str:
    return (
        f"{EXPERIMENT} --config fp1f7 --experiment neurofy-smoke-w{workers} "
        f"--patients {SMOKE_PATIENTS} --max-epochs 30 --workers-per-gpu {workers}"
    )


KERNELS = {
    "verify": KernelSpec(
        slug="neurofy-verify",
        title="Neurofy Verify",
        description="Verify the CHB-MIT copy against the checksums published by PhysioNet.",
        commands=('python "$CODE/scripts/verify_dataset.py"',),
        dataset_sources=(CHBMIT_DATASET,),
    ),
    "preprocess": KernelSpec(
        slug="neurofy-preprocess",
        title="Neurofy Preprocess",
        description="Select the cohort, filter its signals and store them with a manifest.",
        commands=(
            'python "$CODE/scripts/preprocess.py" --output /kaggle/working/chbmit_preprocessed',
        ),
        packages=("pyedflib",),
        dataset_sources=(CHBMIT_DATASET,),
    ),
    "smoke": KernelSpec(
        slug="neurofy-smoke",
        title="Neurofy Smoke",
        description="Benchmark workers per GPU, then check resuming, comparison and Grad-CAM.",
        commands=(
            _benchmark(1),
            _benchmark(2),
            _benchmark(4),
            _benchmark(1),
            f"{EXPERIMENT} --config fp1f7_t8p8 --experiment neurofy-smoke-w1 "
            f"--patients {SMOKE_PATIENTS} --max-epochs 3 --workers-per-gpu 2",
            f"{COMPARE} --experiment neurofy-smoke-w1",
            f"{EXPLAIN} --config fp1f7_t8p8 --experiment neurofy-smoke-w1",
        ),
        packages=("mlflow",),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
    ),
    "train-fp1f7": KernelSpec(
        slug="neurofy-train-fp1f7",
        title="Neurofy Train Fp1f7",
        description="Train Model 1 (Fp1-F7) across all leave-one-seizure-out folds.",
        commands=(f"{EXPERIMENT} --config fp1f7",),
        packages=("mlflow",),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
    ),
    "train-fp1f7-t8p8": KernelSpec(
        slug="neurofy-train-fp1f7-t8p8",
        title="Neurofy Train Fp1f7 T8p8",
        description="Train Model 2 (Fp1-F7 and T8-P8) on top of Model 1's results, then compare.",
        commands=(f"{EXPERIMENT} --config fp1f7_t8p8", COMPARE),
        packages=("mlflow",),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
        kernel_sources=(f"{KAGGLE_USER}/neurofy-train-fp1f7",),
    ),
    "explain": KernelSpec(
        slug="neurofy-explain",
        title="Neurofy Explain",
        description="Explain both models with Grad-CAM, reliability checks and channel ablation.",
        commands=(f"{EXPLAIN} --config fp1f7", f"{EXPLAIN} --config fp1f7_t8p8"),
        packages=("mlflow",),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
        kernel_sources=(f"{KAGGLE_USER}/neurofy-train-fp1f7-t8p8",),
    ),
}


def load_env(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs from a dotenv file."""
    if not path.is_file():
        return {}
    pairs = (
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    return {key.strip(): value.strip().strip("\"'") for key, value in pairs}


def git(*args: str) -> bytes:
    """Run a git command in the repository and return its output."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True).stdout


def build_code_package(commit: str, output_dir: Path) -> None:
    """Write the committed code, its commit hash and the dataset metadata to a folder."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    archive = git("archive", "--format=tar", commit, *CODE_PATHS)
    with tarfile.open(fileobj=io.BytesIO(archive)) as package:
        package.extractall(output_dir, filter="data")

    (output_dir / COMMIT_FILE).write_text(commit, encoding="utf-8")
    metadata = {
        "title": CODE_DATASET_SLUG,
        "id": CODE_DATASET,
        "licenses": [{"name": "other"}],
    }
    (output_dir / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def dataset_publish_command(exists: bool, folder: Path, commit: str) -> list[str]:
    """Return the CLI command that publishes a folder; new datasets default to private."""
    base = ["kaggle", "datasets"]
    if exists:
        return [*base, "version", "-p", str(folder), "-r", "zip", "-m", f"Commit {commit[:7]}"]
    return [*base, "create", "-p", str(folder), "-r", "zip"]


def _dataset_status(env: dict[str, str]) -> str | None:
    result = subprocess.run(
        ["kaggle", "datasets", "status", CODE_DATASET],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip().splitlines()[-1].lower() if result.returncode == 0 else None


def publish_code(commit: str, env: dict[str, str]) -> None:
    """Publish the commit as a new version of the private code dataset and wait until ready."""
    folder = BUILD_ROOT / "code"
    commit_file = folder / COMMIT_FILE
    already_published = commit_file.is_file() and commit_file.read_text().strip() == commit
    status = _dataset_status(env)
    if already_published and status == "ready":
        logger.info("Code dataset already holds commit %s", commit[:7])
        return

    build_code_package(commit, folder)
    subprocess.run(dataset_publish_command(status is not None, folder, commit), check=True, env=env)

    deadline = time.monotonic() + _DATASET_READY_TIMEOUT_SECONDS
    while _dataset_status(env) != "ready":
        if time.monotonic() > deadline:
            raise TimeoutError(f"{CODE_DATASET} not ready after publishing")
        time.sleep(10)
    logger.info("Published commit %s to %s", commit[:7], CODE_DATASET)


def _readable_command(command: str) -> str:
    """Put each option of a shell command on its own line."""
    return command.replace(" --", " \\\n  --")


def _markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _bash_cell(command: str) -> dict:
    return _code_cell(f"%%bash\nset -e\n{_readable_command(command)}\n")


def build_notebook(spec: KernelSpec, commit: str, extra_args: str = "") -> dict:
    """Create a short, readable notebook that runs the kernel's commands."""
    first, *rest = spec.commands
    commands = [f"{first} {extra_args}".strip(), *rest]
    cells = [
        _markdown_cell(
            f"# {spec.title}\n\n"
            f"{spec.description}\n\n"
            f"Runs commit `{commit[:7]}`. The code is in the attached `{CODE_DATASET_SLUG}` "
            "dataset; this notebook only launches it.\n"
        ),
        _code_cell(_SETUP_CELL.format(commit_file=COMMIT_FILE, commit=commit)),
    ]
    if spec.packages:
        cells.append(_bash_cell(f"pip install -q {' '.join(spec.packages)}"))
    cells.extend(_bash_cell(command) for command in commands)
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_metadata(spec: KernelSpec, code_file: str) -> dict:
    """Create the kernel metadata; kernels are always private."""
    metadata = {
        "id": f"{KAGGLE_USER}/{spec.slug}",
        "title": spec.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": str(spec.gpu).lower(),
        "enable_internet": "true",
        "dataset_sources": [CODE_DATASET, *spec.dataset_sources],
        "competition_sources": [],
        "kernel_sources": list(spec.kernel_sources),
        "model_sources": [],
    }
    if spec.gpu:
        metadata["machine_shape"] = T4_X2
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=sorted(KERNELS))
    parser.add_argument("--args", default="", help="Extra arguments for the kernel's first command")
    parser.add_argument("--build-only", action="store_true", help="Generate files without pushing")
    args = parser.parse_args()

    configure_logging()
    if git("status", "--porcelain").strip():
        parser.error("Working tree is dirty; commit first so the run matches a commit")

    spec = KERNELS[args.kernel]
    commit = git("rev-parse", "HEAD").decode().strip()

    kernel_dir = BUILD_ROOT / spec.slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{spec.slug}.ipynb"
    notebook = build_notebook(spec, commit, args.args)
    (kernel_dir / code_file).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(spec, code_file), indent=2),
        encoding="utf-8",
    )
    logger.info("Built %s at commit %s", spec.slug, commit[:7])

    if args.build_only:
        return
    env = {**os.environ, **load_env(REPO_ROOT / ".env")}
    publish_code(commit, env)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True, env=env)
    logger.info("Pushed %s/%s", KAGGLE_USER, spec.slug)


if __name__ == "__main__":
    sys.exit(main())
