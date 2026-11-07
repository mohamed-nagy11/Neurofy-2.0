"""Build a readable Kaggle notebook that runs a pushed commit, and push it."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from neurofy.utils.logger import configure_logging

logger = logging.getLogger("neurofy.scripts.kaggle_push")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = REPO_ROOT / "build" / "kaggle"
REPO_URL = "https://github.com/mohamed-nagy11/Neurofy-2.0"
KAGGLE_USER = "mohamed11nagy11"
CHBMIT_DATASET = "abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric"
PREPROCESSED_DATASET = f"{KAGGLE_USER}/neurofy-chbmit-preprocessed"
T4_X2 = "NvidiaTeslaT4"
CODE_DIR = "/tmp/neurofy"
RESULTS = "--tracking-dir /tmp/mlflow --archive /kaggle/working/mlflow_results.tar.gz"
EXPERIMENT = f"python {CODE_DIR}/scripts/run_experiment.py {RESULTS}"
COMPARE = f"python {CODE_DIR}/scripts/compare_models.py {RESULTS}"
EXPLAIN = f"python {CODE_DIR}/scripts/explain.py {RESULTS}"
RUN_HELPER = '''import shlex
import subprocess
import sys


def run(command):
    """Run a command, stream its output live, and fail the notebook if it fails."""
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Command failed: {command}")
'''
SMOKE_PATIENTS = "chb04 chb02 chb16 chb17 chb19 chb22"
MODEL_1_KERNEL = "neurofy-train-fp1f7"
MODEL_2_KERNEL = "neurofy-train-fp1f7-t8p8"


@dataclass(frozen=True)
class Step:
    """One titled command in a notebook."""

    title: str
    command: str


@dataclass(frozen=True)
class KernelSpec:
    """Settings for one Kaggle kernel."""

    slug: str
    title: str
    description: str
    steps: tuple[Step, ...]
    gpu: bool = False
    dataset_sources: tuple[str, ...] = ()
    kernel_sources: tuple[str, ...] = ()


def _benchmark(workers: int, title: str) -> Step:
    return Step(
        title,
        f"{EXPERIMENT} --config fp1f7 --experiment neurofy-smoke-w{workers} "
        f"--patients {SMOKE_PATIENTS} --max-epochs 30 --workers-per-gpu {workers}",
    )


KERNELS = {
    "verify": KernelSpec(
        slug="neurofy-verify",
        title="Neurofy Verify",
        description="Verify the CHB-MIT copy against the checksums published by PhysioNet.",
        steps=(Step("Verify the dataset", f"python {CODE_DIR}/scripts/verify_dataset.py"),),
        dataset_sources=(CHBMIT_DATASET,),
    ),
    "preprocess": KernelSpec(
        slug="neurofy-preprocess",
        title="Neurofy Preprocess",
        description="Select the cohort, filter its signals and store them with a manifest.",
        steps=(
            Step(
                "Preprocess the cohort",
                f"python {CODE_DIR}/scripts/preprocess.py "
                "--output /kaggle/working/chbmit_preprocessed",
            ),
        ),
        dataset_sources=(CHBMIT_DATASET,),
    ),
    "smoke": KernelSpec(
        slug="neurofy-smoke",
        title="Neurofy Smoke",
        description="Benchmark workers per GPU, then check resuming, comparison and Grad-CAM.",
        steps=(
            _benchmark(1, "Benchmark 1 worker per GPU"),
            _benchmark(2, "Benchmark 2 workers per GPU"),
            _benchmark(4, "Benchmark 4 workers per GPU"),
            _benchmark(1, "Resume finished folds"),
            Step(
                "Train the two-channel model briefly",
                f"{EXPERIMENT} --config fp1f7_t8p8 --experiment neurofy-smoke-w1 "
                f"--patients {SMOKE_PATIENTS} --max-epochs 3 --workers-per-gpu 2",
            ),
            Step("Compare the models", f"{COMPARE} --experiment neurofy-smoke-w1"),
            Step(
                "Explain the two-channel model",
                f"{EXPLAIN} --config fp1f7_t8p8 --experiment neurofy-smoke-w1",
            ),
        ),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
    ),
    "train-fp1f7": KernelSpec(
        slug=MODEL_1_KERNEL,
        title="Neurofy Train Fp1f7",
        description="Train Model 1 (Fp1-F7) across all leave-one-seizure-out folds.",
        steps=(Step("Train Model 1", f"{EXPERIMENT} --config fp1f7 --workers-per-gpu 2"),),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
    ),
    "train-fp1f7-t8p8": KernelSpec(
        slug=MODEL_2_KERNEL,
        title="Neurofy Train Fp1f7 T8p8",
        description="Train Model 2 (Fp1-F7 and T8-P8) across all leave-one-seizure-out folds.",
        steps=(Step("Train Model 2", f"{EXPERIMENT} --config fp1f7_t8p8 --workers-per-gpu 2"),),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
    ),
    "analysis": KernelSpec(
        slug="neurofy-analysis",
        title="Neurofy Analysis",
        description="Compare both models with the paper and each other, then explain them.",
        steps=(
            Step(
                "Compare the models",
                f"python {CODE_DIR}/scripts/compare_models.py "
                "--tracking-dir /tmp/mlflow-comparison "
                "--archive /kaggle/working/mlflow_comparison.tar.gz "
                f"--first-results {MODEL_1_KERNEL} --second-results {MODEL_2_KERNEL}",
            ),
            Step(
                "Explain Model 1",
                f"python {CODE_DIR}/scripts/explain.py --config fp1f7 "
                "--tracking-dir /tmp/mlflow-fp1f7 "
                "--archive /kaggle/working/mlflow_fp1f7.tar.gz "
                f"--restore-from {MODEL_1_KERNEL}",
            ),
            Step(
                "Explain Model 2",
                f"python {CODE_DIR}/scripts/explain.py --config fp1f7_t8p8 "
                "--tracking-dir /tmp/mlflow-fp1f7-t8p8 "
                "--archive /kaggle/working/mlflow_fp1f7_t8p8.tar.gz "
                f"--restore-from {MODEL_2_KERNEL}",
            ),
        ),
        gpu=True,
        dataset_sources=(PREPROCESSED_DATASET,),
        kernel_sources=(f"{KAGGLE_USER}/{MODEL_1_KERNEL}", f"{KAGGLE_USER}/{MODEL_2_KERNEL}"),
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


def is_pushed(commit: str) -> bool:
    """Return whether a commit is on the public main branch, where notebooks clone it from."""
    git("fetch", "--quiet", "origin")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def _run_call(command: str) -> str:
    """Write a `run` call with each option of the command on its own line."""
    readable = command.replace(" --", "
  --")
    return f'run("""
{readable}
""")
'


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


def build_notebook(spec: KernelSpec, commit: str, extra_args: str = "", note: str = "") -> dict:
    """Create a short, readable notebook that installs a commit and runs the kernel's steps."""
    header = (
        f"# {spec.title}\n\n"
        f"{spec.description}\n\n"
        f"Code: [Neurofy 2.0 at `{commit[:7]}`]({REPO_URL}/tree/{commit})\n"
    )
    if note:
        header += f"\n{note}\n"

    cells = [
        _markdown_cell(header),
        _markdown_cell("## Setup
"),
        _code_cell(RUN_HELPER),
        _code_cell(
            f'run("git clone --quiet {REPO_URL}.git {CODE_DIR}")
'
            f'run("git -C {CODE_DIR} checkout --quiet {commit}")
'
            f'run("pip install --quiet {CODE_DIR}")
'
        ),
        _code_cell(f"%env NEUROFY_GIT_COMMIT={commit}
"),
    ]
    for index, step in enumerate(spec.steps):
        command = f"{step.command} {extra_args}".strip() if index == 0 else step.command
        cells.append(_markdown_cell(f"## {step.title}
"))
        cells.append(_code_cell(_run_call(command)))
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
        "dataset_sources": list(spec.dataset_sources),
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
    parser.add_argument("--args", default="", help="Extra arguments for the kernel's first step")
    parser.add_argument("--commit", help="Commit to show; defaults to HEAD")
    parser.add_argument("--note", default="", help="Text to add under the notebook's title")
    parser.add_argument("--no-run", action="store_true", help="Save a new version without running")
    parser.add_argument("--build-only", action="store_true", help="Generate files without pushing")
    args = parser.parse_args()

    configure_logging()
    if not args.commit and git("status", "--porcelain").strip():
        parser.error("Working tree is dirty; commit first so the run matches a commit")

    spec = KERNELS[args.kernel]
    commit = git("rev-parse", args.commit or "HEAD").decode().strip()

    kernel_dir = BUILD_ROOT / spec.slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{spec.slug}.ipynb"
    notebook = build_notebook(spec, commit, args.args, args.note)
    (kernel_dir / code_file).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(spec, code_file), indent=2),
        encoding="utf-8",
    )
    logger.info("Built %s at commit %s", spec.slug, commit[:7])

    if args.build_only:
        return
    if not is_pushed(commit):
        parser.error(f"Commit {commit[:7]} is not on origin/main; push it first")

    # Resolve through PATH; on Windows a bare name can pick a different installed version.
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        parser.error('Kaggle CLI not found; install it with: uv pip install -e ".[kaggle]"')
    command = [kaggle, "kernels", "push", "-p", str(kernel_dir)]
    if args.no_run:
        command.append("--no-run")
    env = {**os.environ, **load_env(REPO_ROOT / ".env")}
    subprocess.run(command, check=True, env=env)
    logger.info("Pushed %s/%s", KAGGLE_USER, spec.slug)


if __name__ == "__main__":
    sys.exit(main())
