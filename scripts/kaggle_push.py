"""Build a private Kaggle notebook from the current commit and push it."""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from neurofy.utils.logger import configure_logging

logger = logging.getLogger("neurofy.scripts.kaggle_push")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = REPO_ROOT / "build" / "kaggle"
KAGGLE_USER = "mohamed11nagy11"
CHBMIT_DATASET = "abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric"
T4_X2 = "NvidiaTeslaT4"
CODE_DIR = "/tmp/neurofy_code"
PREPROCESS_KERNEL = f"{KAGGLE_USER}/neurofy-preprocess"
RESULTS = "--tracking-dir /tmp/mlflow --archive /kaggle/working/mlflow_results.tar.gz"
EXPERIMENT = f"python {{code}}/scripts/run_experiment.py {RESULTS}"
SMOKE_PATIENTS = "chb04 chb02 chb16 chb17 chb19 chb22"

_BOOTSTRAP = """\
import base64, io, os, subprocess, sys, tarfile

CODE_DIR = "{code_dir}"
ARCHIVE = "{archive}"

with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE)), mode="r:gz") as archive:
    archive.extractall(CODE_DIR)

env = {{**os.environ, "PYTHONPATH": f"{{CODE_DIR}}/src", "NEUROFY_GIT_COMMIT": "{commit}"}}


def run(command):
    process = subprocess.Popen(
        command, shell=True, cwd="/kaggle/working", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        sys.stdout.write(line)
    if process.wait() != 0:
        raise RuntimeError(f"Command failed: {{command}}")
"""


@dataclass(frozen=True)
class KernelSpec:
    """Settings for one Kaggle kernel."""

    slug: str
    title: str
    command: str
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
        command="python {code}/scripts/verify_dataset.py",
        dataset_sources=(CHBMIT_DATASET,),
    ),
    "preprocess": KernelSpec(
        slug="neurofy-preprocess",
        title="Neurofy Preprocess",
        command="python {code}/scripts/preprocess.py --output /kaggle/working/chbmit_preprocessed",
        packages=("pyedflib",),
        dataset_sources=(CHBMIT_DATASET,),
    ),
    "smoke": KernelSpec(
        slug="neurofy-smoke",
        title="Neurofy Smoke",
        command=" && ".join(
            [
                _benchmark(1),
                _benchmark(2),
                _benchmark(4),
                _benchmark(1),
                f"{EXPERIMENT} --config fp1f7_t8p8 --experiment neurofy-smoke-dual "
                "--patients chb17 --max-epochs 3",
            ]
        ),
        packages=("mlflow",),
        gpu=True,
        kernel_sources=(PREPROCESS_KERNEL,),
    ),
    "train-fp1f7": KernelSpec(
        slug="neurofy-train-fp1f7",
        title="Neurofy Train Fp1f7",
        command=f"{EXPERIMENT} --config fp1f7",
        packages=("mlflow",),
        gpu=True,
        kernel_sources=(PREPROCESS_KERNEL,),
    ),
    "train-fp1f7-t8p8": KernelSpec(
        slug="neurofy-train-fp1f7-t8p8",
        title="Neurofy Train Fp1f7 T8p8",
        command=f"{EXPERIMENT} --config fp1f7_t8p8",
        packages=("mlflow",),
        gpu=True,
        kernel_sources=(PREPROCESS_KERNEL, f"{KAGGLE_USER}/neurofy-train-fp1f7"),
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


def build_notebook(archive: bytes, commit: str, spec: KernelSpec, command: str) -> dict:
    """Create a notebook that unpacks the embedded code and runs one command."""
    bootstrap = _BOOTSTRAP.format(
        archive=base64.b64encode(archive).decode(),
        commit=commit,
        code_dir=CODE_DIR,
    )
    cells = [bootstrap]
    if spec.packages:
        cells.append(f'run("pip install -q {" ".join(spec.packages)}")')
    cells.append(f'run("{command.format(code=CODE_DIR)}")')
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell.splitlines(keepends=True),
            }
            for cell in cells
        ],
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
    parser.add_argument("--args", default="", help="Extra arguments for the kernel's command")
    parser.add_argument("--dataset", help="Attach this preprocessed dataset, not kernel output")
    parser.add_argument("--build-only", action="store_true", help="Generate files without pushing")
    args = parser.parse_args()

    configure_logging()
    if git("status", "--porcelain").strip():
        parser.error("Working tree is dirty; commit first so the run matches a commit")

    spec = KERNELS[args.kernel]
    if args.dataset:
        spec = replace(
            spec,
            dataset_sources=(*spec.dataset_sources, args.dataset),
            kernel_sources=tuple(s for s in spec.kernel_sources if s != PREPROCESS_KERNEL),
        )
    commit = git("rev-parse", "HEAD").decode().strip()
    archive = git("archive", "--format=tar.gz", "HEAD")
    command = f"{spec.command} {args.args}".strip()

    kernel_dir = BUILD_ROOT / spec.slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{spec.slug}.ipynb"
    notebook = build_notebook(archive, commit, spec, command)
    (kernel_dir / code_file).write_text(json.dumps(notebook), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(spec, code_file), indent=2),
        encoding="utf-8",
    )
    logger.info("Built %s at commit %s (%d KB)", spec.slug, commit[:7], len(archive) // 1024)

    if args.build_only:
        return
    env = {**os.environ, **load_env(REPO_ROOT / ".env")}
    subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True, env=env)
    logger.info("Pushed %s/%s", KAGGLE_USER, spec.slug)


if __name__ == "__main__":
    sys.exit(main())
