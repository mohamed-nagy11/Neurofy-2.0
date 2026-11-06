# Neurofy 2.0

Single-channel EEG seizure prediction. The project reproduces the patient-specific protocol of [Jang et al., *Single-channel EEG-based seizure prediction using deep learning*, Scientific Reports (2026)](https://doi.org/10.1038/s41598-026-44670-7) on the [CHB-MIT dataset](https://physionet.org/content/chbmit/1.0.0/), followed by ablation studies.

| Model | Config | Input |
|---|---|---|
| Model 1 | `fp1f7` | Fp1-F7 |
| Model 2 | `fp1f7_t8p8` | Fp1-F7 and T8-P8 |

## Pipeline

1. **Verify:** check a dataset copy against the checksums published by PhysioNet.
2. **Select:** build the cohort from the annotations, and label preictal (-32 to -2 min before onset) and interictal (at least 3 h from every seizure) data.
3. **Preprocess:** apply a causal 0.1-50 Hz Butterworth filter at the native 256 Hz, and store each patient as one array.
4. **Train:** run leave-one-seizure-out per patient, with one model per held-out seizure, on spectrograms computed on the GPU.
5. **Evaluate:** apply firing power (119 windows, threshold 0.5), then report segment metrics and event-based sensitivity per fold, patient and cohort.
6. **Explain:** run Grad-CAM with reliability checks, and channel ablation for the two-channel model.

## Structure

```text
.
├── src/neurofy/
│   ├── config.py      # Experiment configurations
│   ├── data/          # Annotations, cohort, preprocessing, folds, window batches
│   ├── models/        # Spectrogram transform and network
│   ├── training/      # Training loop, early stopping, checkpoints, parallel runner
│   ├── evaluation/    # Metrics, firing power, aggregation, comparison, explanations, plots
│   └── utils/         # Logging, seeding, device selection, MLflow runs and server
├── scripts/           # Command-line entry points
├── tests/
└── pyproject.toml
```

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
uv pip install -e ".[dev,kaggle]"
```

Secrets go in a `.env` file, using the names in [.env.example](.env.example).

## Usage

Experiments run on Kaggle. `scripts/kaggle_push.py` packs the current commit into a private notebook and pushes it, so a kernel needs neither repository access nor secrets. The working tree must be clean.

```bash
python scripts/kaggle_push.py verify
python scripts/kaggle_push.py preprocess
python scripts/kaggle_push.py smoke
python scripts/kaggle_push.py train-fp1f7 --args "--workers-per-gpu 2"
python scripts/kaggle_push.py train-fp1f7-t8p8 --args "--workers-per-gpu 2"
python scripts/kaggle_push.py explain
```

Use `--dataset <owner/slug>` to attach a preprocessed Kaggle dataset instead of the preprocessing kernel's output.

The scripts also run directly, with `--data-dir` pointing at preprocessed data:

```bash
python scripts/run_experiment.py --config fp1f7 --data-dir chbmit_preprocessed --device cpu
python scripts/compare_models.py
python scripts/explain.py --config fp1f7 --data-dir chbmit_preprocessed --device cpu
```

A session stops taking new folds after `--max-hours`. Running the experiment again skips finished folds.

## Experiment tracking

Runs are tracked with [MLflow](https://mlflow.org). Each model has a parent run, with one nested run per fold.

| Level | Logged data |
|---|---|
| Fold | Configuration, per-epoch metrics, best checkpoint, window and segment metrics, event metrics, per-window predictions, learning curves, confusion matrices, ROC curves, prediction timeline, model summary, run log |
| Model | Patient table, cohort table, per-patient plot |
| Comparison | Results against the paper, paired comparison of the two models |
| Explanation | Band importance, Grad-CAM heatmaps, deletion test, sanity check, channel ablation |

On Kaggle, a local MLflow server stores runs, and the session ends with one `mlflow_results.tar.gz`. To browse it locally:

```bash
kaggle kernels output <owner>/<kernel> --file-pattern mlflow_results.tar.gz -p results
tar -xzf results/mlflow_results.tar.gz -C results
mlflow server --backend-store-uri sqlite:///results/mlflow.db --artifacts-destination results/artifacts
```

## Development

```bash
pytest
ruff check .
ruff format --check .
```
