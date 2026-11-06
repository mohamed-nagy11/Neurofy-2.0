# Neurofy 2.0

Single-channel EEG seizure prediction. The project reproduces [Jang et al., *Single-channel EEG-based seizure prediction using deep learning*, Scientific Reports (2026)](https://doi.org/10.1038/s41598-026-44670-7) on the CHB-MIT dataset, followed by ablation studies.

> **Status:** project scaffold. The dataset and model are placeholders until the paper's pipeline is implemented.

## Structure

```text
.
├── src/neurofy/
│   ├── config.py      # Baseline and ablation configurations
│   ├── data/          # Datasets and data loaders
│   ├── models/        # Model architectures
│   ├── training/      # Training loop, loss, early stopping, checkpoints
│   ├── evaluation/    # Evaluation, metrics, plots, MLflow reporting
│   └── utils/         # Logging, seeding, device selection, MLflow runs
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
uv pip install -e ".[dev]"
```

## Usage

Train a configuration (`baseline`, `ablation1`, `ablation2` or `ablation3`):

```bash
python scripts/train.py --config baseline
```

Evaluate a saved checkpoint on the test split:

```bash
python scripts/evaluate.py --config baseline --checkpoint path/to/best_model.pt
```

## Experiment tracking

Every run is tracked with [MLflow](https://mlflow.org). Set `MLFLOW_TRACKING_URI` to choose where runs are stored.

| Type | Logged data |
|---|---|
| Parameters | Full configuration and parameter count |
| Metrics | Per-epoch train and validation metrics; final best-model and test metrics |
| Weights | Best checkpoint, selected by validation accuracy |
| Plots | Learning curves, confusion matrices and ROC curves |
| Records | Configuration file, metric history, model summary and run log |

```bash
mlflow ui
```

## Development

```bash
pytest
ruff check .
ruff format --check .
```
