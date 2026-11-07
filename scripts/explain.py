"""Explain trained folds with Grad-CAM, reliability checks and channel ablation."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import torch

from neurofy.config import CONFIGS, ExperimentConfig, get_config
from neurofy.data.dataset import load_records, load_signal
from neurofy.data.folds import build_folds
from neurofy.evaluation.aggregation import rows_to_csv
from neurofy.evaluation.explainability import (
    band_importance,
    band_importance_by_time,
    bin_frequencies,
    channel_ablation,
    deletion_test,
    explain_fold,
    grad_cam,
    map_similarity,
    sample_windows_by_outcome,
)
from neurofy.evaluation.plots import (
    plot_band_importance,
    plot_band_importance_by_time,
    plot_deletion_test,
    plot_window_heatmap,
)
from neurofy.models.mobilenet import create_model
from neurofy.training.checkpoint import load_checkpoint
from neurofy.utils.logger import configure_logging
from neurofy.utils.paths import KAGGLE_INPUT, find_directory_containing
from neurofy.utils.runtime import get_device, set_seed
from neurofy.utils.tracking import (
    RESULTS_ARCHIVE,
    experiment_run,
    finished_fold_runs,
    tracking_session,
)

logger = logging.getLogger("neurofy.scripts.explain")

_RELIABILITY_WINDOWS = 128


def _explain(config: ExperimentConfig, data_dir: Path) -> None:
    device = get_device(config.device)
    records = load_records(data_dir)
    frequencies = bin_frequencies(config)
    runs = sorted(
        finished_fold_runs(config.experiment_name, config.name),
        key=lambda run: run.data.tags["fold_key"],
    )
    if not runs:
        raise ValueError(f"No finished folds for {config.name} in {config.experiment_name}")

    band_rows, time_rows, deletion_rows, sanity_rows, ablation_rows = [], [], [], [], []

    with experiment_run(
        config,
        "explanation",
        run_name=f"explain-{config.name}",
        tags={"device": str(device)},
    ):
        for run in runs:
            patient, seizure = run.data.tags["patient"], int(run.data.tags["seizure"])
            record = records[patient]
            fold = build_folds(record, config)[seizure]
            signal = load_signal(record, config.channels, device)
            set_seed(fold.seed)

            model = create_model(config).to(device)
            with TemporaryDirectory() as tmp_dir:
                path = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id,
                    artifact_path="checkpoints/best_model.pt",
                    dst_path=tmp_dir,
                )
                load_checkpoint(Path(path), model, device)
            model.eval()

            explanation = explain_fold(model, fold, signal, config)
            preictal_maps = explanation.preictal_maps.cpu().numpy()
            interictal_maps = explanation.interictal_maps.cpu().numpy()
            identity = {"patient": patient, "seizure": seizure}

            band_rows.append(
                {**identity, "class": "preictal", **band_importance(preictal_maps, frequencies)}
            )
            band_rows.append(
                {**identity, "class": "interictal", **band_importance(interictal_maps, frequencies)}
            )
            for row in band_importance_by_time(
                preictal_maps,
                fold.test_seconds_to_onset,
                frequencies,
            ):
                time_rows.append({**identity, **row})

            count = min(_RELIABILITY_WINDOWS, len(fold.test_preictal))
            generator = torch.Generator(device=device).manual_seed(fold.seed)
            indices = torch.randperm(
                explanation.preictal_spectrograms.shape[0],
                generator=generator,
                device=device,
            )[:count]
            subset = explanation.preictal_spectrograms[indices]
            for row in deletion_test(model, subset, explanation.preictal_maps[indices], generator):
                deletion_rows.append({**identity, **row})

            random_model = create_model(config).to(device).eval()
            random_maps = grad_cam(random_model, subset)
            sanity_rows.append(
                {
                    **identity,
                    "similarity_to_random_model": map_similarity(
                        explanation.preictal_maps[indices],
                        random_maps,
                    ),
                }
            )

            for row in channel_ablation(model, fold, signal, explanation, config):
                ablation_rows.append({**identity, **row})

            with torch.inference_mode():
                preictal_probabilities = torch.sigmoid(
                    model.forward_spectrogram(explanation.preictal_spectrograms)
                )
                interictal_probabilities = torch.sigmoid(
                    model.forward_spectrogram(explanation.interictal_spectrograms)
                )
            for probabilities, is_preictal, spectrograms, maps in (
                (
                    preictal_probabilities,
                    True,
                    explanation.preictal_spectrograms,
                    explanation.preictal_maps,
                ),
                (
                    interictal_probabilities,
                    False,
                    explanation.interictal_spectrograms,
                    explanation.interictal_maps,
                ),
            ):
                samples = sample_windows_by_outcome(probabilities.cpu().numpy(), is_preictal)
                for name, index in samples.items():
                    mlflow.log_figure(
                        plot_window_heatmap(
                            spectrograms[index, 0].cpu().numpy(),
                            maps[index].cpu().numpy(),
                            frequencies,
                            f"{patient} seizure {seizure}: {name.replace('_', ' ')}",
                        ),
                        f"heatmaps/{patient}_seizure-{seizure}_{name}.png",
                    )
            logger.info("Explained %s", fold.key)

        mlflow.log_text(rows_to_csv(band_rows), "band_importance.csv")
        mlflow.log_text(rows_to_csv(time_rows), "band_importance_by_time.csv")
        mlflow.log_text(rows_to_csv(deletion_rows), "deletion_test.csv")
        mlflow.log_text(rows_to_csv(sanity_rows), "sanity_check.csv")
        mlflow.log_text(rows_to_csv(ablation_rows), "channel_ablation.csv")
        mlflow.log_figure(plot_band_importance(band_rows), "plots/band_importance.png")
        mlflow.log_figure(
            plot_band_importance_by_time(time_rows),
            "plots/band_importance_by_time.png",
        )
        mlflow.log_figure(plot_deletion_test(deletion_rows), "plots/deletion_test.png")

        cam_drop = np.mean([row["cam_logit_drop"] for row in deletion_rows])
        random_drop = np.mean([row["random_logit_drop"] for row in deletion_rows])
        similarity = np.mean([row["similarity_to_random_model"] for row in sanity_rows])
        mlflow.log_metrics(
            {
                "reliability/cam_logit_drop": float(cam_drop),
                "reliability/random_logit_drop": float(random_drop),
                "reliability/similarity_to_random_model": float(similarity),
            }
        )
        logger.info(
            "Logit drop %.3f for CAM vs %.3f for moved regions; similarity to random model %.3f",
            cam_drop,
            random_drop,
            similarity,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--data-dir", type=Path, help="Found automatically on Kaggle")
    parser.add_argument("--tracking-dir", type=Path, help="Serve a local MLflow store from here")
    parser.add_argument("--archive", type=Path, default=Path(RESULTS_ARCHIVE))
    parser.add_argument("--experiment", help="MLflow experiment name")
    parser.add_argument("--device", help="Override the device, such as cpu")
    parser.add_argument("--restore-from", help="Attached kernel holding this model's results")
    parser.add_argument("--input-dir", type=Path, default=KAGGLE_INPUT)
    args = parser.parse_args()

    configure_logging()
    overrides = {"experiment_name": args.experiment, "device": args.device}
    config = replace(
        get_config(args.config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    data_dir = args.data_dir or find_directory_containing("manifest.json")

    if args.tracking_dir:
        with tracking_session(
            args.tracking_dir,
            args.archive,
            restore_from=args.restore_from,
            search_root=args.input_dir,
        ):
            _explain(config, data_dir)
    else:
        _explain(config, data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
