"""Evaluate a checkpoint on the test split."""

import argparse
import logging
from pathlib import Path

import mlflow

from neurofy.config import CONFIGS, get_config
from neurofy.data.dataset import get_dataloader
from neurofy.evaluation.evaluator import evaluate
from neurofy.evaluation.reporting import log_evaluation
from neurofy.models.simple import create_model
from neurofy.training.checkpoint import load_checkpoint
from neurofy.training.loss import get_loss_fn
from neurofy.utils.logger import configure_logging
from neurofy.utils.runtime import get_device, set_seed
from neurofy.utils.tracking import experiment_run

logger = logging.getLogger("neurofy.scripts.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    configure_logging()
    config = get_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)

    with experiment_run(config, "evaluate"):
        mlflow.set_tag("device", str(device))
        mlflow.log_param("checkpoint", str(args.checkpoint))

        model = create_model(config).to(device)
        checkpoint = load_checkpoint(args.checkpoint, model, device)
        logger.info("Loaded checkpoint from epoch %d", checkpoint["epoch"])

        result = evaluate(model, get_dataloader(config, "test"), get_loss_fn(), device)
        log_evaluation(result, "test")
        logger.info(
            "Test | loss=%.4f acc=%.4f",
            result.metrics["loss"],
            result.metrics["accuracy"],
        )


if __name__ == "__main__":
    main()
