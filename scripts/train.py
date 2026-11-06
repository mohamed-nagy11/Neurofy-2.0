"""Train a model from a named configuration."""

import argparse

from neurofy.config import CONFIGS, get_config
from neurofy.training.trainer import train
from neurofy.utils.logger import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    args = parser.parse_args()

    configure_logging()
    train(get_config(args.config))


if __name__ == "__main__":
    main()
