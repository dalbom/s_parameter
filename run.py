# Import necessary libraries for parsing arguments, loading configurations, managing filesystem, and timing
import argparse
import importlib
import yaml
import os
import time
from shutil import copy2
from utils.helper import (
    dict2namespace,
)  # A utility to convert dictionaries to namespaces for easier attribute access


def get_args():
    # Initialize argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Training or evaulation")

    # Add argument for specifying the configuration file path
    parser.add_argument("config", type=str, help="The configuration yaml file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the YAML configuration file specified by the user
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert the configuration dictionary to a namespace for easier attribute access
    config = dict2namespace(config)

    # Initialize logging prefix, which can be customized via the configuration file
    log_prefix = ""
    if config.alias is not None:
        log_prefix = config.alias + "_"

    # Generate a unique runtime identifier based on the current time
    runtime = time.strftime("%Y-%b-%d-%H-%M-%S")

    # Setup logging and checkpoint directory paths using the log prefix and runtime
    config.log_name = f"logs/{log_prefix}{runtime}/log.txt"
    config.save_dir = f"logs/{log_prefix}{runtime}/checkpoints"
    config.log_dir = f"logs/{log_prefix}{runtime}"

    # Create necessary directories for logging and checkpoints
    os.makedirs(os.path.join(config.log_dir, "config"))
    os.makedirs(config.save_dir)

    # Copy the configuration file to the logging directory for record-keeping
    copy2(args.config, os.path.join(config.log_dir, "config"))

    # Return parsed arguments and configuration for further use
    return args, config


def main(args, cfg):
    # Dynamically import the model module specified in the configuration
    model_lib = importlib.import_module(cfg.backbone.type)
    # Create a model instance based on the configuration
    model = model_lib.get_model(cfg.backbone)

    # Dynamically import the dataset module specified in the configuration
    dataset_lib = importlib.import_module(cfg.dataset.type)
    # Create a dataset instance for training based on the configuration
    dataset = dataset_lib.get_dataset(cfg.dataset.training)

    # Optionally create a test dataset if specified in the configuration
    if hasattr(cfg.dataset, "test"):
        cfg.trainer.dataset_test = dataset_lib.get_dataset(cfg.dataset.test)

    # Assign the dataset and model to the trainer configuration
    cfg.trainer.dataset = dataset
    cfg.trainer.backbone = model
    cfg.trainer.log_dir = cfg.log_dir

    # Dynamically import the trainer module specified in the configuration
    trainer_lib = importlib.import_module(cfg.trainer.type)
    # Create a trainer instance based on the configuration
    trainer = trainer_lib.get_trainer(cfg.trainer)

    # Decide between inference or cross-validation based on the presence of a checkpoint in the configuration
    if hasattr(cfg.trainer, "checkpoint"):
        trainer.inference()
    else:
        trainer.cross_validate()


if __name__ == "__main__":
    # Entry point of the script: parse arguments, load configuration, and start main process
    args, cfg = get_args()
    main(args, cfg)
