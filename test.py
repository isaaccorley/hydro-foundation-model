import argparse
import os

import lightning
import mlflow  # noqa: F401
import torch
from hydra.utils import instantiate, get_class
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import OmegaConf

import src  # noqa: F401


torch.set_float32_matmul_precision("medium")


def main(args):
    lightning.pytorch.seed_everything(args.seed)
    config = OmegaConf.load(args.config)
    module = get_class(config.module._target_).load_from_checkpoint(
        args.ckpt, map_location="cpu", weights=config.module.weights
    )

    if args.root is None:
        datamodule = instantiate(config.datamodule)
    else:
        datamodule = instantiate(config.datamodule, root=args.root)

    if args.logger == "mlflow":
        logger = MLFlowLogger(
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", config.experiment_name + "_test"
            ),
            run_id=os.environ.get("MLFLOW_RUN_ID", None),
        )
        logger.log_hyperparams(dict(config))
    else:
        logger = TensorBoardLogger(
            save_dir="lightning_logs", name=config.experiment_name + "_test"
        )

    devices = [args.device] if args.device is not None else config.trainer.devices
    trainer = instantiate(config.trainer, logger=logger, devices=devices)
    trainer.test(module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        required=False,
        help="Optionally override root from config",
    )
    parser.add_argument(
        "--logger", type=str, choices=["tensorboard", "mlflow"], default="mlflow"
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
