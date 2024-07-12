import argparse
import os

import lightning  # noqa: F401
import mlflow  # noqa: F401
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import OmegaConf

import src  # noqa: F401


lightning.pytorch.seed_everything(0)
torch.set_float32_matmul_precision("medium")


def main(args):
    config = OmegaConf.load(args.config)
    module = instantiate(config.module)

    if args.root is None:
        datamodule = instantiate(config.datamodule)
    else:
        datamodule = instantiate(config.datamodule, root=args.root)

    if args.logger == "mlflow":
        logger = MLFlowLogger(
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", config.experiment_name
            ),
            run_id=os.environ.get("MLFLOW_RUN_ID", None),
        )
        logger.log_hyperparams(dict(config))
    else:
        logger = TensorBoardLogger(
            save_dir="lightning_logs", name=config.experiment_name
        )

    callbacks = [LearningRateMonitor(logging_interval="step")]
    devices = [args.device] if args.device is not None else config.trainer.devices
    trainer = instantiate(
        config.trainer, logger=logger, callbacks=callbacks, devices=devices
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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
    args = parser.parse_args()
    main(args)
