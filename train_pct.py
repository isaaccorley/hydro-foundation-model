import argparse

import lightning
import mlflow  # noqa: F401
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import OmegaConf

import src  # noqa: F401

torch.set_float32_matmul_precision("medium")


def main(args):
    lightning.pytorch.seed_everything(args.seed)
    config = OmegaConf.load(args.config)
    module = instantiate(config.module)

    if args.root is None:
        datamodule = instantiate(config.datamodule)
    else:
        datamodule = instantiate(config.datamodule, root=args.root)

    if args.logger == "mlflow":
        run_name = (
            "swinv2_hydro" if config.module.model == "hydro" else "swinv2_imagenet"
        )
        run_name = f"{run_name}_{int(args.limit_train_batches * 100)}"
        logger = MLFlowLogger(
            experiment_name=config.experiment_name, run_name=run_name, log_model=False
        )
        hparams = dict(config)
        hparams["train_pct"] = args.limit_train_batches
        logger.log_hyperparams(hparams)
    else:
        logger = TensorBoardLogger(
            save_dir="lightning_logs", name=config.experiment_name
        )

    callbacks = [LearningRateMonitor(logging_interval="step")]
    devices = [args.device] if args.device is not None else config.trainer.devices
    trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=devices,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
    )
    trainer.fit(module, datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")


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
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=0.01)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
