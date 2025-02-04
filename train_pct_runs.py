import argparse
import gc

import pandas as pd
import lightning
import mlflow  # noqa: F401
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import OmegaConf

import src  # noqa: F401

torch.set_float32_matmul_precision("medium")


def main(args):
    metrics = []
    for i, frac in enumerate(args.train_fractions):
        for seed in range(args.num_seed_runs):
            print(f"Train Fraction: {frac} ({i+1}/{len(args.train_fractions)} - Repeat: {seed+1}/{args.num_seed_runs})")
            lightning.pytorch.seed_everything(seed)
            config = OmegaConf.load(args.config)
            module = instantiate(config.module, tmax=None)

            dm_kwargs = dict(seed=seed, train_fraction=frac)
            if args.root is not None:
                dm_kwargs["root"] = args.root
            if args.batch_size is not None:
                dm_kwargs["batch_size"] = args.batch_size

            datamodule = instantiate(config.datamodule, **dm_kwargs)

            logger = None
            if args.logger == "tensorboard":
                name = f"{config.experiment_name}-{frac}-{seed}"
                logger = TensorBoardLogger(
                    save_dir="lightning_logs", name=name
                )

            callbacks = [LearningRateMonitor(logging_interval="step")]
            if args.save_last_weights:
                callbacks.append(ModelCheckpoint(save_last=True))
            devices = [args.device] if args.device is not None else config.trainer.devices
            trainer = instantiate(
                config.trainer,
                logger=logger,
                callbacks=callbacks,
                devices=devices,
                max_epochs=args.max_epochs,
                enable_checkpointing=False,
                check_val_every_n_epoch=10,
            )
            trainer.fit(module, datamodule)
            run_metrics = trainer.test(datamodule=datamodule, ckpt_path="last")  # uses the last model
            run_metrics = run_metrics[0]
            run_metrics["train_fraction"] = frac
            run_metrics["seed"] = seed
            metrics.append(run_metrics)
            del module, datamodule, trainer
            torch.cuda.empty_cache()
            gc.collect()

    metrics = pd.DataFrame.from_dict(metrics)
    metrics.to_csv(args.output)


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
        "--logger", type=str, choices=["tensorboard"], default=None
    )
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--train_fractions",
        type=float,
        nargs="+",
        default=[0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.005, 0.01],
    )
    parser.add_argument("--num_seed_runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="metrics.csv")
    parser.add_argument("--save_last_weights", action="store_true")
    args = parser.parse_args()
    main(args)
