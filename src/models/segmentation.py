import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.datasets import unbind_samples
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    FBetaScore,
    JaccardIndex,
    Precision,
    Recall,
)
from torchmetrics.wrappers import ClasswiseWrapper


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, *args, image_size=256, tmax=50, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights: str = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]
        self.hparams["image_size"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "hydro":
            from .unet import SwinV2UNet

            self.model = SwinV2UNet(
                encoder="swinv2-hydro",
                encoder_weights=weights,
                classes=num_classes,
                in_channels=in_channels,
            )
        elif model.startswith("swin"):
            from .unet import SwinV2UNet

            self.model = SwinV2UNet(
                encoder=backbone,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=num_classes,
            )

        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'upernet'."
            )

        # Freeze backbone
        if self.hparams["freeze_backbone"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        # total_steps = self.trainer.estimated_stepping_batches
        # scheduler = OneCycleLR(optimizer, max_lr=self.hparams["lr"], total_steps=total_steps)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.tmax, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    def configure_losses(self) -> None:
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        class_weights = self.hparams["class_weights"]
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights)

        if loss == "ce":
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )
        elif loss == "jaccard":
            # JaccardLoss requires a list of classes to use instead of a class
            # index to ignore.
            classes = [
                i for i in range(self.hparams["num_classes"]) if i != ignore_index
            ]

            self.criterion = smp.losses.JaccardLoss(mode="multiclass", classes=classes)
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = self.hparams["ignore_index"]

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallPrecision": Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallRecall": Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "AverageAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AveragePrecision": Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AverageRecall": Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AverageF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AverageIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": ClasswiseWrapper(
                    Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                        ignore_index=ignore_index,
                    )
                ),
                "Precision": ClasswiseWrapper(
                    Precision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                        ignore_index=ignore_index,
                    )
                ),
                "Recall": ClasswiseWrapper(
                    Recall(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                        ignore_index=ignore_index,
                    )
                ),
                "F1Score": ClasswiseWrapper(
                    FBetaScore(
                        task="multiclass",
                        num_classes=num_classes,
                        beta=1.0,
                        average="none",
                        multidim_average="global",
                        ignore_index=ignore_index,
                    )
                ),
                "IoU": ClasswiseWrapper(
                    JaccardIndex(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                    )
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)
        self.train_metrics(y_hat_hard, y)
        self.log_dict({f"{k}": v for k, v in self.train_metrics.compute().items()})
        if batch_idx < 5:
            self.log_image(batch=batch, y_hat=y_hat, batch_idx=batch_idx, split="train")
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)

        self.val_metrics(y_hat_hard, y)
        self.log_dict(
            {f"{k}": v for k, v in self.val_metrics.compute().items()}, on_epoch=True
        )

        if batch_idx < 10:
            self.log_image(batch=batch, y_hat=y_hat, batch_idx=batch_idx, split="val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)

        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)
        self.test_metrics(y_hat_hard, y)
        self.log_dict(
            {f"{k}": v for k, v in self.test_metrics.compute().items()}, on_epoch=True
        )

        if batch_idx < 100:
            self.log_image(batch=batch, y_hat=y_hat, batch_idx=batch_idx, split="test")

    @torch.no_grad()
    def log_image(self, batch: Any, y_hat: torch.Tensor, batch_idx: int, split: str):
        if (
            hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and (
                hasattr(self.logger.experiment, "add_figure")
                or hasattr(self.logger.experiment, "log_figure")
            )
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()

            if hasattr(datamodule, "denormalize"):
                batch["image"] = datamodule.denormalize(batch["image"])

            sample = unbind_samples(batch)[0]

            fig = datamodule.plot(sample)

            if fig:
                for logger in self.loggers:
                    summary_writer = logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(
                            f"image/{split}/{batch_idx}",
                            fig,
                            global_step=self.global_step,
                        )
                    elif hasattr(summary_writer, "log_image"):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            filename = os.path.join(tmpdir, "image.png")
                            fig.savefig(filename, bbox_inches="tight")
                            image = mlflow.Image(filename)
                            summary_writer.log_image(
                                run_id=logger.run_id,
                                image=image,
                                key=f"{split}/predictions",
                                step=self.global_step,
                            )
                plt.close()
