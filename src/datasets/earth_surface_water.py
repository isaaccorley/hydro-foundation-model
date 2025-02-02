import os
import logging
from typing import Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules.geo import NonGeoDataModule

from .transforms import Denormalize
from .utils import get_fraction_dataset

logging.getLogger("rasterio._env").setLevel(logging.ERROR)


class EarthSurfaceWater(NonGeoDataset):
    """Earth Surface Water dataset.

    The Earth Surface Water dataset is a dataset of Sentinel-2 images and water masks.

    Described in: https://www.sciencedirect.com/science/article/pii/S0303243421001793
    Code/data download link: https://github.com/xinluo2018/WatNet/tree/main
    """

    all_bands = [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
        "B12",
    ]
    directory = "dset-s2"
    classes = ["Background", "Water"]
    splits = ["train", "test"]
    split_to_directory = {
        "train": "tra",
        "test": "val",
    }
    band_sets = ("all", "rgb")

    def __init__(
        self, root, split="train", bands="all", transforms=None, pad_sizes=True
    ):
        assert split in self.splits
        assert bands in self.band_sets
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.pad_sizes = pad_sizes
        self.load_files()

    def load_files(self):
        self.filenames = []
        img_root = os.path.join(
            self.root, self.directory, f"{self.split_to_directory[self.split]}_scene"
        )
        mask_root = os.path.join(
            self.root, self.directory, f"{self.split_to_directory[self.split]}_truth"
        )
        for fn in sorted(os.listdir(img_root)):
            image_fn = os.path.join(img_root, fn)
            parts = fn[:-4].split("_")

            idx = "_".join(parts[:-2])
            mask_fn = os.path.join(mask_root, f"{idx}_{parts[-1]}_Truth.tif")
            self.filenames.append((image_fn, mask_fn))

    def load_image(self, path):
        with rasterio.open(path) as f:
            image = f.read()
        image = torch.from_numpy(image)
        if self.pad_sizes:
            pad = torch.zeros((6, 1440, 1568)).float()
            pad[:, : image.shape[1], : image.shape[2]] = image
            return pad
        else:
            return image.float()

    def load_mask(self, path):
        with rasterio.open(path) as f:
            mask = f.read().squeeze()
        mask = torch.from_numpy(mask)
        if self.pad_sizes:
            pad = torch.zeros((1440, 1568)).long()
            pad[: mask.shape[0], : mask.shape[1]] = mask
            return pad
        else:
            return mask.long()

    def __getitem__(self, idx):
        image_fn, mask_fn = self.filenames[idx]
        image = self.load_image(image_fn)
        mask = self.load_mask(mask_fn)

        if self.bands == "rgb":
            image = image[(2, 1, 0), ...]

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.filenames)

    def plot(self, sample, show_titles=True):
        img = (
            sample["image"][(2, 1, 0), ...] if self.bands == "all" else sample["image"]
        )
        img = img.numpy().transpose(1, 2, 0)
        img = np.clip(img / 3000, 0, 1)
        mask = sample["mask"].numpy()

        if "prediction" in sample:
            n_cols = 3
            width = 15
            prediction = sample["prediction"].numpy()
        else:
            n_cols = 2
            width = 10

        fig, axs = plt.subplots(1, n_cols, figsize=(width, 5))
        axs[0].imshow(img)
        axs[1].imshow(mask, vmin=0, vmax=1, interpolation="none")
        if "prediction" in sample:
            axs[2].imshow(
                prediction,
                vmin=0,
                vmax=1,
                interpolation="none",
            )
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Labels")
            if "prediction" in sample:
                axs[2].set_title("Predictions")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        return fig


class EarthSurfaceWaterDataModule(NonGeoDataModule):
    means = (
        torch.tensor([771.4490, 989.0422, 975.8994, 2221.6182, 1854.8079, 1328.8887])
        / 10000.0
    )
    stds = (
        torch.tensor([738.8903, 812.4620, 1000.6935, 1314.1964, 1384.8275, 1225.1549])
        / 10000.0
    )

    def __init__(
        self,
        image_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        train_fraction: Optional[float] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(EarthSurfaceWater, batch_size, num_workers, **kwargs)

        if "bands" in kwargs and kwargs["bands"] == "rgb":
            self.mean = self.means[[3, 2, 1]]
            self.std = self.stds[[3, 2, 1]]
        else:
            self.mean = self.means
            self.std = self.stds

        self.image_size = image_size
        self.train_fraction = train_fraction
        self.seed = seed

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=0.0, std=10000.0),
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.8, 1.2), ratio=(1, 1), p=1.0
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
        )
        self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=0.0, std=10000.0),
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((image_size, image_size)),
            data_keys=None,
        )
        self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=0.0, std=10000.0),
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((image_size, image_size)),
            data_keys=None,
        )

        self.denormalize = Denormalize(self.train_aug)

    def setup(self, stage=None):
        if stage in ["fit"]:
            ds = self.dataset_class(split="train", **self.kwargs)
            if self.train_fraction is not None:
                ds = get_fraction_dataset(ds, self.train_fraction, self.seed)
            self.train_dataset = ds
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(split="test", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(split="test", **self.kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Hack because kornia doesn't work with int masks yet (only float)
        batch["mask"] = batch["mask"].float()
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        batch["mask"] = batch["mask"].long().squeeze(dim=1)
        return batch
