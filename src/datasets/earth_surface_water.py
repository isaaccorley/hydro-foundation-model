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


class PadMissingBands:
    def __call__(self, sample):
        h, w = sample["image"].shape[1:]
        zero_band = torch.zeros((h, w), dtype=torch.float)
        B01 = zero_band
        B02 = sample["image"][0]
        B03 = sample["image"][1]
        B04 = sample["image"][2]
        B05 = zero_band
        B06 = zero_band
        B07 = zero_band
        B08 = sample["image"][3]
        B8A = zero_band
        B09 = zero_band
        B11 = sample["image"][4]
        B12 = sample["image"][5]

        sample["image"] = torch.stack(
            [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12], dim=0
        )
        return sample


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

    PAD_SHAPE = (1536, 1792)

    def __init__(
        self,
        root,
        split="train",
        bands="all",
        transforms=None,
        pad_sizes=False,
        pad_bands=False,
        use_patched_version=True,
    ):
        assert split in self.splits
        assert bands in self.band_sets
        if use_patched_version:
            assert not pad_sizes, "Patched version doesn't support pad_sizes"
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.pad_sizes = pad_sizes
        self.pad_bands = pad_bands
        self.pad_bands_transform = PadMissingBands()
        self.use_patched_version = use_patched_version
        self.filenames = []

        self.load_files()

    def load_files(self):
        split_dir = f"{self.split_to_directory[self.split]}{'-patched' if self.use_patched_version else ''}"
        img_root = os.path.join(self.root, f"{split_dir}_scene")
        mask_root = os.path.join(self.root, f"{split_dir}_truth")

        for fn in sorted(os.listdir(img_root)):
            image_fn = os.path.join(img_root, fn)
            if self.use_patched_version:
                mask_fn = os.path.join(mask_root, fn.replace("image", "mask"))
            else:
                parts = fn[:-4].split("_")
                idx = "_".join(parts[:-2])
                mask_fn = os.path.join(mask_root, f"{idx}_{parts[-1]}_Truth.tif")
            self.filenames.append((image_fn, mask_fn))

    def load_image(self, path):
        with rasterio.open(path) as f:
            image = f.read()
        image = torch.from_numpy(image)
        if self.pad_sizes:
            pad = torch.zeros((6, self.PAD_SHAPE[0], self.PAD_SHAPE[1])).float()
            pad[:, : image.shape[1], : image.shape[2]] = image
            return pad
        else:
            return image.float()

    def load_mask(self, path):
        with rasterio.open(path) as f:
            mask = f.read().squeeze()
        mask = torch.from_numpy(mask)
        if self.pad_sizes:
            pad = torch.zeros(self.PAD_SHAPE).long()
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

        if self.bands != "rgb" and self.pad_bands:
            sample = self.pad_bands_transform(sample)

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
    """
    means = (
        torch.tensor([771.4490, 989.0422, 975.8994, 2221.6182, 1854.8079, 1328.8887])
        / 10000.0
    )
    stds = (
        torch.tensor([738.8903, 812.4620, 1000.6935, 1314.1964, 1384.8275, 1225.1549])
        / 10000.0
    )
    """

    means = (
        torch.tensor(
            [
                340.76769064,
                429.9430203,
                614.21682446,
                590.23569706,
                950.68368468,
                1792.46290469,
                2075.46795189,
                2218.94553375,
                2266.46036911,
                2246.0605464,
                1594.42694882,
                1009.32729131,
            ]
        )
    )
    stds = (
        torch.tensor(
            [
                554.81258967,
                572.41639287,
                582.87945694,
                675.88746967,
                729.89827633,
                1096.01480586,
                1273.45393088,
                1365.45589904,
                1356.13789355,
                1302.3292881,
                1079.19066363,
                818.86747235,
            ]
        )
    )

    padded_means = (
        torch.tensor(
            [
                0.0,
                771.4490,
                989.0422,
                975.8994,
                0.0,
                0.0,
                0.0,
                2221.6182,
                0.0,
                0.0,
                1854.8079,
                1328.8887,
            ]
        )
    )
    padded_stds = (
        torch.tensor(
            [
                10000.0,
                738.8903,
                812.4620,
                1000.6935,
                10000.0,
                10000.0,
                10000.0,
                1314.1964,
                10000.0,
                10000.0,
                1384.8275,
                1225.1549,
            ]
        )
    )

    def __init__(
        self,
        image_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        means: Optional[list[float]] = None,
        stds: Optional[list[float]] = None,
        train_fraction: Optional[float] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(EarthSurfaceWater, batch_size, num_workers, **kwargs)

        if "bands" in kwargs and kwargs["bands"] == "rgb":
            self.mean = self.means[[3, 2, 1]]
            self.std = self.stds[[3, 2, 1]]
        else:
            if "pad_bands" in kwargs and kwargs["pad_bands"]:
                self.mean = self.padded_means
                self.std = self.padded_stds
            else:
                self.mean = self.means
                self.std = self.stds

        if means is not None:
            self.mean = torch.tensor(means)
        if stds is not None:
            self.std = torch.tensor(stds)

        self.image_size = image_size
        self.train_fraction = train_fraction
        self.seed = seed

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.8, 1.2), ratio=(1, 1), p=1.0
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
        )
        self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((image_size, image_size)),
            data_keys=None,
        )
        self.test_aug = K.AugmentationSequential(
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
            print("Train set length:", len(ds))
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
