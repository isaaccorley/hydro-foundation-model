import os
import glob
import warnings

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import percentile_normalization
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from .transforms import Denormalize

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def get_band(path):
    return int(path.split("_")[-2])


class MADOS(NonGeoDataset):
    classes = [
        "Background",
        "Marine Debris",
        "Dense Sargassum",
        "Sparse Floating Algae",
        "Natural Organic Material",
        "Ship",
        "Oil Spill",
        "Marine Water",
        "Sediment-Laden Water",
        "Foam",
        "Turbid Water",
        "Shallow Water",
        "Waves & Wakes",
        "Oil Platform",
        "Jellyfish",
        "Sea snot",
    ]
    confidences = ["High", "Moderate", "Low"]
    debris_existences = ["Very close", "Away", "No"]

    cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0),
                (255, 0, 0),
                (0, 128, 0),
                (50, 205, 50),
                (165, 42, 42),
                (255, 165, 0),
                (216, 191, 216),
                (0, 0, 128),
                (255, 215, 0),
                (128, 0, 128),
                (189, 183, 107),
                (0, 206, 209),
                (255, 228, 196),
                (105, 105, 105),
                (255, 105, 180),
                (255, 255, 0),
            ]
        )
        / 255.0
    )

    conf_cmap = ListedColormap(
        np.array(
            [
                (255, 0, 0),
                (0, 128, 0),
                (50, 205, 50),
            ]
        )
        / 255.0
    )
    directory = "MADOS"
    splits = {
        "train": os.path.join("splits", "train_X.txt"),
        "val": os.path.join("splits", "val_X.txt"),
        "test": os.path.join("splits", "test_X.txt"),
    }
    all_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    )
    wavelengths = (
        "442",
        "492",
        "559",
        "665",
        "704",
        "739",
        "780",
        "833",
        "864",
        "1610",
        "2186",
    )
    resolutions = (
        "60",
        "10",
        "10",
        "10",
        "20",
        "20",
        "20",
        "10",
        "20",
        "20",
        "20",
    )
    band_sets = ("all", "rgb")
    image_size = (240, 240)

    def __init__(self, root, split="train", bands="all", transforms=None):
        assert split in self.splits
        assert bands in self.band_sets
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.ids = self.load_files()

        if bands == "all":
            self.band_indices = list(range(len(self.all_bands)))
        else:
            self.band_indices = (3, 2, 1)

    def load_files(self):
        with open(
            os.path.join(self.root, self.directory, self.splits[self.split])
        ) as f:
            ids = f.read().strip().splitlines()
            ids = [tuple(i.rsplit("_", 1)) for i in ids]
        return ids

    def load_image(self, index):
        scene, crop = self.ids[index]

        paths = glob.glob(
            os.path.join(
                self.root,
                self.directory,
                scene,
                "**",
                f"{scene}_*L2R_rhorc*_{crop}.tif",
            )
        )
        paths = sorted(paths, key=get_band)
        if self.bands == "rgb":
            paths = [paths[i] for i in (3, 2, 1)]

        images = []
        for path in paths:
            with rasterio.open(path) as f:
                image = f.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="float32",
                    resampling=Resampling.bilinear,
                )
                image = torch.from_numpy(image)
                images.append(image)
        return torch.stack(images, dim=0)

    def load_mask(self, index):
        scene, crop = self.ids[index]
        path = os.path.join(
            self.root, self.directory, scene, "10", f"{scene}_L2R_cl_{crop}.tif"
        )
        with rasterio.open(path) as f:
            mask = f.read().squeeze()
        mask = torch.from_numpy(mask).long()
        mask = mask - 1
        return mask

    def __getitem__(self, index):
        image = self.load_image(index)
        mask = self.load_mask(index)
        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def plot(self, sample, show_titles=True, suptitle=None, lower_pct=2, upper_pct=98):
        img = (
            sample["image"][(3, 2, 1), ...] if self.bands == "all" else sample["image"]
        )
        img = percentile_normalization(
            img.numpy(), lower=lower_pct, upper=upper_pct
        ).transpose(1, 2, 0)
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
        axs[1].imshow(
            mask, vmin=-1, vmax=self.cmap.N - 2, cmap=self.cmap, interpolation="none"
        )
        if "prediction" in sample:
            axs[2].imshow(
                prediction,
                vmin=0,
                vmax=self.cmap.N - 1,
                cmap=self.cmap,
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


class MADOSDataModule(NonGeoDataModule):
    # Stats from https://github.com/marine-debris/marine-debris.github.io/blob/main/semantic_segmentation/unet/dataloader.py
    class_weights = torch.tensor(
        [
            0.00336,
            0.00241,
            0.00336,
            0.00142,
            0.00775,
            0.18452,
            0.34775,
            0.20638,
            0.00062,
            0.1169,
            0.09188,
            0.01309,
            0.00917,
            0.00176,
            0.00963,
        ]
    )
    means = torch.tensor(
        [
            0.0582676,
            0.05223386,
            0.04381474,
            0.0357083,
            0.03412902,
            0.03680401,
            0.03999107,
            0.03566642,
            0.03965081,
            0.0267993,
            0.01978944,
        ]
    )
    stds = torch.tensor(
        [
            0.03240627,
            0.03432253,
            0.0354812,
            0.0375769,
            0.03785412,
            0.04992323,
            0.05884482,
            0.05545856,
            0.06423746,
            0.04211187,
            0.03019115,
        ]
    )

    def __init__(
        self,
        image_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(MADOS, batch_size, num_workers, **kwargs)
        if "bands" in kwargs and kwargs["bands"] == "rgb":
            self.mean = self.means[[3, 2, 1]]
            self.std = self.stds[[3, 2, 1]]
        else:
            self.mean = self.means
            self.std = self.stds

        self.image_size = image_size

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
            self.train_dataset = self.dataset_class(split="train", **self.kwargs)
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
