import os
import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import percentile_normalization
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def get_band(path):
    return int(path.split("_")[-2])


class MagicBathyNet(NonGeoDataset):
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
