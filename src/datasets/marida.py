import os
import json
import glob

import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import kornia.augmentation as K
from matplotlib.colors import ListedColormap
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import percentile_normalization
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.geo import NonGeoDataModule


class MARIDA(NonGeoDataset):
    classes = [
        "Marine Debris"
        "Dense Sargassum"
        "Sparse Sargassum"
        "Natural Organic Material"
        "Ship"
        "Clouds"
        "Marine Water"
        "Sediment-Laden Water"
        "Foam"
        "Turbid Water"
        "Shallow Water"
        "Waves"
        "Cloud Shadows"
        "Wakes"
        "Mixed Water"
    ]
    confidences = ["High", "Moderate", "Low"]
    debris_existences = ["Very close", "Away", "No"]

    cmap = ListedColormap(
        np.array([
            (255, 0, 0),
            (0, 128, 0),
            (50, 205, 50),
            (165, 42, 42),
            (255, 165, 0),
            (192, 192, 192),
            (0, 0, 128),
            (255, 215, 0),
            (128, 0, 128),
            (189, 183, 107),
            (0, 206, 209),
            (255, 245, 238),
            (128, 128, 128),
            (255, 255, 0),
            (188, 143, 143)
        ]) / 255.0
    )

    conf_cmap = ListedColormap(
        np.array([
            (255, 0, 0),
            (0, 128, 0),
            (50, 205, 50),
        ]) / 255.0
    )
    splits = {
        "train": os.path.join("splits", "train_X.txt"),
        "val": os.path.join("splits", "val_X.txt"),
        "test": os.path.join("splits", "test_X.txt")
    }
    multilabel_mapping_filename = "labels_mapping.txt"
    band_sets = ["all", "rgb"]

    def __init__(self, root, split="train", bands="all", transforms=None):
        assert split in self.splits
        assert bands in self.band_sets
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.load_files()

    def load_files(self):
        with open(os.path.join(self.root, self.multilabel_mapping_filename)) as f:
            self.multilabels = json.load(f)

        with open(os.path.join(self.root, self.splits[self.split])) as f:
            prefixes = [f"S2_{prefix}" for prefix in f.read().strip().splitlines()]

        self.masks = sorted(glob.glob(os.path.join(self.root, "patches", "**", "*_cl.tif")))
        self.masks = [mask for mask in self.masks if os.path.basename(mask).replace("_cl.tif", "") in prefixes]
        self.images = [mask.replace("_cl.tif", ".tif") for mask in self.masks]
        self.conf_masks = [mask.replace("_cl.tif", "_conf.tif") for mask in self.masks]

    def load_image(self, path):
        with rasterio.open(path) as f:
            image = f.read()
        image = torch.from_numpy(image).float()
        return image

    def load_mask(self, path):
        with rasterio.open(path) as f:
            mask = f.read().squeeze()
        mask = torch.from_numpy(mask).long()
        return mask

    def __getitem__(self, idx):
        image = self.load_image(self.images[idx])
        if self.bands == "rgb":
            image = image[(3, 2, 1), ...]

        mask = self.load_mask(self.masks[idx])
        # mask_conf = self.load_mask(self.conf_masks[idx])

        sample = {
            "image": image,
            "mask": mask,
            # "mask_conf": mask_conf
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def plot(self, sample, show_titles=True, suptitle=None, lower_pct=2, upper_pct=98):
        img = sample["image"][(3, 2, 1), ...] if self.bands == "all" else sample["image"]
        img = percentile_normalization(img.numpy(), lower=lower_pct, upper=upper_pct).transpose(1, 2, 0)
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
        axs[1].imshow(mask, vmin=0, vmax=self.cmap.N - 1, cmap=self.cmap, interpolation="none")
        if "prediction" in sample:
            axs[2].imshow(prediction, vmin=0, vmax=self.cmap.N - 1, cmap=self.cmap, interpolation="none")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Labels")
            if "prediction" in sample:
                axs[2].set_title("Predictions")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        return fig


class MARIDADataModule(NonGeoDataModule):
    # Stats from https://github.com/marine-debris/marine-debris.github.io/blob/main/semantic_segmentation/unet/dataloader.py
    class_weights = torch.tensor(
        [
            0.00452,
            0.00203,
            0.00254,
            0.00168,
            0.00766,
            0.15206,
            0.20232,
            0.35941,
            0.00109,
            0.20218,
            0.03226,
            0.00693,
            0.01322,
            0.01158,
            0.00052,
        ]
    )
    means = torch.tensor([
        0.05197577,
        0.04783991,
        0.04056812,
        0.03163572,
        0.02972606,
        0.03457443,
        0.03875053,
        0.03436435,
        0.0392113,
        0.02358126,
        0.01588816,
    ])
    stds = torch.tensor([
        0.04725893,
        0.04743808,
        0.04699043,
        0.04967381,
        0.04946782,
        0.06458357,
        0.07594915,
        0.07120246,
        0.08251058,
        0.05111466,
        0.03524419,
    ])

    def __init__(self, batch_size: int = 64, num_workers: int = 0, **kwargs) -> None:
        super().__init__(MARIDA, batch_size, num_workers, **kwargs)
        if "bands" in kwargs and kwargs["bands"] == "rgb":
            self.mean = self.means[(3, 2, 1)]
            self.std = self.stds[(3, 2, 1)]
        else:
            self.mean = self.means
            self.std = self.stds

        augmentations = [
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        ]

        self.train_aug = AugmentationSequential(
            *augmentations,
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )
