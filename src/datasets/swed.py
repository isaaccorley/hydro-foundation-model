import os

from typing import Optional
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import percentile_normalization

from .transforms import Denormalize
from .utils import get_fraction_dataset


class SWED(NonGeoDataset):
    all_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    filter_filename = "train_nodata.csv"
    classes = ["Background", "Water"]
    splits = ["train", "test"]
    band_sets = ("all", "rgb")

    def __init__(self, root, split="train", bands="all", transforms=None):
        assert split in self.splits
        assert bands in self.band_sets
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.load_files()

    def load_files(self):
        with open(os.path.join(self.root, self.filter_filename), "r") as f:
            invalid = f.read().strip().splitlines()

        self.filenames = []
        img_root = os.path.join(self.root, self.split, "images")
        mask_root = os.path.join(self.root, self.split, "labels")
        for fn in sorted(os.listdir(img_root)):
            image_fn = os.path.join(img_root, fn)
            if self.split == "train":
                mask_fn = os.path.join(mask_root, fn.replace("_image_", "_chip_"))
                if os.path.basename(mask_fn) in invalid:
                    continue
            else:
                mask_fn = os.path.join(mask_root, fn.replace("_image_", "_label_"))
            self.filenames.append((image_fn, mask_fn))

    def load_image(self, path):
        if self.split == "train":
            image = np.load(path).transpose(2, 0, 1)
        else:
            with rasterio.open(path) as f:
                image = f.read().astype(np.int16)
        image = torch.from_numpy(image).float()
        return image

    def load_mask(self, path):
        if self.split == "train":
            mask = np.load(path).squeeze()
        else:
            with rasterio.open(path) as f:
                mask = f.read().squeeze().astype(np.int16)
        mask = torch.from_numpy(mask).long()
        mask = torch.clip(mask, -1, 1)
        return mask

    def __getitem__(self, idx):
        image_fn, mask_fn = self.filenames[idx]
        image = self.load_image(image_fn)
        mask = self.load_mask(mask_fn)

        if self.bands == "rgb":
            image = image[(3, 2, 1), ...]

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.filenames)

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


class SWEDDataModule(NonGeoDataModule):
    means = (
        torch.tensor(
            [
                560.0963,
                669.8031,
                938.8026,
                1104.3842,
                1374.6296,
                1826.4390,
                2012.0269,
                2095.9023,
                2159.6445,
                2191.1631,
                2105.7415,
                1568.9823,
            ]
        )
    )
    stds = (
        torch.tensor(
            [
                678.8931,
                748.4851,
                918.1321,
                1278.0764,
                1362.1965,
                1479.4902,
                1598.6714,
                1661.6722,
                1692.9138,
                1803.0081,
                1924.1908,
                1635.6689,
            ]
        )
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
        super().__init__(SWED, batch_size, num_workers, **kwargs)

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
            data_keys=None,
        )
        self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=None,
        )
        if image_size != 256:
            self.val_aug.append(K.Resize((image_size, image_size)))
            self.test_aug.append(K.Resize((image_size, image_size)))

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
