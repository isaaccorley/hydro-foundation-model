import glob
import os

import lightning
import rasterio
import torch


class ShipS2AIS(torch.utils.data.Dataset):
    """
    The Ship-S2-AIS is a dataset of 140x140 Sentinel-2 L1C imagery for classifying if
    an image patch contains a ship or not.

    The dataset contains
        - 1,449 and 329 ship samples in the train/test sets
        - 10,000 and 2,400 negative samples in the train/test sets

    Results from ESA EO Week presentation indicate VGG-16 classifier gets:
        RGB: 0.940 F1 | 0.952 Rec. | 0.929 Prec. | 0.972 Acc
        MSI: 0.887 F1 | 0.844 Rec. | 0.935 Prec. | 0.950 Acc
    """

    directory = "Sentinel-2-database-for-ship-detection"
    splits = dict(train="train", test="test")
    classes = ["neg", "ships"]
    band_sets = ["all", "rgb"]
    bands_names = ["B02", "B03", "B04", "B11", "B12"]

    def __init__(self, root, split="train", bands="all", transforms=None):
        assert split in self.splits
        assert bands in self.band_sets
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.images, self.targets = self.load_files(root, split)

    def load_files(self, root, split):
        folder = os.path.join(root, self.directory, self.splits[split])
        images, targets = [], []
        for i, cls in enumerate(self.classes):
            imgs = sorted(glob.glob(os.path.join(folder, cls, "**", "*.tif")))
            tgts = [i] * len(imgs)
            images.extend(imgs)
            targets.extend(tgts)
        return images, targets

    def load_image(self, path):
        with rasterio.open(path) as f:
            x = f.read().astype("float32")
        x = torch.from_numpy(x)
        return x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, label = self.images[index], self.targets[index]
        image = self.load_image(path)
        label = torch.tensor(label)

        if self.bands == "rgb":
            image = image[[2, 1, 0]]

        if self.transforms is not None:
            image = self.transforms(image)

        return dict(image=image, label=label)


class ShipS2AISDataModule(lightning.LightningDataModule):
    def __init__(
        self, root, bands="all", transforms=None, batch_size=32, num_workers=8, seed=0
    ):
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        self.train_dataset = ShipS2AIS(
            root=self.root,
            split="train",
            bands=self.bands,
            transforms=self.transforms,
        )
        self.test_dataset = ShipS2AIS(
            root=self.root,
            split="test",
            bands=self.bands,
            transforms=self.transforms
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
