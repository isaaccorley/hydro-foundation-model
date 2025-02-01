import glob
import os

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt

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

    def __len__(self):
        return len(self.images)

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

    def __getitem__(self, index):
        path, label = self.images[index], self.targets[index]
        image = self.load_image(path)
        label = torch.tensor(label)

        if self.bands == "rgb":
            image = image[[2, 1, 0]]

        if self.transforms is not None:
            image = self.transforms(image)

        return dict(image=image, label=label)

    def plot(self, sample):
        image, label = sample["image"], sample["label"]
        if self.bands != "rgb":
            image = image[[2, 1, 0]]
        image = image.numpy().transpose(1, 2, 0)
        fig = plt.figure()
        plt.imshow(np.clip(image / 3000, 0, 1))
        plt.title(self.classes[label])
        plt.tight_layout()
        return fig
