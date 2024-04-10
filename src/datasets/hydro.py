import os
import glob

import torch
import rasterio
import numpy as np


class HydroDataset(torch.utils.data.Dataset):
    BAND_SETS = ("rgb", "all")

    def __init__(self, root, bands, transforms=None, ext=".tif"):
        assert bands in self.BAND_SETS
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.images = glob.glob(os.path.join(root, f"*{ext}"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with rasterio.open(path) as f:
            x = f.read().astype(np.float32)
        x = torch.from_numpy(x).to(torch.float)

        if self.bands == "rgb":
            x = x[[3, 2, 1], ...]

        if self.transforms is not None:
            x = self.transforms(x)

        return dict(image=x)
