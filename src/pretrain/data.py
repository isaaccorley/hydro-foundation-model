"""Datasets for DINO pretraining: RGB quicklooks or 12-band MSI tiles, each -> multi-crop views."""

import glob
import os

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.pretrain.augment import MultiCropTransform
from src.pretrain.stats import MAD_STD, MEAN, MEDIAN, QUANTIFICATION, STD, detect_boa_offset

NORM_STATS = {"robust": (MEDIAN, MAD_STD), "plain": (MEAN, STD)}


class RGBHydroDataset(Dataset):
    """Reads precomputed RGB quicklook jpgs (see hydroloc's `scripts/make_jpg.py`)."""

    def __init__(self, root: str, transform: MultiCropTransform, ext: str = ".jpg") -> None:
        self.paths = sorted(glob.glob(os.path.join(root, f"*{ext}")))
        if not self.paths:
            raise FileNotFoundError(f"no {ext} files found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class MSIHydroDataset(Dataset):
    """Reads Sentinel-2 L2A tifs -> per-patch BOA offset removal -> reflectance -> z-score.

    Zero DN is nodata: excluded from offset detection and set to 0 after normalization.
    Normalization is per-pixel affine so it commutes with the crop/resize/flip in the
    multi-crop transform; doing it once here before cropping is equivalent and cheaper.
    """

    def __init__(
        self,
        root: str,
        transform: MultiCropTransform,
        band_indices: tuple[int, ...] = tuple(range(12)),
        norm: str = "robust",
        clamp: tuple[float, float] | None = (-5.0, 15.0),
        ext: str = ".tif",
    ) -> None:
        self.paths = sorted(glob.glob(os.path.join(root, f"*{ext}")))
        if not self.paths:
            raise FileNotFoundError(f"no {ext} files found in {root}")
        self.transform = transform
        self.band_indices = list(band_indices)
        center, scale = NORM_STATS[norm]
        self.mean = center[self.band_indices].view(-1, 1, 1)
        self.std = scale[self.band_indices].view(-1, 1, 1)
        self.clamp = clamp

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        with rasterio.open(self.paths[idx]) as f:
            dn = f.read(indexes=[i + 1 for i in self.band_indices]).astype(np.float32)
        valid = torch.from_numpy(dn != 0)
        offset = detect_boa_offset(dn)
        x = (torch.from_numpy(dn) - offset) / QUANTIFICATION
        x = (x - self.mean) / self.std
        if self.clamp is not None:
            x = x.clamp(*self.clamp)
        x = torch.where(valid, x, torch.zeros_like(x))
        return self.transform(x)
