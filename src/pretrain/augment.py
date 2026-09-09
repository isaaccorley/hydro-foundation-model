"""DINO multi-crop augmentation, shared between RGB (PIL) and MSI (tensor) inputs.

Photometric jitter (color jitter / grayscale / blur / solarize) is applied only for RGB, where
it's meaningful. MSI uses geometric-only augmentation on the already-normalized 12-band tensor:
color jitter developed for natural RGB has no clean analogue for reflectance bands.

Vertical flips are on for both: satellite imagery has no canonical "up".
"""

import random

import torch
import torchvision.transforms as T
from PIL import ImageFilter, ImageOps


class GaussianBlur:
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.p, self.radius_min, self.radius_max = p, radius_min, radius_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


def rgb_photometric() -> T.Compose:
    return T.Compose(
        [
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            Solarization(p=0.2),
        ]
    )


class MultiCropTransform:
    """Returns `n_global + n_local` views of one image (global views first)."""

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 6,
        global_size: int = 256,
        local_size: int = 96,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.4),
        photometric=None,
        finalize=None,
    ) -> None:
        self.n_global, self.n_local = n_global, n_local
        bicubic = T.InterpolationMode.BICUBIC
        self.global_crop = T.RandomResizedCrop(global_size, scale=global_scale, interpolation=bicubic, antialias=True)
        self.local_crop = T.RandomResizedCrop(local_size, scale=local_scale, interpolation=bicubic, antialias=True)
        self.flip = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
        self.photometric = photometric
        self.finalize = finalize

    def _view(self, img, crop):
        x = self.flip(crop(img))
        if self.photometric is not None:
            x = self.photometric(x)
        if self.finalize is not None:
            x = self.finalize(x)
        return x

    def __call__(self, img) -> list[torch.Tensor]:
        views = [self._view(img, self.global_crop) for _ in range(self.n_global)]
        views += [self._view(img, self.local_crop) for _ in range(self.n_local)]
        return views


def multicrop_collate(batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """list of per-sample view lists -> list of per-view batches, each (B, C, H, W)."""
    n_views = len(batch[0])
    return [torch.stack([sample[i] for sample in batch]) for i in range(n_views)]
