import kornia.augmentation as K
import torch
import torch.nn as nn
import torchvision.transforms as T

from typing import Sequence

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Denormalize(nn.Sequential):
    def __init__(self, transforms: K.AugmentationSequential):
        norms = [t for t in transforms if isinstance(t, K.Normalize)]
        denorms = [
            K.Denormalize(mean=norm.flags["mean"], std=norm.flags["std"])
            for norm in reversed(norms)
        ]
        super().__init__(*denorms)

    @torch.no_grad()
    def forward(self, x):
        return super().forward(x)


class DenormalizeTorchvision(nn.Sequential):
    def __init__(
        self, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
    ) -> None:
        self.mean = mean
        self.std = std
        self.inv_mean = torch.tensor([-x for x in mean])
        self.inv_std = torch.tensor([1 / x for x in std])
        super().__init__(
            T.Normalize(mean=torch.zeros(self.inv_mean.shape), std=self.inv_std),
            T.Normalize(mean=self.inv_mean, std=torch.ones(self.inv_std.shape)),
        )
