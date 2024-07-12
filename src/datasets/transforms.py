import kornia.augmentation as K
import torch
import torch.nn as nn

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
