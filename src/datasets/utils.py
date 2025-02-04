from torch.utils.data import Subset, Dataset
import numpy as np
import torch.nn as nn
from typing import Any
import torchvision.transforms as T
from .transforms import DenormalizeTorchvision


def get_fraction_dataset(dataset: Dataset, fraction: float, seed: int = 42):
    """Returns a new dataset that is `fraction` size of the original dataset.

    Args:
        dataset (Dataset): The original dataset.
        fraction (float): Fraction of dataset to keep (e.g., 0.5 for 50%).
        seed (int): Random seed for reproducibility.

    Returns:
        Subset: A subset dataset with the given fraction of samples.
    """
    assert 0 < fraction <= 1, "Fraction must be in the range (0,1]."
    np.random.seed(seed)
    num_samples = int(len(dataset) * fraction)
    if num_samples == 0:
        num_samples = 1
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)


def get_inverse_normalize(transforms: list[Any]) -> nn.Module:
    """Find and compute the inverse normalization from a list of transforms."""
    # Find the T.Normalize transform in the list of transforms
    idx = None
    for i, t in enumerate(transforms):
        if isinstance(t, T.Normalize):
            idx = i
            break

    # If no normalize transform is used, use identity
    if idx is None:
        return nn.Identity()
    # Otherwise, define the inverse normalize fn
    else:
        norm = transforms[idx]
        mean, std = norm.mean, norm.std
        return DenormalizeTorchvision(mean, std)
