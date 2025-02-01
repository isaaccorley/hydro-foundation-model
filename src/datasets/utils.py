from torch.utils.data import Subset, Dataset
import numpy as np


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
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)
