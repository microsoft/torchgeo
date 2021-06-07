from typing import Any, Tuple

import torch
from torch.utils.data import ConcatDataset, random_split
from torchvision.datasets import VisionDataset

from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .landcoverai import LandCoverAI
from .nwpu import VHR10

__all__ = ("CV4AKenyaCropType", "LandCoverAI", "VHR10")


def get_datasets(
    dataset: str,
    root: str = "data",
    seed: int = 0,
) -> Tuple[VisionDataset, VisionDataset]:
    """Initialize train and test datasets.

    Parameters:
        dataset: name of dataset
        root: root directory containing data
        seed: random seed for reproducible results

    Returns:
        train and test datasets

    Raises:
        ValueError: if ``dataset`` is not a supported dataset
    """
    if dataset == "cv4akenyacroptype":
        pass
    elif dataset == "landcoverai":
        pass
    elif dataset == "vhr10":
        # Get both splits
        positive_dataset = VHR10(root, split="positive", download=True)
        negative_dataset = VHR10(root, split="negative", download=True)

        # Combine the splits
        concat_dataset: ConcatDataset[Tuple[Any, Any]] = ConcatDataset(
            [positive_dataset, negative_dataset]
        )

        # Randomly split dataset
        train_dataset, test_dataset = random_split(
            concat_dataset, [600, 200], generator=torch.Generator().manual_seed(seed)
        )
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'")

    return train_dataset, test_dataset
