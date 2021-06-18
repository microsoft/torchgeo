from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


# TODO: figure out why mypy is angry:
# https://discuss.pytorch.org/t/how-to-correctly-annotate-subclasses-of-nn-module/74317/2
class RandomHorizontalFlip(nn.Module):  # type: ignore[misc,name-defined]
    """Horizontally flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.

        Parameters:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly flip the image and target tensors.

        Parameters:
            sample: a single data sample

        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            if "image" in sample:
                sample["image"] = F.hflip(sample["image"])
                width, height = F._get_image_size(sample["image"])

                if "boxes" in sample:
                    sample["boxes"][:, [0, 2]] = width - sample["boxes"][:, [2, 0]]
                if "masks" in sample:
                    sample["masks"] = sample["masks"].flip(-1)

        return sample


class RandomVerticalFlip(nn.Module):  # type: ignore[misc,name-defined]
    """Vertically flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.

        Parameters:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly flip the image and target tensors.

        Parameters:
            sample: a single data sample

        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            if "image" in sample:
                sample["image"] = F.vflip(sample["image"])
                width, height = F._get_image_size(sample["image"])

                if "boxes" in sample:
                    sample["boxes"][:, [1, 3]] = height - sample["boxes"][:, [3, 1]]
                if "masks" in sample:
                    sample["masks"] = sample["masks"].flip(-2)

        return sample


class Identity(nn.Module):  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Do nothing.

        Parameters:
            sample: the input

        Returns:
            the unchanged input
        """
        return sample
