# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


class RandomHorizontalFlip(Module):  # type: ignore[misc,name-defined]
    """Horizontally flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.

        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.

        Args:
            sample: a single data sample

        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            if "image" in sample:
                sample["image"] = sample["image"].flip(-1)

                if "boxes" in sample:
                    height, width = sample["image"].shape[-2:]
                    sample["boxes"][:, [0, 2]] = width - sample["boxes"][:, [2, 0]]

            if "masks" in sample:
                sample["masks"] = sample["masks"].flip(-1)

        return sample


class RandomVerticalFlip(Module):  # type: ignore[misc,name-defined]
    """Vertically flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.

        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.

        Args:
            sample: a single data sample

        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            if "image" in sample:
                sample["image"] = sample["image"].flip(-2)

                if "boxes" in sample:
                    height, width = sample["image"].shape[-2:]
                    sample["boxes"][:, [1, 3]] = height - sample["boxes"][:, [3, 1]]

            if "masks" in sample:
                sample["masks"] = sample["masks"].flip(-2)

        return sample


class Identity(Module):  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Do nothing.

        Args:
            sample: the input

        Returns:
            the unchanged input
        """
        return sample
