# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Dict, Tuple

import torch
from einops import rearrange
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


class ExtractPatches(Module):  # type: ignore[misc,name-defined]
    """Extract patches from single image/mask."""

    def __init__(self, patch_size: Tuple[int, int]) -> None:
        """Initialize a new transform instance.

        Args:
            patch_size: a tuple of the (height, width) size of patches
        """
        super().__init__()
        self.ph, self.pw = patch_size

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform the image from image->patches or mask if they exist.

        Args:
            sample: the input

        Returns:
            the patched input
        """
        if "image" in sample:
            sample["image"] = rearrange(
                sample["image"],
                "b c (h ph) (w pw) -> (b h w) c ph pw",
                ph=self.ph, pw=self.pw
            )
        if "mask" in sample:
            if sample["mask"].ndim == 4:
                sample["mask"] = rearrange(
                    sample["mask"],
                    "b c (h ph) (w pw) -> (b h w) c ph pw",
                    ph=self.ph, pw=self.pw
                )
            elif sample["mask"].ndim == 3:
                sample["mask"] = rearrange(
                    sample["mask"],
                    "b (h ph) (w pw) -> (b h w) ph pw",
                    ph=self.ph, pw=self.pw
                )
            else:
                raise ValueError(
                    f"Expected 3/4 dimensional mask but got {sample['mask'].ndim}"
                )

        return sample


class CombinePatches(Module):  # type: ignore[misc,name-defined]
    """Combine patches to single image/mask."""

    def __init__(self, image_size: Tuple[int, int]) -> None:
        """Initialize a new transform instance.

        Args:
            image_size: a tuple of the (height, width) of the original image
        """
        super().__init__()
        self.h, self.w = image_size

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform the image from patches->image or mask if they exist.

        Args:
            sample: the input

        Returns:
            the combined input
        """
        if "image" in sample:
            ph, pw = sample["image"].shape[-2:]

            sample["image"] = rearrange(
                sample["image"],
                "(b h w) c ph pw -> b c (h ph) (w pw)",
                h=self.h // ph,
                w=self.w // pw,
            )
        if "mask" in sample:
            ph, pw = sample["mask"].shape[-2:]

            if sample["mask"].ndim == 4:
                sample["mask"] = rearrange(
                    sample["mask"],
                    "(b h w) c ph pw -> b c (h ph) (w pw)",
                    h=self.h // ph,
                    w=self.w // pw,
                )
            elif sample["mask"].ndim == 3:
                sample["mask"] = rearrange(
                    sample["mask"],
                    "(b h w) ph pw -> b (h ph) (w pw)",
                    h=self.h // ph,
                    w=self.w // pw,
                )
            else:
                raise ValueError(
                    f"Expected 3/4 dimensional mask but got {sample['mask'].ndim}"
                )

        return sample
