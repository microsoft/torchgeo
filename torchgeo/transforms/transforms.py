# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]
from torch.nn.functional import pad

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


class PadTo(Module):  # type: ignore[misc,name-defined]
    """Pads the given sample to a specified size."""

    def __init__(
        self,
        size: Tuple[int, int],
        image_mode: str = "constant",
        image_value: Union[int, float] = 0,
        mask_mode: str = "constant",
        mask_value: Union[int, float] = 0,
    ) -> None:
        """Initialize a new transform instance.

        Args:
            size: a tuple of ints in the format (height, width) that give the spatial
                dimensions to pad inputs to
            image_mode: the type of padding to perform on the image (valid values
                are those accepted by torch.nn.functional.pad)
            image_value: fill value for 'constant' padding applied to the image
            mask_mode: the type of padding to perform on the mask (valid values
                are those accepted by torch.nn.functional.pad)
            mask_value: fill value for 'constant' padding applied to the mask
        """
        super().__init__()
        self.size = size
        self.image_mode = image_mode
        self.image_value = image_value
        self.mask_mode = mask_mode
        self.mask_value = mask_value

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Pad the inputs to a specifized size.

        The input will be padded along the bottom of the height dimension and along the
        right of the width dimension.

        Args:
            sample: a single data sample

        Returns:
            a sample where the spatial dimensions are padded to the desired sizes
        """
        if "image" in sample:
            _, height, width = sample["image"].shape

            height_pad = self.size[0] - height
            width_pad = self.size[1] - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode=self.image_mode,
                value=self.image_value,
            )

        if "masks" in sample:
            height, width = sample["masks"].shape

            height_pad = self.size[0] - height
            width_pad = self.size[1] - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["masks"] = pad(
                sample["masks"],
                (0, width_pad, 0, height_pad),
                mode=self.mask_mode,
                value=self.mask_value,
            )

            sample["masks"] = sample["masks"]

        return sample
