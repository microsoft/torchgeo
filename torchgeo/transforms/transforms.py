# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Dict, List, Union

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


class AugmentationSequential(Module):  # type: ignore[misc]
    """Wrapper around kornia AugmentationSequential to handle input dicts."""

    def __init__(self, *args: Module, data_keys: List[str]) -> None:
        """Initialize a new augmentation sequential instance.

        Args:
            *args: Sequence of kornia augmentations
            data_keys: List of inputs to augment (e.g. ["image", "mask", "boxes"])
        """
        super().__init__()
        self.data_keys = data_keys

        keys = []
        for key in data_keys:
            if key == "image":
                keys.append("input")
            elif key == "boxes":
                keys.append("bbox")
            else:
                keys.append(key)

        self.augs = K.AugmentationSequential(*args, data_keys=keys)

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Perform augmentations and update data dict.

        Args:
            sample: the input

        Returns:
            the augmented input
        """
        # Kornia augmentations require masks & boxes to be float
        if "mask" in self.data_keys:
            mask_dtype = sample["mask"].dtype
            sample["mask"] = sample["mask"].to(torch.float)  # type:ignore[attr-defined]
        if "boxes" in self.data_keys:
            boxes_dtype = sample["boxes"].dtype
            sample["boxes"] = sample["boxes"].to(
                torch.float  # type:ignore[attr-defined]
            )

        inputs = [sample[k] for k in self.data_keys]
        outputs_list: Union[Tensor, List[Tensor]] = self.augs(*inputs)
        outputs_list = (
            outputs_list if isinstance(outputs_list, list) else [outputs_list]
        )
        outputs: Dict[str, Tensor] = {
            k: v for k, v in zip(self.data_keys, outputs_list)
        }
        sample.update(outputs)

        # Convert masks & boxes to previous dtype
        if "mask" in self.data_keys:
            sample["mask"] = sample["mask"].to(mask_dtype)
        if "boxes" in self.data_keys:
            sample["boxes"] = sample["boxes"].to(boxes_dtype)

        return sample
