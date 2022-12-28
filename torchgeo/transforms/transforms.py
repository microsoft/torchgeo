# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Any, Dict, List, Optional, Tuple, Union

import kornia
import torch
from kornia.augmentation import GeometricAugmentationBase2D
from kornia.contrib import compute_padding, extract_tensor_patches
from torch import Tensor
from torch.nn.modules import Module


class AugmentationSequential(Module):
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

        self.augs = kornia.augmentation.AugmentationSequential(*args, data_keys=keys)

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
            sample["mask"] = sample["mask"].to(torch.float)
        if "boxes" in self.data_keys:
            boxes_dtype = sample["boxes"].dtype
            sample["boxes"] = sample["boxes"].to(torch.float)

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


# TODO: contribute these to Kornia
class _ExtractTensorPatches(GeometricAugmentationBase2D):  # type: ignore[misc]
    """Chop up a tensor into a grid."""

    def __init__(self, window_size: Union[int, Tuple[int, int]]) -> None:
        """Initialize a new _ExtractTensorPatches instance.

        Args:
            window_size: the size of each patch
        """
        super().__init__(p=1)
        self.flags = {"window_size": window_size}

    def compute_transformation(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        """Compute the transformation.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters

        Returns:
            the transformation
        """
        # TODO: this isn't correct, but we don't need it at the moment anyway
        return self.identity_matrix(input)

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        size = flags["window_size"]
        h, w = input.shape[-2:]
        padding = compute_padding((h, w), size)
        input = extract_tensor_patches(input, size, size, padding)
        input = torch.flatten(input, 0, 1)  # [B, N, C?, H, W] -> [B*N, C?, H, W]
        return input


class _RandomNCrop(GeometricAugmentationBase2D):  # type: ignore[misc]
    """Take N random crops of a tensor."""

    def __init__(self, size: Tuple[int, int], num: int) -> None:
        """Initialize a new _RandomNCrop instance.

        Args:
            size: desired output size (out_h, out_w) of the crop
            num: number of crops to take
        """
        super().__init__(p=1)
        self.flags = {"size": size, "num": num}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """

    def forward(self, sample):
        images, masks = [], []
        for i in range(self.num_patches_per_tile):
            crop = K.RandomCrop(self.patch_size, p=1.0)
            image = crop(sample["image"].squeeze(0))
            images.append(image.squeeze(0))
            if "mask" in sample:
                mask = crop(sample["mask"].float(), params=crop._params)
                masks.append(mask.squeeze().long())

        sample["image"] = torch.stack(images)
        if "mask" in sample:
            sample["mask"] = torch.stack(masks)
        return sample

    def n_random_crop(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get n random crops."""
        images, masks = [], []
        for _ in range(self.num_patches_per_tile):
            image, mask = sample["image"], sample["mask"]
            # RandomCrop needs image and mask to be in float
            mask = mask.to(torch.float)
            image, mask = self.random_crop(image, mask)
            images.append(image.squeeze())
            masks.append(mask.squeeze(0).long())
        sample["image"] = torch.stack(images)  # (t,c,h,w)
        sample["mask"] = torch.stack(masks)  # (t, 1, h, w)
        return sample
