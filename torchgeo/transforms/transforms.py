# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Any, Dict, List, Optional, Tuple, Union

import kornia
import torch
from kornia.augmentation import GeometricAugmentationBase2D
from kornia.augmentation.random_generator import CropGenerator
from kornia.contrib import compute_padding, extract_tensor_patches
from kornia.geometry import crop_by_indices
from torch import Tensor
from torch.nn.modules import Module


# TODO: contribute these to Kornia and delete this file
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


class _ExtractTensorPatches(GeometricAugmentationBase2D):
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
        out: Tensor = self.identity_matrix(input)
        return out

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


class _RandomNCrop(GeometricAugmentationBase2D):
    """Take N random crops of a tensor."""

    def __init__(self, size: Tuple[int, int], num: int) -> None:
        """Initialize a new _RandomNCrop instance.

        Args:
            size: desired output size (out_h, out_w) of the crop
            num: number of crops to take
        """
        super().__init__(p=1)
        self._param_generator: _NCropGenerator = _NCropGenerator(size, num)
        self.flags = {"size": size, "num": num}

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
        out: Tensor = self.identity_matrix(input)
        return out

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
        out = []
        for i in range(flags["num"]):
            out.append(crop_by_indices(input, params["src"][i], flags["size"]))
        return torch.cat(out)


class _NCropGenerator(CropGenerator):
    """Generate N random crops."""

    def __init__(self, size: Union[Tuple[int, int], Tensor], num: int) -> None:
        """Initialize a new _NCropGenerator instance.

        Args:
            size: desired output size (out_h, out_w) of the crop
            num: number of crops to generate
        """
        super().__init__(size)
        self.num = num

    def forward(
        self, batch_shape: torch.Size, same_on_batch: bool = False
    ) -> Dict[str, Tensor]:
        """Generate the crops.

        Args:
            batch_shape: input size (b, c?, in_h, in_w)
            same_on_batch: apply the same transformation across the batch

        Returns:
            the randomly generated parameters
        """
        out = []
        for _ in range(self.num):
            out.append(super().forward(batch_shape, same_on_batch))
        return {
            "src": torch.stack([x["src"] for x in out]),
            "dst": torch.stack([x["dst"] for x in out]),
            "input_size": out[0]["input_size"],
            "output_size": out[0]["output_size"],
        }
