# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Any, Dict, List, Optional, Tuple, Union

import kornia
import torch
from einops import rearrange
from kornia.augmentation import GeometricAugmentationBase2D
from kornia.augmentation.random_generator import CropGenerator
from kornia.geometry import crop_by_indices
from torch import Tensor
from torch.nn.modules import Module


# TODO: contribute these to Kornia and delete this file
class AugmentationSequential(Module):
    """Wrapper around kornia AugmentationSequential to handle input dicts.

    .. deprecated:: 0.4
       Use :class:`kornia.augmentation.container.AugmentationSequential` instead.
    """

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

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Perform augmentations and update data dict.

        Args:
            batch: the input

        Returns:
            the augmented input
        """
        # Kornia augmentations require all inputs to be float
        dtype = {}
        for key in self.data_keys:
            dtype[key] = batch[key].dtype
            batch[key] = batch[key].float()

        # Kornia requires masks to have a channel dimension
        if "mask" in batch and len(batch["mask"].shape) == 3:
            batch["mask"] = rearrange(batch["mask"], "b h w -> b () h w")

        inputs = [batch[k] for k in self.data_keys]
        outputs_list: Union[Tensor, List[Tensor]] = self.augs(*inputs)
        outputs_list = (
            outputs_list if isinstance(outputs_list, list) else [outputs_list]
        )
        outputs: Dict[str, Tensor] = {
            k: v for k, v in zip(self.data_keys, outputs_list)
        }
        batch.update(outputs)

        # Convert all inputs back to their previous dtype
        for key in self.data_keys:
            batch[key] = batch[key].to(dtype[key])

        # Torchmetrics does not support masks with a channel dimension
        if "mask" in batch and batch["mask"].shape[1] == 1:
            batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")

        return batch


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
