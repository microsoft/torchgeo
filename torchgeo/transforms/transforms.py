# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Any, Dict, List, Tuple, Union

import kornia.augmentation as K
import torch
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


class ConstantNormalize(Module):
    """Normalize image tensor by a constant."""

    def __init__(self, val: float) -> None:
        """Initialize a new instance of ConstantNormalize.

        Args:
            val: value by which to normalize
        """
        super().__init__()
        self.val = val

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Normalize sample image by a constant.

        Args:
            sample: the input

        Returns:
            output sample with normalized image
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= self.val
        return sample


class RepeatedRandomCrop(Module):
    """Take repeated random crops from a tile.

    This is usefule to create *num_patches_per_tile* random
    patches from a tile, to have a manageable input dimension
    size for various models.

    .. versionadded: 0.4
    """

    def __init__(
        self,
        patch_size: Union[Tuple[int, int], int],
        num_patches_per_tile: int,
        data_keys: List[str],
    ) -> None:
        """Initialize a new instance of RepeatedRandomCrop.

        Args:
            patch_size: Size of random patch from image and mask (height, width)
            num_patches_per_tile: number of patches to
                randomly crop from a tile
            data_keys: data keys contained in *sample*
        """
        super().__init__()
        self.num_patches_per_tile = num_patches_per_tile
        self.rcrop = K.AugmentationSequential(
            K.RandomCrop(patch_size), data_keys=data_keys
        )

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Perform random cropping on a tile to create patches.

        Args:
            sample: the input

        Returns:
            stacked random patches
        """
        images, masks = [], []
        for i in range(self.num_patches_per_tile):
            image, mask = self.rcrop(sample["image"], sample["mask"].float())
            images.append(image.squeeze(0))
            masks.append(mask.squeeze().long())

        sample["image"] = torch.stack(images)
        sample["mask"] = torch.stack(masks)
        return sample


class PadSegmentationSamples(Module):
    """Pad Segmentation samples to a next multiple.

    This is useful for several segmentation models that
    except the input dimensions to be a multiple of 32.

    .. versionadded: 0.4
    """

    def __init__(self, multiple: int = 32):
        """Initialize a new instance of PadSegmentationSamples.

        Args:
            multiple: what next multiple to pad to

        Raises:
            AssertionError if *multiple* is not positve.
        """
        super().__init__()
        assert multiple > 0, "Multiple argument must be positive"
        self.multiple = multiple

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Pad samples to next multiple.

        Args:
            sample: contains image and mask tile from dataset

        Returns:
            stacked randomly cropped patches from input tile
        """
        if "image" in sample:
            dim_key = "image"
        else:
            dim_key = "mask"

        h, w = sample[dim_key].shape[1], sample[dim_key].shape[2]
        new_h = int(self.multiple * ((h // self.multiple) + 1))
        new_w = int(self.multiple * ((w // self.multiple) + 1))

        padto = K.PadTo((new_h, new_w))

        if "image" in sample:
            sample["image"] = padto(sample["image"])[0]
        # mask has long type but Kornia expects float
        if "mask" in sample:
            sample["mask"] = padto(sample["mask"].float()).long()[0, 0]
        return sample
