from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Randomly flip the image and target tensors.

        Parameters:
            image: image to be flipped
            target: optional bounding boxes and masks to flip

        Returns:
            randomly flipped image and target
        """
        if torch.rand(1) < self.p:
            image = F.hflip(image)

            if target is not None:
                width, height = F._get_image_size(image)

                if "boxes" in target:
                    target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)

        return image, target


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Randomly flip the image and target tensors.

        Parameters:
            image: image to be flipped
            target: optional bounding boxes and masks to flip

        Returns:
            randomly flipped image and target
        """
        if torch.rand(1) < self.p:
            image = F.vflip(image)

            if target is not None:
                width, height = F._get_image_size(image)

                if "boxes" in target:
                    target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-2)

        return image, target
