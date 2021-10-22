# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Implementation of a random convolutional feature projection model."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Conv2d, Module

Module.__module__ = "torch.nn"
Conv2d.__module__ = "torch.nn"


class RCF(Module):
    """This model extracts random convolutional features (RCFs) from its input.

    RCFs are used in Multi-task Observation using Satellite Imagery & Kitchen Sinks
    (MOSAIKS) method proposed in https://www.nature.com/articles/s41467-021-24638-z.

    .. note::

        This Module is *not* trainable. It is only used as a feature extractor.
    """

    def __init__(
        self,
        in_channels: int = 4,
        features: int = 16,
        kernel_size: int = 3,
        bias: float = -1.0,
    ) -> None:
        """Initializes the RCF model.

        This is a static model that serves to extract fixed length feature vectors from
        input patches.

        Args:
            in_channels: number of input channels
            features: number of features to compute, must be divisible by 2
            kernel_size: size of the kernel used to compute the RCFs
            bias: bias of the convolutional layer
        """
        super().__init__()

        assert features % 2 == 0

        # We register the weight and bias tensors as "buffers". This does two things:
        # makes them behave correctly when we call .to(...) on the module, and makes
        # them explicitely _not_ Parameters of the model (which might get updated) if
        # a user tries to train with this model.
        self.register_buffer(
            "weights",
            torch.randn(
                features // 2,
                in_channels,
                kernel_size,
                kernel_size,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "biases",
            torch.zeros(  # type: ignore[attr-defined]
                features // 2, requires_grad=False
            )
            + bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the RCF model.

        Args:
            x: a tensor with shape (B, C, H, W)

        Returns:
            a tensor of size (B, ``self.num_features``)
        """
        x1a = F.relu(
            F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0),
            inplace=True,
        )
        x1b = F.relu(
            -F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0),
            inplace=False,
        )

        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:  # case where we passed a single input
            output = torch.cat((x1a, x1b), dim=0)  # type: ignore[attr-defined]
            return cast(Tensor, output)
        else:  # case where we passed a batch of > 1 inputs
            assert len(x1a.shape) == 2
            output = torch.cat((x1a, x1b), dim=1)  # type: ignore[attr-defined]
            return cast(Tensor, output)
