"""Simple fully convolutional neural network (FCN) implementations."""

from typing import OrderedDict

import torch.nn as nn
from torch import Tensor


class FCN(nn.modules.Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.layers = OrderedDict(
            [
                ("conv1", conv1),
                ("relu1", nn.modules.LeakyReLU(inplace=True)),
                ("conv2", conv2),
                ("relu2", nn.modules.LeakyReLU(inplace=True)),
                ("conv3", conv3),
                ("relu3", nn.modules.LeakyReLU(inplace=True)),
                ("conv4", conv4),
                ("relu4", nn.modules.LeakyReLU(inplace=True)),
                ("conv5", conv5),
                ("relu5", nn.modules.LeakyReLU(inplace=True)),
            ]
        )

        self.backbone = nn.modules.Sequential(self.layers)

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x
