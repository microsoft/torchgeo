"""Simple fully convolutional neural network (FCN) implementations."""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


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

        self.conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = F.leaky_relu(self.conv1(inputs), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = self.last(x)
        return x
