# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Bidirectional Convolutional Long Short-Term Memory model."""

import torch
from torch import Tensor, nn

from .convlstm import ConvLSTM


class BidirectionalConvLSTM(nn.Module):
    """Bidirectional ConvLSTM model for image time series.

    Processes an image sequence in chronological and reverse-chronological order,
    then concatenates the terminal features from both directions. This provides
    each prediction with context from the beginning and end of the sequence.

    Inspired by the bidirectional ConvLSTM baseline from the `U-TAE/PaPs
    repository <https://github.com/VSainteuf/utae-paps>`_.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/1506.04214
    * https://arxiv.org/abs/2107.07933

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int] = 64,
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        num_layers: int = 1,
        bias: bool = True,
        num_classes: int = 1,
        head_kernel_size: int = 1,
    ) -> None:
        """Initialize a BidirectionalConvLSTM model.

        Args:
            input_dim: Number of channels in the input.
            hidden_dim: Number of hidden channels. Can be a single integer or a
                list with one value per layer.
            kernel_size: Convolutional kernel size. Can be a single integer, a
                tuple of two integers, or a list with one value per layer.
            num_layers: Number of recurrent layers in each direction.
            bias: Whether convolutions include a learnable bias.
            num_classes: Number of output segmentation classes.
            head_kernel_size: Kernel size of the segmentation head.
        """
        super().__init__()
        self.forward_encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
        )
        self.backward_encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
        )
        self.forward_encoder.head = nn.Identity()
        self.backward_encoder.head = nn.Identity()

        final_hidden_dim = self.forward_encoder.hidden_dim[-1]
        self.head = nn.Conv2d(
            in_channels=2 * final_hidden_dim,
            out_channels=num_classes,
            kernel_size=head_kernel_size,
            padding=head_kernel_size // 2,
        )

    def _normalize_lengths(
        self, input_tensor: Tensor, lengths: Tensor | None
    ) -> Tensor:
        """Normalize sequence lengths for indexing.

        Args:
            input_tensor: Input of shape ``(B, T, C, H, W)``.
            lengths: Sequence lengths of shape ``(B,)``.

        Returns:
            Integer sequence lengths clamped to the available time dimension.

        Raises:
            ValueError: If *input_tensor* is not 5-D or *lengths* has the wrong
                shape.
        """
        if input_tensor.ndim != 5:
            raise ValueError('Expected input_tensor with shape (B, T, C, H, W).')

        batch_size, sequence_length = input_tensor.shape[:2]
        if lengths is None:
            return torch.full(
                (batch_size,),
                sequence_length,
                dtype=torch.long,
                device=input_tensor.device,
            )
        if lengths.shape != (batch_size,):
            raise ValueError('lengths must have shape (B,).')
        return lengths.to(device=input_tensor.device, dtype=torch.long).clamp(
            min=1, max=sequence_length
        )

    def _reverse_valid_prefix(self, input_tensor: Tensor, lengths: Tensor) -> Tensor:
        """Reverse valid timesteps while leaving trailing padding in place.

        Args:
            input_tensor: Input of shape ``(B, T, ...)``.
            lengths: Valid sequence lengths of shape ``(B,)``.

        Returns:
            Input with the valid prefix of each sample reversed.
        """
        batch_size, sequence_length = input_tensor.shape[:2]
        indices = torch.arange(sequence_length, device=input_tensor.device).expand(
            batch_size, sequence_length
        )
        reverse_indices = torch.where(
            indices < lengths[:, None], lengths[:, None] - indices - 1, indices
        )
        gather_indices = reverse_indices.reshape(
            batch_size, sequence_length, *([1] * (input_tensor.ndim - 2))
        ).expand_as(input_tensor)
        return input_tensor.gather(dim=1, index=gather_indices)

    def forward_features(
        self, input_tensor: Tensor, lengths: Tensor | None = None
    ) -> Tensor:
        """Extract terminal features from both temporal directions.

        Args:
            input_tensor: Input tensor of shape ``(B, T, C, H, W)``.
            lengths: Valid sequence lengths of shape ``(B,)``.

        Returns:
            Concatenated features of shape ``(B, 2 * hidden_dim[-1], H, W)``.
        """
        lengths = self._normalize_lengths(input_tensor, lengths)
        reverse_input = self._reverse_valid_prefix(input_tensor, lengths)
        forward_outputs, _ = self.forward_encoder.forward_features(input_tensor)
        backward_outputs, _ = self.backward_encoder.forward_features(reverse_input)

        indices = lengths - 1
        batch_indices = torch.arange(input_tensor.shape[0], device=input_tensor.device)
        forward_features = forward_outputs[-1][batch_indices, indices]
        backward_features = backward_outputs[-1][batch_indices, indices]
        return torch.cat([forward_features, backward_features], dim=1)

    def forward(self, input_tensor: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Generate a segmentation prediction.

        Args:
            input_tensor: Input tensor of shape ``(B, T, C, H, W)``.
            lengths: Valid sequence lengths of shape ``(B,)``.

        Returns:
            Prediction tensor of shape ``(B, num_classes, H, W)``.
        """
        return self.head(self.forward_features(input_tensor, lengths=lengths))
