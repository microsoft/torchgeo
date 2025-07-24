# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Convolutional Long Short-Term Memory (ConvLSTM) model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import torch
import torch.nn as nn

T = TypeVar("T")


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        bias: bool = True,
    ) -> None:
        """Initializes a ConvLSTMCell.

        Args:
            input_dim: Number of channels of input tensor.
            hidden_dim: Number of channels of hidden state.
            kernel_size: Size of the convolutional kernel.
            bias: Whether or not to add the bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self, input_tensor: torch.Tensor, cur_state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the ConvLSTMCell.

        Args:
            input_tensor: Tensor of shape (b, c, h, w).
            cur_state: Tuple containing the current hidden and cell states.

        Returns:
            A tuple containing the next hidden and cell states.
        """
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state.

        Args:
            batch_size: The batch size.
            image_size: The height and width of the image.

        Returns:
            A tuple of tensors for the initial hidden and cell states.
        """
        height, width = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class ConvLSTM(nn.Module):
    """Convolutional LSTM model.

    This model is a sequence-processing model that uses convolutional operations
    within the LSTM cells. It is particularly useful for spatio-temporal data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        kernel_size: tuple[int, int] | Sequence[tuple[int, int]],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        """Initializes the ConvLSTM model.

        Args:
            input_dim: Number of channels in the input.
            hidden_dim: Number of hidden channels. Can be a single int (for all
                layers) or a sequence of ints (one for each layer).
            kernel_size: Size of the convolutional kernel. Can be a single tuple
                (for all layers) or a sequence of tuples (one for each layer).
            num_layers: Number of LSTM layers stacked on each other.
            batch_first: If ``True``, then the input and output tensors are
                provided as (b, t, c, h, w).
            bias: If ``True``, adds a learnable bias to the output.
            return_all_layers: If ``True``, will return the list of computations
                for all layers.
        """
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(kernel_size, list):
            kernel_size = [
                ks if isinstance(ks, tuple) else (ks, ks) for ks in kernel_size
            ]
        elif isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        hidden_dim_list = self._extend_for_multilayer(hidden_dim, num_layers)
        kernel_size_list = self._extend_for_multilayer(kernel_size, num_layers)

        if not len(kernel_size_list) == len(hidden_dim_list) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_list
        self.kernel_size = kernel_size_list
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the ConvLSTM.

        Args:
            input_tensor: A 5-D Tensor of shape (t, b, c, h, w) or (b, t, c, h, w).
            hidden_state: An optional initial hidden state.

        Returns:
            A tuple of two lists:
            1. layer_output_list: List of Tensors of shape (b, t, c, h, w).
            2. last_state_list: List of tuples of (h, c) for the last time step.
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initializes the hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(
        kernel_size: tuple[int, int] | Sequence[tuple[int, int]]
    ) -> None:
        """Checks if the kernel_size is correctly formatted."""
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all(isinstance(elem, tuple) for elem in kernel_size)
            )
        ):
            raise ValueError("`kernel_size` must be a tuple or a list of tuples.")

    @staticmethod
    def _extend_for_multilayer(
        param: T, num_layers: int
    ) -> list[T]:
        """Extends a parameter to a list of length num_layers."""
        if not isinstance(param, list):
            return [param] * num_layers
        return param
