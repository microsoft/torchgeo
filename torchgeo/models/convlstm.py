# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2017 Andrea Palazzi

"""Convolutional Long Short-Term Memory (ConvLSTM) model."""

from typing import Literal, cast

import torch
from timm.layers.classifier import ClassifierHead
from torch import nn


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
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

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

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/1506.04214

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int] = 64,
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        num_layers: int = 1,
        bias: bool = True,
        return_all_layers: bool = False,
        num_classes: int = 1,
        head_kernel_size: int = 1,
        convolutional_head: bool = False,
    ) -> None:
        """Initializes the ConvLSTM model.

        Args:
            input_dim: Number of channels in the input.
            hidden_dim: Number of hidden channels. Can be a single int (for all
                layers) or a list of ints (one for each layer).
            kernel_size: Size of the convolutional kernel. Can be:

                * a single integer (for square kernels)
                * a tuple of two integers (for rectangular kernels)
                * a list of integers or tuples (one for each layer)
            num_layers: Number of LSTM layers stacked on each other.
            bias: If ``True``, adds a learnable bias to the output.
            return_all_layers: If ``True``, will return the list of computations
                for all layers.
            num_classes: Optional number of segmentation classes for an attached
                prediction head.
            head_kernel_size: Kernel size for the optional segmentation head.
            convolutional_head: If ``False``, uses global average pooling followed by a
                fully connected head for image-level prediction. If ``True``, uses a
                convolutional head for dense prediction.
        """
        super().__init__()

        # Normalize hidden_dim to a list of ints
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * num_layers
        else:
            self.hidden_dim = hidden_dim

        # Normalize kernel_size to a list of tuples
        if isinstance(kernel_size, int | tuple):
            ks_list = [kernel_size] * num_layers
        else:
            ks_list = kernel_size

        self.kernel_size = [(ks, ks) if isinstance(ks, int) else ks for ks in ks_list]

        if not len(self.kernel_size) == len(self.hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.num_classes = num_classes

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
        self.head = ClassifierHead(
            in_features=self.hidden_dim[-1],
            num_classes=self.num_classes,
            use_conv=convolutional_head,
            pool_type='' if convolutional_head else 'avg',
        )

    def forward_features(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of ConvLSTM feature extraction.

        .. versionadded:: 0.10

        Args:
            input_tensor: A 5-D Tensor of shape (b, t, c, h, w).
            hidden_state: An optional initial hidden state.

        Returns:
            A tuple containing layer_output_list and last_state_list.
        """
        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_state, c_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_state, c_state = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=(h_state, c_state),
                )
                output_inner.append(h_state)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h_state, c_state))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def forward(
        self,
        input_tensor: torch.Tensor,
        lengths: torch.Tensor | None = None,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Forward pass for segmentation with the prediction head.

        Args:
            input_tensor: A 5-D Tensor of shape (b, t, c, h, w).
            lengths: Optional sequence lengths (B,) before padding/truncation.
                Values larger than the available sequence length use the final
                timestep.
            hidden_state: An optional initial hidden state.

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        layer_output_list, _ = self.forward_features(
            input_tensor, hidden_state=hidden_state
        )
        layer_output = layer_output_list[-1]

        if lengths is None:
            features = layer_output[:, -1]
        else:
            idx = lengths.to(device=layer_output.device, dtype=torch.long) - 1
            idx = idx.clamp(min=0, max=layer_output.size(1) - 1)
            batch_idx = torch.arange(layer_output.size(0), device=idx.device)
            features = layer_output[batch_idx, idx]

        return self.head(features)

    def _init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initializes the hidden states for all layers.

        Args:
            batch_size: The size of the batch dimension.
            image_size: A tuple of (height, width) for the spatial dimensions.

        Returns:
            A list of tuples, where each tuple contains the hidden state and cell state
            tensors for a layer. Each tensor has shape (batch_size, hidden_dim, height, width).
        """
        init_states = []
        for i in range(self.num_layers):
            cell = cast(ConvLSTMCell, self.cell_list[i])
            init_states.append(cell.init_hidden(batch_size, image_size))
        return init_states


class Conv3dLSTM(ConvLSTM):
    """Conv3d projection followed by ConvLSTM for image time series.

    This model first extracts spatiotemporal features with a 3D convolution, then
    feeds the projected sequence to :class:`ConvLSTM`. The prediction head can
    return per-pixel maps or chip-level predictions, making the model suitable for
    regression and classification tasks.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        input_dim: int,
        conv3d_dim: int = 64,
        hidden_dim: int | list[int] = 64,
        conv3d_kernel_size: int | tuple[int, int, int] = (3, 3, 3),
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        num_layers: int = 1,
        bias: bool = True,
        return_all_layers: bool = False,
        num_outputs: int = 1,
        head_kernel_size: int = 1,
        output_mode: Literal['pixel', 'chip'] = 'pixel',
        return_sequence: bool = False,
        pooling: Literal['avg', 'max'] = 'avg',
    ) -> None:
        """Initialize a new Conv3dLSTM model.

        Args:
            input_dim: Number of channels per timestep in the input.
            conv3d_dim: Number of output channels from the 3D convolution.
            hidden_dim: Number of ConvLSTM hidden channels. Can be a single int
                or a list of ints, one for each layer.
            conv3d_kernel_size: Size of the 3D convolutional kernel. Can be a
                single integer or a tuple of ``(time, height, width)``.
            kernel_size: Size of the ConvLSTM convolutional kernel.
            num_layers: Number of LSTM layers stacked on each other.
            bias: If ``True``, adds a learnable bias to convolutions.
            return_all_layers: If ``True``, will return computations for all
                ConvLSTM layers from :meth:`forward_features`.
            num_outputs: Number of output channels, classes, labels, or values.
            head_kernel_size: Kernel size for the prediction head.
            output_mode: Whether to return per-pixel maps or chip-level outputs.
            return_sequence: If ``True``, return predictions for every timestep.
            pooling: Pooling method used when ``output_mode='chip'``.
        Raises:
            ValueError: If an unsupported option or even kernel size is provided.
        """
        if isinstance(conv3d_kernel_size, int):
            conv3d_kernel_size_tuple = (
                conv3d_kernel_size,
                conv3d_kernel_size,
                conv3d_kernel_size,
            )
        else:
            conv3d_kernel_size_tuple = conv3d_kernel_size

        if any(k % 2 == 0 for k in conv3d_kernel_size_tuple):
            raise ValueError('conv3d_kernel_size must be odd to preserve shape.')

        if head_kernel_size % 2 == 0:
            raise ValueError('head_kernel_size must be odd to preserve shape.')

        if output_mode not in {'pixel', 'chip'}:
            raise ValueError("output_mode must be 'pixel' or 'chip'.")

        if pooling not in {'avg', 'max'}:
            raise ValueError("pooling must be 'avg' or 'max'.")

        super().__init__(
            input_dim=conv3d_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            return_all_layers=return_all_layers,
            num_classes=num_outputs,
            head_kernel_size=head_kernel_size,
            convolutional_head=output_mode == 'pixel',
        )

        self.input_dim = input_dim
        self.conv3d_dim = conv3d_dim
        self.conv3d_kernel_size = conv3d_kernel_size_tuple
        self.num_outputs = num_outputs
        self.output_mode = output_mode
        self.return_sequence = return_sequence
        self.pooling = pooling

        conv3d_padding = (
            conv3d_kernel_size_tuple[0] // 2,
            conv3d_kernel_size_tuple[1] // 2,
            conv3d_kernel_size_tuple[2] // 2,
        )
        self.input_projection = nn.Conv3d(
            in_channels=input_dim,
            out_channels=conv3d_dim,
            kernel_size=conv3d_kernel_size_tuple,
            padding=conv3d_padding,
            bias=bias,
        )

        match pooling:
            case 'avg':
                self.pool = nn.AdaptiveAvgPool2d(1)
            case 'max':
                self.pool = nn.AdaptiveMaxPool2d(1)

    def forward_features(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of Conv3dLSTM feature extraction.

        Args:
            input_tensor: A 5-D Tensor of shape (b, t, c, h, w).
            hidden_state: An optional initial hidden state.

        Returns:
            A tuple containing layer_output_list and last_state_list.

        Raises:
            ValueError: If ``input_tensor`` is not 5-D.
        """
        if input_tensor.ndim != 5:
            raise ValueError('Expected input_tensor with shape (B, T, C, H, W).')

        features = self.input_projection(input_tensor.permute(0, 2, 1, 3, 4))
        features = features.permute(0, 2, 1, 3, 4)
        return super().forward_features(features, hidden_state=hidden_state)

    def _select_features(
        self, layer_output: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select final or length-indexed features.

        Args:
            layer_output: Hidden features of shape (B, T, C, H, W).
            lengths: Optional sequence lengths (B,) before padding/truncation.

        Returns:
            Hidden features of shape (B, C, H, W).
        """
        if lengths is None:
            return layer_output[:, -1]

        idx = lengths.to(device=layer_output.device, dtype=torch.long) - 1
        idx = idx.clamp(min=0, max=layer_output.size(1) - 1)
        batch_idx = torch.arange(layer_output.size(0), device=idx.device)
        return layer_output[batch_idx, idx]

    def _format_output(self, output: torch.Tensor) -> torch.Tensor:
        """Format pixel-level or chip-level output.

        Args:
            output: Prediction map of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            Prediction tensor formatted according to ``output_mode``.
        """
        if self.output_mode == 'pixel':
            return output

        if output.ndim == 5:
            b, t, c, h, w = output.shape
            output = output.reshape(b * t, c, h, w)
            output = self.pool(output).flatten(1)
            return output.reshape(b, t, c)

        if output.ndim == 4:
            return self.pool(output).flatten(1)

        return output

    def forward(
        self,
        input_tensor: torch.Tensor,
        lengths: torch.Tensor | None = None,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: A 5-D Tensor of shape (B, T, C, H, W).
            lengths: Optional sequence lengths (B,) before padding/truncation.
            hidden_state: An optional initial hidden state.

        Returns:
            If ``output_mode='pixel'`` and ``return_sequence=False``, an output
            tensor of shape (B, C, H, W). If ``return_sequence=True``, the shape is
            (B, T, C, H, W). If ``output_mode='chip'``, spatial dimensions are
            pooled away and the corresponding shapes are (B, C) or (B, T, C).
        """
        layer_output_list, _ = self.forward_features(
            input_tensor, hidden_state=hidden_state
        )
        layer_output = layer_output_list[-1]

        if self.return_sequence:
            b, t, c, h, w = layer_output.shape
            features = layer_output.reshape(b * t, c, h, w)
            output = self.head(features)
            if output.ndim == 4:
                output = output.reshape(b, t, self.num_outputs, h, w)
            else:
                output = output.reshape(b, t, self.num_outputs)
        else:
            features = self._select_features(layer_output, lengths=lengths)
            output = self.head(features)

        return self._format_output(output)
