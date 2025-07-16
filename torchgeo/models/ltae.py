# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Copyright (c) 2020 VSainteuf (Vivien Sainte Fare Garnot)

"""Lightweight Temporal Attention Encoder (L-TAE) model."""

from typing import Sequence

import math
import torch
import torch.nn as nn


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder (L-TAE).

    This model implements a lightweight temporal attention encoder that processes
    time series data using a multi-head attention mechanism. It is designed to
    efficiently encode temporal sequences into fixed-length embeddings.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2007.00586

    .. versionadded:: 0.8

    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 8,
        n_neurons: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        d_model: int | None = 256,
        T: int = 1000,
        len_max_seq: int = 24,
        positions: list[int] | None = None,
    ) -> None:
        """Sequence-to-embedding encoder.

        Args:
            in_channels: Number of channels of the input embeddings
            n_head: Number of attention heads
            d_k: Dimension of the key and query vectors
            n_neurons: Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout: dropout
            T: Period to use for the positional encoding
            len_max_seq: Maximum sequence length, used to pre-compute the positional encoding table
            positions: List of temporal positions to use instead of position in the sequence
            d_model: If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model

        """
        super().__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = n_neurons
        self.d_model = d_model if d_model is not None else in_channels
        self.inconv: nn.Sequential | None = None

        if d_model is not None:
            self.inconv = nn.Sequential(
                nn.Conv1d(in_channels, d_model, 1), nn.LayerNorm([d_model, len_max_seq])
            )

        # Use PyTorch's built-in positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, T)

        # Use PyTorch's built-in MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.inlayernorm = nn.LayerNorm(self.in_channels)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        assert self.n_neurons[0] == self.d_model

        activation = nn.ReLU(inplace=True)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                    nn.BatchNorm1d(self.n_neurons[i + 1]),
                    activation,
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_neurons[-1])

        Raises:
            AssertionError: If input tensor dimensions don't match expected shape
        """
        sz_b, seq_len, d = x.shape
        assert d == self.in_channels, (
            f'Input channels {d} does not match expected channels {self.in_channels}'
        )

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply multi-head attention
        # PyTorch's MultiheadAttention expects query, key, value
        enc_output, _ = self.attention(x, x, x)

        # Process through MLP
        # Take the mean over the sequence dimension to get a fixed-size representation
        enc_output = enc_output.mean(dim=1)  # (batch_size, d_model)
        enc_output: torch.Tensor = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        return enc_output


class PositionalEncoding(nn.Module):
    """Positional encoding module using sinusoidal functions."""

    def __init__(self, d_model: int, dropout: float = 0.1, T: int = 1000) -> None:
        """Initialize the positional encoding.

        Args:
            d_model: The dimension of the embeddings
            dropout: Dropout rate
            T: Period for the sinusoidal functions
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(T) / d_model))
        pe = torch.zeros(1, T, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
