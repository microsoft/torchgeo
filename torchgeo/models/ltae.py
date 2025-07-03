# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Copyright (c) 2020 VSainteuf (Vivien Sainte Fare Garnot)

"""Lightweight Temporal Attention Encoder (LTAE) model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder (L-TAE).

    This model implements a lightweight temporal attention encoder that processes
    time series data using a multi-head attention mechanism. It is designed to
    efficiently encode temporal sequences into fixed-length embeddings.

    The model architecture is based on the paper:
    "Lightweight Temporal Self-Attention for Classifying Satellite Images Time Series: https://arxiv.org/pdf/2007.00586"

    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 8,
        n_neurons: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        d_model: int = 256,
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

        if positions is None:
            positions = [len_max_seq + 1]

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(
                nn.Conv1d(in_channels, d_model, 1),
                nn.LayerNorm([d_model, len_max_seq])
            )
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(positions[0], self.d_model // n_head, T=T)
        self.position_enc = nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]
            torch.cat([sin_tab for _ in range(n_head)], dim=1),
            freeze=True
        )

        self.inlayernorm = nn.LayerNorm(self.in_channels)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        assert self.n_neurons[0] == self.d_model

        activation = nn.ReLU(inplace=True)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([
                nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                nn.BatchNorm1d(self.n_neurons[i + 1]),
                activation
            ])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_neurons[-1])
        """
        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positions is None:
            src_pos = (
                torch.arange(1, seq_len + 1, dtype=torch.long)
                .expand(sz_b, seq_len)
                .to(x.device)
            )
        else:
            src_pos = (
                torch.arange(0, seq_len, dtype=torch.long)
                .expand(sz_b, seq_len)
                .to(x.device)
            )

        enc_output = x + self.position_enc(src_pos)
        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        enc_output = (
            enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)
        )  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        return enc_output  # type: ignore[no-any-return]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    This module implements multi-head attention mechanism that allows the model
    to jointly attend to information from different representation subspaces.
    """

    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        """Initialize the Multi-Head Attention module.

        Args:
            n_head: Number of attention heads
            d_k: Dimension of key and query vectors
            d_in: Input dimension
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # Initialize query parameter
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        # Initialize key transformation
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        # Initialize attention mechanism
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the multi-head attention module.

        Args:
            q: Query tensor of shape (batch_size, seq_len, d_in)
            k: Key tensor of shape (batch_size, seq_len, d_in)
            v: Value tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Tuple containing:
                - output: Transformed output tensor of shape (n_head, batch_size, d_in//n_head)
                - attn: Attention weights of shape (n_head, batch_size, seq_len)
        """
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        # Prepare query
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        # Transform key
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        # Prepare value
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1))  # type: ignore[no-untyped-call]
        v = v.view(n_head * sz_b, seq_len, -1)

        # Apply attention
        output, attn = self.attention(q, k, v)

        # Reshape attention weights
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        # Reshape output
        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism.

    Implements the attention mechanism described in the paper
    "Attention is All You Need" with scaling factor and dropout.
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        """Initialize the Scaled Dot-Product Attention module.

        Args:
            temperature: Scaling factor for the dot product attention
            attn_dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout, inplace=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple containing:
                - output: Attention output
                - attn: Attention weights
        """
        # Compute attention scores
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        # Apply softmax and dropout
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Compute output
        output = torch.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(
    positions: int | list[int],
    d_hid: int,
    T: int = 1000,
    device: str = "cpu"
) -> torch.Tensor:
    """Generate sinusoidal position encoding table.

    Args:
        positions: Number of positions or list of positions
        d_hid: Hidden dimension
        T: Period for the sinusoidal functions
        device: Device to put the tensor on

    Returns:
        torch.Tensor: Position encoding table
    """
    def cal_angle(position: int | float, hid_idx: int) -> float:
        """Calculate angle for positional encoding."""
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)  # type: ignore[no-any-return]

    def get_posi_angle_vec(position: int | float) -> list[float]:
        """Get position angle vector."""
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    if isinstance(positions, list):
        sinusoid_table = [get_posi_angle_vec(pos_i) for pos_i in positions]
    else:
        sinusoid_table = [get_posi_angle_vec(pos_i) for pos_i in range(positions)]

    sinusoid_table = np.array(sinusoid_table)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).to(device)


def get_sinusoid_encoding_table_var(
    positions: int | list[int],
    d_hid: int,
    clip: int = 4,
    offset: int = 3,
    T: int = 1000,
    device: str = "cpu"
) -> torch.Tensor:
    """Generate variable sinusoidal position encoding table.

    Args:
        positions: Number of positions or list of positions
        d_hid: Hidden dimension
        clip: Clipping value
        offset: Offset value
        T: Period for the sinusoidal functions
        device: Device to put the tensor on

    Returns:
        torch.Tensor: Position encoding table
    """
    def cal_angle(position: int | float, hid_idx: int) -> float:
        """Calculate angle for positional encoding."""
        return position / np.power(T, 2 * (hid_idx + offset // 2) / d_hid)  # type: ignore[no-any-return]

    def get_posi_angle_vec(position: int | float) -> list[float]:
        """Get position angle vector."""
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    if isinstance(positions, list):
        sinusoid_table = [get_posi_angle_vec(pos_i) for pos_i in positions]
    else:
        sinusoid_table = [get_posi_angle_vec(pos_i) for pos_i in range(positions)]

    sinusoid_table = np.array(sinusoid_table)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    sinusoid_table[:, clip:] = torch.zeros(sinusoid_table[:, clip:].shape)

    return torch.FloatTensor(sinusoid_table).to(device)

if __name__ == "__main__":
    model = LTAE(in_channels=128)
    x = torch.randn(4, 24, 128)
    output = model(x)
    print(output.shape)