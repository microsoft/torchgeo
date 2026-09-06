# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# Copyright (c) 2020 VSainteuf (Vivien Sainte Fare Garnot)

"""Lightweight Temporal Attention Encoder (L-TAE) model."""

import math
from collections.abc import Sequence
from typing import cast

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn


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
        n_neurons: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        d_model: int | None = 256,
        T: int = 1000,
        len_max_seq: int = 24,
        positions: Sequence[int] | None = None,
    ) -> None:
        """Sequence-to-embedding encoder.

        Args:
            in_channels: Number of channels of the input embeddings
            n_head: Number of attention heads
            d_k: Dimension of the key and query vectors
            n_neurons: Defines the dimensions of the successive feature spaces of the
                MLP that processes the concatenated outputs of the attention heads
            dropout: dropout
            T: Period to use for the positional encoding
            len_max_seq: Maximum sequence length, used to pre-compute the positional
                encoding table
            positions: List of temporal positions to use instead of position in the
                sequence
            d_model: If specified, the input tensors will first processed by a fully
                connected layer to project them into a feature space of dimension
                d_model
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
        self.pos_encoder = IndexPositionalEncoding(self.d_model, dropout, T)

        # Use PyTorch's built-in MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=n_head, dropout=dropout, batch_first=True
        )

        self.inlayernorm = nn.LayerNorm(self.in_channels)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        assert self.n_neurons[0] == self.d_model

        activation = nn.ReLU()

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)

        Returns:
            Output tensor of shape (batch_size, n_neurons[-1])
        """
        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = rearrange(x, 'b t c -> b c t')
            x = self.inconv(x)
            x = rearrange(x, 'b c t -> b t c')

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply multi-head attention
        # PyTorch's MultiheadAttention expects query, key, value
        attention_output, _ = self.attention(x, x, x)

        # Process through MLP
        # Take the mean over the sequence dimension to get a fixed-size representation
        mlp_input = reduce(attention_output, 'b t c -> b c', 'mean')
        output: Tensor = self.outlayernorm(self.dropout(self.mlp(mlp_input)))

        return output


class IndexPositionalEncoding(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        # Get positional encoding up to the sequence length
        pe = self.pe[:, : x.size(1)]  # ty: ignore[not-subscriptable]
        output: Tensor = self.dropout(x + pe)
        return output


class DatePositionalEncoding(nn.Module):
    """Date-based sinusoidal positional encoder for L-TAE 2D.

    Unlike :class:`IndexPositionalEncoding`, this encoder maps actual acquisition
    dates (integer day values) rather than sequence indices, and supports
    repeating the encoding across attention heads.
    """

    def __init__(self, d: int, T: int = 1000, repeat: int | None = None) -> None:
        """Initialize the positional encoder.

        Args:
            d: Dimension of the encoding per head (``d_model // n_head``).
            T: Period for the sinusoidal functions.
            repeat: Number of times to repeat the encoding (``n_head``).
        """
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        denom = torch.zeros(d)
        for i in range(0, d, 2):
            denom[i] = 1.0 / (T ** (i / d))
            if i + 1 < d:
                denom[i + 1] = 1.0 / (T ** (i / d))
        self.register_buffer('denom', denom)

    def forward(self, batch_positions: Tensor) -> Tensor:
        """Encode batch positions.

        Args:
            batch_positions: Integer positions of shape ``(B, T)``.

        Returns:
            Positional encoding of shape ``(B, T, d)`` or ``(B, T, d * repeat)``.
        """
        denom = cast(Tensor, self.denom)
        positions = rearrange(batch_positions, 'b t -> b t 1').float()
        pe = positions * rearrange(denom, 'd -> 1 1 d')
        pe[..., 0::2].sin_()
        pe[..., 1::2].cos_()
        if self.repeat is not None:
            pe = repeat(pe, 'b t d -> b t (repeat d)', repeat=self.repeat)
        return pe


class LearnedQueryAttention(nn.Module):
    """Multi-head attention with a shared learned master query for L-TAE.

    The query is a learned parameter (not derived from the input), shared
    across all spatial positions. Keys and values are projected from the input.
    """

    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        """Initialize multi-head attention.

        Args:
            n_head: Number of attention heads.
            d_k: Dimension of key/query space per head.
            d_in: Total input/value dimension (``d_model``).
        """
        super().__init__()
        self.n_head = n_head
        self.d_in = d_in
        embed_dim = n_head * d_k
        self.learned_query = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.normal_(self.learned_query, std=math.sqrt(2.0 / d_k))
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            dropout=0.1,
            kdim=d_in,
            vdim=d_in,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, d_in)

    def forward(
        self, v: Tensor, pad_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            v: Value/input tensor of shape ``(B, T, d_in)``.
            pad_mask: Boolean padding mask of shape ``(B, T)``.

        Returns:
            Tuple of (output ``(n_head, B, d_in // n_head)``,
            attention ``(n_head, B, T)``).
        """
        query = repeat(self.learned_query, 'q d -> b q d', b=len(v))
        output, attn = self.mha(
            query,
            v,
            v,
            key_padding_mask=pad_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        output = self.out_proj(output)
        output = rearrange(output, 'b 1 (h d) -> h b d', h=self.n_head)
        attn = rearrange(attn, 'b h 1 t -> h b t')
        return output, attn


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder for 2D image time series (L-TAE 2D).

    Applies a shared L-TAE over all spatial positions of a ``(B, T, C, H, W)``
    image time series, producing a single ``(B, C', H, W)`` feature map.
    Attention weights are returned per head for use in skip-connection
    aggregation (e.g. in U-TAE).

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2007.00586

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        d_model: int | None = 256,
        T: int = 1000,
        positional_encoding: bool = True,
    ) -> None:
        """Initialize L-TAE 2D.

        Args:
            in_channels: Number of channels of the input embeddings.
            n_head: Number of attention heads.
            d_k: Dimension of the key and query vectors per head.
            mlp: Channel widths of the MLP that processes concatenated head
                outputs. ``mlp[0]`` must equal ``d_model``.
            dropout: Dropout rate.
            d_model: If given, projects input to this dimension first.
            T: Period for sinusoidal positional encoding.
            positional_encoding: If False, no positional encoding is applied.
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.d_model = d_model if d_model is not None else in_channels

        assert n_head > 0, 'n_head must be positive'
        assert in_channels % n_head == 0, 'in_channels must be divisible by n_head'
        assert self.d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.inconv: nn.Conv1d | None = None
        if d_model is not None:
            self.inconv = nn.Conv1d(in_channels, d_model, 1)

        n_neurons = list(mlp)
        assert n_neurons[0] == self.d_model
        if n_neurons[-1] % n_head != 0:
            raise ValueError('mlp[-1] must be divisible by n_head')

        self.positional_encoder: DatePositionalEncoding | None = None
        if positional_encoding:
            self.positional_encoder = DatePositionalEncoding(
                self.d_model // n_head, T=T, repeat=n_head
            )

        self.attention_heads = LearnedQueryAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_neurons[-1])

        layers: list[nn.Module] = []
        for i in range(len(n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(n_neurons[i], n_neurons[i + 1]),
                    nn.BatchNorm1d(n_neurons[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        batch_positions: Tensor | None = None,
        pad_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, C, H, W)``.
            batch_positions: Acquisition dates of shape ``(B, T)``. If provided
                and positional encoding is enabled, these positions are encoded
                and added to the temporal features.
            pad_mask: Boolean padding mask of shape ``(B, T)`` where ``True``
                marks padded (invalid) timesteps.

        Returns:
            Output feature map of shape ``(B, mlp[-1], H, W)`` and attention
            weights of shape ``(n_head, B, T, H, W)``.
        """
        b, _, _, h, w = x.shape

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b t -> (b h w) t', h=h, w=w)

        out = rearrange(x, 'b t c h w -> (b h w) c t')
        out = self.in_norm(out)

        if self.inconv is not None:
            out = self.inconv(out)

        out = rearrange(out, 'spatial c t -> spatial t c')

        if self.positional_encoder is not None and batch_positions is not None:
            positions = repeat(batch_positions, 'b t -> (b h w) t', h=h, w=w)
            out = out + self.positional_encoder(positions)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = rearrange(out, 'heads spatial channels -> spatial (heads channels)')
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)
        out = rearrange(out, '(b h w) c -> b c h w', b=b, h=h, w=w)

        attn = rearrange(attn, 'heads (b h w) t -> heads b t h w', b=b, h=h, w=w)
        return out, attn
