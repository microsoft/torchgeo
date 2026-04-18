# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2025 Frank Feng
# Based on the original code: https://github.com/ucam-eo/tessera

"""Tessera pixel time-series foundation model."""

from typing import Any

import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from torch import Tensor, nn
from torchvision.models._api import Weights, WeightsEnum

# Normalization statistics from https://github.com/ucam-eo/tessera/blob/b994972f637d1985185725153b55cf1624a7a445/tessera_infer/src/datasets/ssl_dataset.py#L20-L26
_S2_BAND_MEAN = [
    1711.0938,
    1308.8511,
    1546.4543,
    3010.1293,
    3106.5083,
    2068.3044,
    2685.0845,
    2931.5889,
    2514.6928,
    1899.4922,
]
_S2_BAND_STD = [
    1926.1026,
    1862.9751,
    1803.1792,
    1741.7837,
    1677.4543,
    1888.7862,
    1736.3090,
    1715.8104,
    1514.5199,
    1398.4779,
]
_S1_BAND_MEAN = [5484.0407, 3003.7812]
_S1_BAND_STD = [1871.2334, 1726.0670]
_TESSERA_S2_MEAN = [*_S2_BAND_MEAN, 0.0]
_TESSERA_S2_STD = [*_S2_BAND_STD, 1.0]
_TESSERA_S1_MEAN = [*_S1_BAND_MEAN, 0.0]
_TESSERA_S1_STD = [*_S1_BAND_STD, 1.0]
_TESSERA_MEAN = _TESSERA_S2_MEAN + _TESSERA_S1_MEAN
_TESSERA_STD = _TESSERA_S2_STD + _TESSERA_S1_STD


class _PixelTimeSeriesNormalize(nn.Module):
    """Normalize pixel time series data."""

    def __init__(
        self, mean: list[float], std: list[float], inplace: bool = False
    ) -> None:
        super().__init__()
        self.normalize = T.Normalize(mean=mean, std=std, inplace=inplace)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PixelTimeSeriesNormalize.

        Args:
            tensor: Input tensor of shape (time, channels) or (batch, time, channels).

        Returns:
            Normalized tensor with the same shape as input.
        """
        assert tensor.ndim in [2, 3], (
            'Input must be a 2D (time, channels) or 3D (batch, time, channels) tensor'
        )
        x: torch.Tensor
        if tensor.ndim == 2:
            x = rearrange(tensor, 't c -> () c () t')
            x = self.normalize(x)
            x = rearrange(x, '() c () t -> t c')
        else:
            x = rearrange(tensor, 'b t c -> b c () t')
            x = self.normalize(x)
            x = rearrange(x, 'b c () t -> b t c')
        return x


_tessera_transforms = torch.nn.Sequential(
    _PixelTimeSeriesNormalize(mean=_TESSERA_MEAN, std=_TESSERA_STD, inplace=True)
)
_tessera_s2_transforms = torch.nn.Sequential(
    _PixelTimeSeriesNormalize(mean=_TESSERA_S2_MEAN, std=_TESSERA_S2_STD, inplace=True)
)
_tessera_s1_transforms = torch.nn.Sequential(
    _PixelTimeSeriesNormalize(mean=_TESSERA_S1_MEAN, std=_TESSERA_S1_STD, inplace=True)
)


class TemporalAwarePooling(nn.Module):
    """Temporal-aware pooling with attention mechanism."""

    def __init__(self, input_dim: int) -> None:
        """Initialize a new TemporalAwarePooling instance.

        Args:
            input_dim: Input dimension for the attention query.
        """
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the temporal-aware pooling.

        Args:
            x: Input tensor of shape (B, seq_len, input_dim).

        Returns:
            Pooled tensor of shape (B, input_dim).
        """
        x_context, _ = self.temporal_context(x)
        w = torch.softmax(self.query(x_context), dim=1)
        output: Tensor = (w * x).sum(dim=1)
        return output


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding based on day of year."""

    def __init__(self, d_model: int) -> None:
        """Initialize a new TemporalPositionalEncoder instance.

        Args:
            d_model: Model embedding dimension.
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: Tensor) -> Tensor:
        """Forward pass of the temporal positional encoder.

        Args:
            doy: Day of year tensor of shape (B, T) with values 0-365.

        Returns:
            Positional encoding tensor of shape (B, T, d_model).
        """
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=doy.device)
            * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )

        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerEncoder(nn.Module):
    """Transformer encoder for pixel time series data."""

    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize a new TransformerEncoder instance.

        Args:
            band_num: Number of input spectral bands.
            latent_dim: Base latent dimension (will be multiplied by 4 internally).
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout probability.
        """
        super().__init__()
        input_dim = band_num

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )

        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.attn_pool = TemporalAwarePooling(latent_dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the transformer encoder.

        Args:
            x: Input tensor of shape (B, seq_len, bands + 1) where last channel is DOY.

        Returns:
            Encoded tensor of shape (B, latent_dim * 4).
        """
        bands = x[:, :, :-1]
        doy = x[:, :, -1]

        bands_embedded = self.embedding(bands)
        temporal_encoding = self.temporal_encoder(doy)

        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        output: Tensor = self.attn_pool(x)
        return output


class Tessera(nn.Module):
    """Tessera pixel time series foundation model.

    Tessera is a foundation model for pixel-level time series data from
    Sentinel-1 and Sentinel-2 satellites. It uses separate transformer
    encoders for SAR and optical data with temporal-aware pooling.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2506.20380

    .. versionadded:: 0.9
    """

    def __init__(self, embed_dim: int = 128) -> None:
        """Initialize a new Tessera instance.

        Args:
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.s2_backbone = TransformerEncoder(
            band_num=10,
            latent_dim=embed_dim,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=4096,
            dropout=0.1,
        )
        self.s1_backbone = TransformerEncoder(
            band_num=2,
            latent_dim=embed_dim,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=4096,
            dropout=0.1,
        )
        self.dim_reducer = nn.Sequential(nn.Linear(embed_dim * 8, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Tessera model.

        Args:
            x: Input tensor of shape (B, seq_len, 14) containing:
                - Channels 0-9: Sentinel-2 bands (B2-B12)
                - Channel 10: Sentinel-2 day of year
                - Channels 11-12: Sentinel-1 VV and VH
                - Channel 13: Sentinel-1 day of year

        Returns:
            Fused embedding tensor of shape (B, embed_dim).

        Raises:
            AssertionError: If input does not have 14 channels.
        """
        assert x.shape[-1] == 14, f'Expected 14 channels, got {x.shape[-1]}'
        s2_x, s1_x = x[..., :11], x[..., 11:]
        s2_feat = self.s2_backbone(s2_x)
        s1_feat = self.s1_backbone(s1_x)
        fused = torch.cat([s2_feat, s1_feat], dim=-1)
        output: Tensor = self.dim_reducer(fused)
        return output


class Tessera_Weights(WeightsEnum):
    """Tessera model weights.

    .. versionadded:: 0.9
    """

    TESSERA = Weights(
        url='https://hf.co/isaaccorley/tessera/resolve/acec3c1eb62d97a78b2cf65eb8cbf42587c57e93/model-b00edea0.pth',
        transforms=_tessera_transforms,
        meta={
            'dataset': 'TESSERA',
            'publication': 'https://arxiv.org/abs/2506.20380',
            'repo': 'https://github.com/ucam-eo/tessera',
            'bands': [
                'B2',
                'B3',
                'B4',
                'B5',
                'B6',
                'B7',
                'B8',
                'B8A',
                'B11',
                'B12',
                'S2_DOY',
                'VV',
                'VH',
                'S1_DOY',
            ],
            'in_chans': 14,
            'embed_dim': 128,
        },
    )

    TESSERA_SENTINEL2_ENCODER = Weights(
        url='https://hf.co/isaaccorley/tessera/resolve/acec3c1eb62d97a78b2cf65eb8cbf42587c57e93/s2_encoder-38fd63b9.pth',
        transforms=_tessera_s2_transforms,
        meta={
            'dataset': 'TESSERA',
            'publication': 'https://arxiv.org/abs/2506.20380',
            'repo': 'https://github.com/ucam-eo/tessera',
            'bands': [
                'B2',
                'B3',
                'B4',
                'B5',
                'B6',
                'B7',
                'B8',
                'B8A',
                'B11',
                'B12',
                'S2_DOY',
            ],
            'in_chans': 11,
            'embed_dim': 512,
        },
    )

    TESSERA_SENTINEL1_ENCODER = Weights(
        url='https://hf.co/isaaccorley/tessera/resolve/acec3c1eb62d97a78b2cf65eb8cbf42587c57e93/s1_encoder-7797f44d.pth',
        transforms=_tessera_s1_transforms,
        meta={
            'dataset': 'TESSERA',
            'publication': 'https://arxiv.org/abs/2506.20380',
            'repo': 'https://github.com/ucam-eo/tessera',
            'bands': ['VV', 'VH', 'S1_DOY'],
            'in_chans': 3,
            'embed_dim': 512,
        },
    )


def tessera(
    weights: Tessera_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """Tessera pixel time series foundation model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2506.20380

    .. versionadded:: 0.9

    Args:
        weights: Pre-trained model weights to use. If using encoder-only weights
            (``TESSERA_SENTINEL1_ENCODER`` or ``TESSERA_SENTINEL2_ENCODER``),
            returns the respective encoder backbone instead of the full model.
        *args: Additional arguments to pass to :class:`Tessera`.
        **kwargs: Additional keyword arguments to pass to :class:`Tessera`.

    Returns:
        A Tessera model or encoder backbone.
    """
    model = Tessera(*args, **kwargs)

    if weights is None:
        return model

    if weights == Tessera_Weights.TESSERA:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
        return model
    elif weights == Tessera_Weights.TESSERA_SENTINEL2_ENCODER:
        model.s2_backbone.load_state_dict(
            weights.get_state_dict(progress=True), strict=True
        )
        return model.s2_backbone
    elif weights == Tessera_Weights.TESSERA_SENTINEL1_ENCODER:
        model.s1_backbone.load_state_dict(
            weights.get_state_dict(progress=True), strict=True
        )
        return model.s1_backbone
    else:
        raise ValueError(f'Unsupported weights: {weights}')
