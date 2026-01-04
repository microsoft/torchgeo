# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tessera pixel time series foundation model.

Reference implementation:

* https://github.com/developmentseed/pixelverse
"""

from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models._api import Weights, WeightsEnum


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

    Reference implementation:

    * https://github.com/ucam-eo/tessera

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.00557

    .. versionadded:: 0.8
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


class _TesseraTransforms(nn.Module):
    """Transforms for Tessera model normalization."""

    mean: Tensor
    std: Tensor

    def __init__(self, mean: list[float], std: list[float]) -> None:
        """Initialize a new _TesseraTransforms instance.

        Args:
            mean: Mean values for each channel.
            std: Standard deviation values for each channel.
        """
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, x: Tensor) -> Tensor:
        """Apply normalization to input tensor.

        Args:
            x: Input tensor of shape (..., C).

        Returns:
            Normalized tensor.
        """
        output: Tensor = (x - self.mean) / self.std
        return output


# Sentinel-2 and Sentinel-1 band statistics
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
_TESSERA_MEAN = [*_TESSERA_S2_MEAN, *_TESSERA_S1_MEAN]
_TESSERA_STD = [*_TESSERA_S2_STD, *_TESSERA_S1_STD]

_tessera_transforms = _TesseraTransforms(mean=_TESSERA_MEAN, std=_TESSERA_STD)
_tessera_s2_transforms = _TesseraTransforms(mean=_TESSERA_S2_MEAN, std=_TESSERA_S2_STD)
_tessera_s1_transforms = _TesseraTransforms(mean=_TESSERA_S1_MEAN, std=_TESSERA_S1_STD)


class Tessera_Weights(WeightsEnum):  # type: ignore[misc]
    """Tessera model weights.

    .. versionadded:: 0.8
    """

    TESSERA = Weights(
        url='https://hf.co/isaaccorley/tessera/resolve/51afe75b724d387ef9fcb6f6e090a5be0b906919/model.pt',
        transforms=_tessera_transforms,
        meta={
            'dataset': 'Major TOM',
            'model': 'tessera',
            'publication': 'https://arxiv.org/abs/2503.00557',
            'repo': 'https://github.com/ucam-eo/tessera',
            'ssl_method': 'contrastive',
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


class Tessera_S2_Encoder_Weights(WeightsEnum):  # type: ignore[misc]
    """Tessera Sentinel-2 encoder weights.

    .. versionadded:: 0.8
    """

    TESSERA = Weights(
        url='https://hf.co/isaaccorley/tessera/resolve/11dda783c258148bc6342832df6ef8dc05963702/s2_encoder.pt',
        transforms=_tessera_s2_transforms,
        meta={
            'dataset': 'Major TOM',
            'model': 'tessera_s2_encoder',
            'publication': 'https://arxiv.org/abs/2503.00557',
            'repo': 'https://github.com/ucam-eo/tessera',
            'ssl_method': 'contrastive',
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


class Tessera_S1_Encoder_Weights(WeightsEnum):  # type: ignore[misc]
    """Tessera Sentinel-1 encoder weights.

    .. versionadded:: 0.8
    """

    TESSERA = Weights(
        url='https://hf.co/isaaccorley/tessera/resolve/439ae74f34d3db458976138907302ac1b2ca4903/s1_encoder.pt',
        transforms=_tessera_s1_transforms,
        meta={
            'dataset': 'Major TOM',
            'model': 'tessera_s1_encoder',
            'publication': 'https://arxiv.org/abs/2503.00557',
            'repo': 'https://github.com/ucam-eo/tessera',
            'ssl_method': 'contrastive',
            'bands': ['VV', 'VH', 'S1_DOY'],
            'in_chans': 3,
            'embed_dim': 512,
        },
    )


def tessera(
    weights: Tessera_Weights | None = None, *args: Any, **kwargs: Any
) -> Tessera:
    """Tessera pixel time series foundation model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.00557

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Tessera`.
        **kwargs: Additional keyword arguments to pass to :class:`Tessera`.

    Returns:
        A Tessera model.
    """
    model = Tessera(*args, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    return model


def tessera_s2_encoder(
    weights: Tessera_S2_Encoder_Weights | None = None, *args: Any, **kwargs: Any
) -> TransformerEncoder:
    """Tessera Sentinel-2 transformer encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.00557

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Tessera`.
        **kwargs: Additional keyword arguments to pass to :class:`Tessera`.

    Returns:
        A TransformerEncoder for Sentinel-2 data.
    """
    model = Tessera(*args, **kwargs).s2_backbone
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    return model


def tessera_s1_encoder(
    weights: Tessera_S1_Encoder_Weights | None = None, *args: Any, **kwargs: Any
) -> TransformerEncoder:
    """Tessera Sentinel-1 transformer encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.00557

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Tessera`.
        **kwargs: Additional keyword arguments to pass to :class:`Tessera`.

    Returns:
        A TransformerEncoder for Sentinel-1 data.
    """
    model = Tessera(*args, **kwargs).s1_backbone
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    return model
