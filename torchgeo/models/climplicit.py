# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
#
# Based on the original code: https://github.com/ecovision-uzh/climplicit

"""Climplicit climatic implicit location encoder."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models._api import Weights, WeightsEnum

# CHELSA bioclimatic variables reconstructed by the regression head, in output order.
_CHELSA_VARIABLES = (
    'cmi',
    'clt',
    'hurs',
    'pet',
    'pr',
    'rsds',
    'sfcWind',
    'tas',
    'tasmax',
    'tasmin',
    'vpd',
)

# Per-variable mean/std used to de-standardize the CHELSA reconstruction, shape (1, 11).
_CHELSA_MEAN = torch.tensor(
    [
        -264.1493656,
        3912.44628016,
        5921.65964573,
        9385.47468266,
        697.03653109,
        15219.37926928,
        3498.8511804,
        2819.56006368,
        2864.08583811,
        2773.46759638,
        8039.37322797,
    ]
).unsqueeze(0)
_CHELSA_STD = torch.tensor(
    [
        1042.67560332,
        1767.94018571,
        1185.91587823,
        6639.79069994,
        883.56243405,
        7843.49167037,
        1637.09237995,
        174.43791946,
        181.69448751,
        167.07485901,
        7516.98198719,
    ]
).unsqueeze(0)


class Direct(nn.Module):
    """Direct positional encoding.

    Linearly rescales ``(lon, lat)`` coordinates in degrees to the ``[-1, 1]`` range
    expected by a SIREN, based on the supported coordinate extent.
    """

    def __init__(
        self,
        lon_min: float = -180.0,
        lon_max: float = 180.0,
        lat_min: float = -90.0,
        lat_max: float = 90.0,
    ) -> None:
        """Initialize a new Direct instance.

        Args:
            lon_min: Minimum longitude in degrees.
            lon_max: Maximum longitude in degrees.
            lat_min: Minimum latitude in degrees.
            lat_max: Maximum latitude in degrees.
        """
        super().__init__()
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def forward(self, coords: Tensor) -> Tensor:
        """Rescale coordinates to ``[-1, 1]``.

        Args:
            coords: Coordinate tensor of shape (B, 2) holding ``(lon, lat)`` in degrees.

        Returns:
            Rescaled coordinate tensor of shape (B, 2).
        """
        lon, lat = coords[:, 0], coords[:, 1]
        lon = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1
        lat = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1
        return torch.stack([lon, lat], dim=1).float()


class Sine(nn.Module):
    """Sine activation with a configurable angular frequency ``w0``."""

    def __init__(self, w0: float = 1.0) -> None:
        """Initialize a new Sine instance.

        Args:
            w0: Angular frequency of the sine activation.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        """Apply the sine activation.

        Args:
            x: Input tensor.

        Returns:
            ``sin(w0 * x)``.
        """
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """A single SIREN layer with an optional residual connection and H-SIREN activation.

    When ``residual_connections`` is True, the pre-activation vector of the previous layer
    is averaged into the current pre-activation vector (only when their shapes match),
    keeping the residual stream within SIREN's distributional constraints. When ``h_siren``
    is True, the first layer applies ``sin(sinh(2x))`` instead of ``sin(x)``, which widens
    the supported frequency set and reduces over-smoothing.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        activation: nn.Module | None = None,
        residual_connections: bool = False,
        h_siren: bool = False,
    ) -> None:
        """Initialize a new Siren instance.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            w0: Angular frequency of the sine activation.
            c: Constant used to scale the weight initialization variance.
            is_first: Whether this is the first layer of the network.
            activation: Activation module to use. Defaults to :class:`Sine`.
            residual_connections: Whether to enable the ReSIREN residual connection.
            h_siren: Whether to use the H-SIREN first-layer activation.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.h_siren = h_siren
        self.residual_connections = residual_connections

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out)
        w_std = (1 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        weight.uniform_(-w_std, w_std)
        bias.uniform_(-w_std, w_std)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.activation = Sine(w0) if activation is None else activation

    def forward(
        self, x: Tensor, prev_gaussian: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the SIREN layer.

        Args:
            x: Input tensor of shape (B, dim_in).
            prev_gaussian: Pre-activation vector of the previous layer, used for the
                residual connection.

        Returns:
            A tuple of the activated output and the pre-activation vector (gaussian).
        """
        out = F.linear(x, self.weight, self.bias)
        if (
            self.residual_connections
            and prev_gaussian is not None
            and out.shape == prev_gaussian.shape
        ):
            out = (out + prev_gaussian) / 2
        gaussian = out
        if self.h_siren and self.is_first:
            out = torch.sinh(2 * out)
        out = self.activation(out)
        return out, gaussian


class SirenNet(nn.Module):
    """Sinusoidal Representation Network (SIREN) with residual connections (ReSIREN).

    Adapted from the SatCLIP location encoder
    (https://github.com/microsoft/satclip/blob/main/satclip/location_encoder.py) and
    extended with residual connections and the H-SIREN first-layer activation.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        h_siren: bool = False,
        residual_connections: bool = False,
    ) -> None:
        """Initialize a new SirenNet instance.

        Args:
            dim_in: Input dimension.
            dim_hidden: Hidden dimension of every SIREN layer.
            dim_out: Output (embedding) dimension.
            num_layers: Number of hidden SIREN layers.
            w0: Angular frequency for the hidden layers.
            w0_initial: Angular frequency for the first layer.
            h_siren: Whether to use the H-SIREN first-layer activation.
            residual_connections: Whether to enable ReSIREN residual connections.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for ind in range(num_layers):
            is_first = ind == 0
            self.layers.append(
                Siren(
                    dim_in=dim_in if is_first else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if is_first else w0,
                    is_first=is_first,
                    h_siren=h_siren,
                    residual_connections=residual_connections,
                )
            )
        self.last_layer = Siren(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, activation=nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the network.

        Args:
            x: Input tensor of shape (B, dim_in).

        Returns:
            Output embedding of shape (B, dim_out).
        """
        gaussian = None
        for layer in self.layers:
            x, gaussian = layer(x, gaussian)
        x, _ = self.last_layer(x, gaussian)
        return x


class Climplicit(nn.Module):
    """Climplicit climatic implicit location encoder.

    Climplicit encodes ``(lon, lat)`` coordinates (and optionally a month) into a 256-d
    climatic embedding using a residual H-SIREN network pretrained to reconstruct CHELSA
    bioclimatic variables. When no month is provided, embeddings for March, June,
    September, and December are concatenated into a single 1024-d embedding. The model can
    optionally return the de-standardized CHELSA reconstruction instead of the embedding.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2504.05089

    .. versionadded:: 0.10
    """

    def __init__(self, return_chelsa: bool = False) -> None:
        """Initialize a new Climplicit instance.

        Args:
            return_chelsa: If True, return the de-standardized CHELSA reconstruction
                instead of the climatic embedding.
        """
        super().__init__()
        self.return_chelsa = return_chelsa
        self.pos_embedding = Direct(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)
        self.location_encoder = SirenNet(
            dim_in=4,
            dim_hidden=512,
            dim_out=256,
            num_layers=16,
            h_siren=True,
            residual_connections=True,
        )
        self.chelsa_regressor = nn.Linear(256, len(_CHELSA_VARIABLES))
        self.register_buffer('chelsa_mean', _CHELSA_MEAN, persistent=False)
        self.register_buffer('chelsa_std', _CHELSA_STD, persistent=False)

    def _embed(self, loc: Tensor, month: Tensor) -> Tensor:
        """Encode rescaled coordinates for a single month.

        Args:
            loc: Rescaled coordinate tensor of shape (B, 2).
            month: Month tensor of shape (B,) with values in ``[1, 12]``.

        Returns:
            The climatic embedding, or the CHELSA reconstruction if ``return_chelsa``.
        """
        angle = month / 12 * math.pi * 2
        loc_month = torch.cat(
            [
                loc,
                torch.sin(angle).unsqueeze(dim=-1),
                torch.cos(angle).unsqueeze(dim=-1),
            ],
            dim=-1,
        )
        x = self.location_encoder(loc_month)
        if self.return_chelsa:
            x = self.chelsa_regressor(x)
            x = x * self.chelsa_std + self.chelsa_mean
        return x

    def forward(self, coordinates: Tensor, month: Tensor | None = None) -> Tensor:
        """Forward pass of the Climplicit model.

        Args:
            coordinates: Coordinate tensor of shape (B, 2) holding ``(lon, lat)`` in
                degrees.
            month: Optional month tensor of shape (B,) with values in ``[1, 12]``. If
                None, embeddings for March, June, September, and December are concatenated.

        Returns:
            A climatic embedding of shape (B, 256) (or (B, 1024) if *month* is None), or
            the CHELSA reconstruction of shape (B, 11) (or (B, 44)) if ``return_chelsa``.
        """
        loc = self.pos_embedding(coordinates)
        if month is None:
            res = []
            for m in (3, 6, 9, 12):
                months = torch.ones(coordinates.shape[0], device=coordinates.device) * m
                res.append(self._embed(loc, months))
            return torch.cat(res, dim=-1)
        return self._embed(loc, month)


# Transforms are a no-op: Climplicit consumes raw (lon, lat) coordinates directly.
_climplicit_transforms = nn.Identity()


class Climplicit_Weights(WeightsEnum):
    """Climplicit model weights.

    .. versionadded:: 0.10
    """

    CHELSA = Weights(
        url='https://hf.co/Jobedo/climplicit/resolve/main/climplicit-3341daa2.pth',
        transforms=_climplicit_transforms,
        meta={
            'dataset': 'CHELSA',
            'publication': 'https://arxiv.org/abs/2504.05089',
            'repo': 'https://github.com/ecovision-uzh/climplicit',
            'variables': list(_CHELSA_VARIABLES),
            'embed_dim': 256,
        },
    )


def climplicit(
    weights: Climplicit_Weights | None = None, *args: Any, **kwargs: Any
) -> Climplicit:
    """Climplicit climatic implicit location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2504.05089

    .. versionadded:: 0.10

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Climplicit`.
        **kwargs: Additional keyword arguments to pass to :class:`Climplicit`.

    Returns:
        A Climplicit model.
    """
    model = Climplicit(*args, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)

    return model
