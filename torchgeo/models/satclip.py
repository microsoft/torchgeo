# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/microsoft/satclip. Copyright (c) Microsoft Corporation.

"""SatCLIP location encoder.

Reference:
    Klemmer et al., SatCLIP: Global, General-Purpose Location Embeddings with
    Satellite Imagery
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchvision.models._api import Weights, WeightsEnum


class SphericalHarmonics(nn.Module):
    """Real spherical harmonics positional encoding.

    .. versionadded:: 0.10
    """

    orders: Tensor
    diag_coeffs: Tensor
    subdiag_coeffs: Tensor
    alpha: Tensor
    beta: Tensor

    def __init__(self, legendre_polys: int) -> None:
        """Initialize a new SphericalHarmonics instance.

        Args:
            legendre_polys: Number of Legendre polynomials.
        """
        super().__init__()
        self.L = int(legendre_polys)
        self.embedding_dim = self.L * self.L

        orders = torch.arange(self.L, dtype=torch.int64)
        diag_coeffs = torch.ones(self.L, dtype=torch.float64)
        if self.L > 1:
            diag_terms = torch.sqrt(1.0 + 1.0 / (2.0 * orders[1:].to(torch.float64)))
            diag_coeffs[1:] = torch.cumprod(diag_terms, dim=0)
        subdiag_coeffs = torch.sqrt(2.0 * orders[:-1].to(torch.float64) + 3.0)
        alpha = torch.zeros((self.L, self.L), dtype=torch.float64)
        beta = torch.zeros((self.L, self.L), dtype=torch.float64)
        for degree in range(2, self.L):
            m = torch.arange(degree - 1, dtype=torch.float64)
            alpha[degree, : degree - 1] = torch.sqrt(
                ((2.0 * degree + 1.0) / (2.0 * degree - 3.0))
                * (
                    (4.0 * (degree - 1) * (degree - 1) - 1.0)
                    / (degree * degree - m * m)
                )
            )
            beta[degree, : degree - 1] = torch.sqrt(
                ((2.0 * degree + 1.0) / (2.0 * degree - 3.0))
                * ((((degree - 1) * (degree - 1)) - m * m) / (degree * degree - m * m))
            )

        self.register_buffer('orders', orders, persistent=False)
        self.register_buffer('diag_coeffs', diag_coeffs, persistent=False)
        self.register_buffer('subdiag_coeffs', subdiag_coeffs, persistent=False)
        self.register_buffer('alpha', alpha, persistent=False)
        self.register_buffer('beta', beta, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Coordinates of shape ``(B, 2)`` as ``(longitude, latitude)`` in degrees.

        Returns:
            Spherical harmonic features of shape ``(B, L * L)``.
        """
        # Clamp latitudes off the poles to keep the recurrence stable.
        lon = x[:, 0]
        lat = x[:, 1].clamp(min=-89.95, max=89.95)
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta * cos_theta, min=0))

        dtype = x.dtype
        device = x.device
        batch_size = x.shape[0]
        orders = self.orders.to(device=device)
        order_values = orders.to(dtype=dtype)

        p00 = x.new_full((batch_size,), math.sqrt(1.0 / (4.0 * math.pi)))
        diag_terms: list[Tensor] = [p00]
        if self.L > 1:
            diag_coeffs = self.diag_coeffs.to(device=device, dtype=dtype)
            diag_values = (
                rearrange(p00, 'b -> b 1')
                * rearrange(diag_coeffs[1:], 'm -> 1 m')
                * rearrange(sin_theta, 'b -> b 1').pow(
                    rearrange(order_values[1:], 'm -> 1 m')
                )
            )
            diag_terms.extend(diag_values.unbind(dim=1))

        subdiag_coeffs = self.subdiag_coeffs.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        rows: list[Tensor] = [
            torch.cat(
                [diag_terms[0].unsqueeze(1), x.new_zeros((batch_size, self.L - 1))],
                dim=1,
            )
        ]
        if self.L > 1:
            rows.append(
                torch.cat(
                    [
                        (subdiag_coeffs[0] * cos_theta * diag_terms[0]).unsqueeze(1),
                        diag_terms[1].unsqueeze(1),
                        x.new_zeros((batch_size, self.L - 2)),
                    ],
                    dim=1,
                )
            )
        for degree in range(2, self.L):
            recurrence = (
                rearrange(alpha[degree, : degree - 1], 'm -> 1 m')
                * rearrange(cos_theta, 'b -> b 1')
                * rows[degree - 1][:, : degree - 1]
                - rearrange(beta[degree, : degree - 1], 'm -> 1 m')
                * rows[degree - 2][:, : degree - 1]
            )
            row_terms: list[Tensor] = [
                recurrence,
                (
                    subdiag_coeffs[degree - 1] * cos_theta * diag_terms[degree - 1]
                ).unsqueeze(1),
                diag_terms[degree].unsqueeze(1),
            ]
            if degree < self.L - 1:
                row_terms.append(x.new_zeros((batch_size, self.L - degree - 1)))
            rows.append(torch.cat(row_terms, dim=1))

        plm = torch.stack(rows, dim=1)
        phases = rearrange(phi, 'b -> b 1') * rearrange(order_values, 'm -> 1 m')
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)
        degree_blocks: list[Tensor] = []
        for degree in range(self.L):
            center = (math.pi * plm[:, degree, 0]).unsqueeze(1)
            if degree == 0:
                degree_blocks.append(center)
                continue
            m = orders[1 : degree + 1]
            base = math.sqrt(2.0) * plm[:, degree, 1 : degree + 1]
            negative = torch.flip(base * sin_terms[:, m], dims=(1,))
            positive = base * cos_terms[:, m]
            degree_blocks.append(torch.cat([negative, center, positive], dim=1))
        return torch.cat(degree_blocks, dim=1)


class SineActivation(nn.Module):
    """Sine activation with frequency scaling.

    .. versionadded:: 0.10
    """

    def __init__(self, w0: float = 1.0) -> None:
        """Initialize a new SineActivation instance.

        Args:
            w0: Frequency scaling factor.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """SIREN linear layer with sine activation.

    Reference:
        Sitzmann et al., Implicit Neural Representations with Periodic
        Activation Functions

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn.Module | None = None,
        dropout: bool = False,
    ) -> None:
        """Initialize a new Siren instance.

        Args:
            dim_in: Number of input dimensions.
            dim_out: Number of output dimensions.
            w0: Frequency scaling factor.
            c: Weight initialization scale constant.
            is_first: Whether this is the first SIREN layer.
            use_bias: Whether to include a bias term.
            activation: Activation module.
            dropout: Whether to apply dropout.
        """
        super().__init__()
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        scale = (1 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        weight.uniform_(-scale, scale)
        if bias is not None:
            bias.uniform_(-scale, scale)

        self.weight = nn.Parameter(weight)
        self.bias: nn.Parameter | None = (
            nn.Parameter(bias) if bias is not None else None
        )
        self.activation = SineActivation(w0) if activation is None else activation

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        return self.activation(out)


class SirenNet(nn.Module):
    """Multilayer SIREN network.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: nn.Module | None = None,
    ) -> None:
        """Initialize a new SirenNet instance.

        Args:
            dim_in: Number of input dimensions.
            dim_hidden: Number of hidden dimensions.
            dim_out: Number of output dimensions.
            num_layers: Number of hidden SIREN layers.
            w0: Frequency scaling for hidden layers.
            w0_initial: Frequency scaling for the first layer.
            use_bias: Whether to include bias terms.
            final_activation: Activation applied to the last layer.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Siren(
                    dim_in=dim_in if index == 0 else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if index == 0 else w0,
                    use_bias=use_bias,
                    is_first=index == 0,
                    dropout=True,
                )
                for index in range(num_layers)
            ]
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=nn.Identity() if final_activation is None else final_activation,
            dropout=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, dim_in)``.

        Returns:
            Output tensor of shape ``(B, dim_out)``.
        """
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class SatCLIP(nn.Module):
    """SatCLIP location encoder.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        legendre_polys: int = 40,
        capacity: int = 512,
        embed_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        """Initialize a new SatCLIP instance.

        Args:
            legendre_polys: Number of Legendre polynomials.
            capacity: Number of hidden dimensions.
            embed_dim: Number of output dimensions.
            num_hidden_layers: Number of hidden SIREN layers.
        """
        super().__init__()
        self.posenc = SphericalHarmonics(legendre_polys)
        self.nnet = SirenNet(
            dim_in=self.posenc.embedding_dim,
            dim_hidden=capacity,
            dim_out=embed_dim,
            num_layers=num_hidden_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Coordinates of shape ``(B, 2)`` as ``(longitude, latitude)`` in degrees.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        return self.nnet(self.posenc(x))


class SatCLIP_Weights(WeightsEnum):
    """SatCLIP location encoder weights.

    .. versionadded:: 0.10
    """

    SATCLIP_VIT16_L40 = Weights(
        url='https://huggingface.co/microsoft/SatCLIP-ViT16-L40/resolve/0ef2acc0b91d3c8cfdf4a0cc03207095989287ab/satclip-vit16-l40.ckpt',
        transforms=nn.Identity(),
        meta={
            'dataset': 'S2-100K',
            'model': 'satclip',
            'publication': 'https://arxiv.org/abs/2311.17179',
            'repo': 'https://github.com/microsoft/satclip',
            'ssl_method': 'clip',
            'image_encoder': 'vit16',
            'legendre_polys': 40,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
        },
    )


def satclip(
    weights: SatCLIP_Weights | None = None, *args: Any, **kwargs: Any
) -> SatCLIP:
    """SatCLIP location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.17179

    .. versionadded:: 0.10

    Args:
        weights: Pre-trained SatCLIP weights to load.
        *args: Additional arguments to pass to :class:`SatCLIP`.
        **kwargs: Additional keyword arguments to pass to :class:`SatCLIP`.

    Returns:
        A SatCLIP location encoder.
    """
    if weights:
        kwargs['legendre_polys'] = weights.meta['legendre_polys']
        kwargs['capacity'] = weights.meta['capacity']
        kwargs['embed_dim'] = weights.meta['embed_dim']
        kwargs['num_hidden_layers'] = weights.meta['num_hidden_layers']

    model = SatCLIP(*args, **kwargs)

    if weights:
        # SatCLIP releases ship as Lightning checkpoints with the location encoder
        # weights under the 'model.location.nnet.*' namespace.
        checkpoint = weights.get_state_dict(progress=True)
        prefix = 'model.location.'
        state_dict = {
            key[len(prefix) :]: value
            for key, value in checkpoint['state_dict'].items()
            if key.startswith(prefix + 'nnet.')
        }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        assert missing_keys == []
        assert unexpected_keys == []

    return model
