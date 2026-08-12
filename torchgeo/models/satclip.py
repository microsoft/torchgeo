# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/microsoft/satclip. Copyright (c) Microsoft Corporation.

"""SatCLIP location encoder."""

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
            legendre_polys: Number of Legendre polynomials. Higher values add
                higher-frequency spatial basis functions and increase the output
                feature dimension quadratically as ``legendre_polys ** 2``.
        """
        super().__init__()
        self.L = legendre_polys
        self.embedding_dim = self.L * self.L

        # d_m = product_{k=1}^m sqrt(1 + 1/(2k)); s_m = sqrt(2m + 3).
        orders = torch.arange(self.L, dtype=torch.float64)
        diag_coeffs = torch.ones(self.L, dtype=torch.float64)
        if self.L > 1:
            diag_terms = torch.sqrt(1 + 1 / (2 * orders[1:]))
            diag_coeffs[1:] = torch.cumprod(diag_terms, dim=0)

        subdiag_coeffs = torch.sqrt(2 * orders[:-1] + 3)

        alpha = torch.zeros((self.L, self.L), dtype=torch.float64)
        beta = torch.zeros((self.L, self.L), dtype=torch.float64)

        # c_l = (2l + 1)/(2l - 3)
        # alpha_lm = sqrt(c_l * (4(l - 1)^2 - 1)/(l^2 - m^2))
        # beta_lm = sqrt(c_l * ((l - 1)^2 - m^2)/(l^2 - m^2))
        for deg in range(2, self.L):
            m = torch.arange(deg - 1, dtype=torch.float64)
            prev_deg_sq = (deg - 1) ** 2
            order_sq = m**2

            deg_scale = (2 * deg + 1) / (2 * deg - 3)
            denom = deg**2 - order_sq
            alpha_num = 4 * prev_deg_sq - 1
            beta_num = prev_deg_sq - order_sq

            alpha_sq = deg_scale * (alpha_num / denom)
            beta_sq = deg_scale * (beta_num / denom)
            alpha[deg, : deg - 1] = torch.sqrt(alpha_sq)
            beta[deg, : deg - 1] = torch.sqrt(beta_sq)

        self.register_buffer('orders', orders, persistent=False)
        self.register_buffer('diag_coeffs', diag_coeffs, persistent=False)
        self.register_buffer('subdiag_coeffs', subdiag_coeffs, persistent=False)
        self.register_buffer('alpha', alpha, persistent=False)
        self.register_buffer('beta', beta, persistent=False)

    def _legendre_polynomials(
        self, x: Tensor, cos_theta: Tensor, sin_theta: Tensor, orders: Tensor
    ) -> list[Tensor]:
        """Compute the normalized associated Legendre polynomials.

        Args:
            x: Input coordinates used to create tensors with matching properties.
            cos_theta: Cosine of the colatitude for each coordinate.
            sin_theta: Sine of the colatitude for each coordinate.
            orders: Spherical harmonic orders in the input dtype.

        Returns:
            Associated Legendre polynomials for each degree. Row ``l`` has shape
            ``(B, l + 1)``.
        """
        dtype = x.dtype
        device = x.device
        batch_size = x.shape[0]

        p00 = x.new_full((batch_size,), math.sqrt(1 / (4 * math.pi)))
        diag_terms: list[Tensor] = [p00]
        if self.L > 1:
            diag_coeffs = self.diag_coeffs.to(device=device, dtype=dtype)
            diag_vals = rearrange(p00, 'b -> b 1')
            diag_vals = diag_vals * rearrange(diag_coeffs[1:], 'm -> 1 m')
            sin_theta = rearrange(sin_theta, 'b -> b 1')
            powers = rearrange(orders[1:], 'm -> 1 m')
            diag_vals = diag_vals * sin_theta.pow(powers)
            diag_terms.extend(diag_vals.unbind(dim=1))

        subdiag_coeffs = self.subdiag_coeffs.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)

        row = rearrange(diag_terms[0], 'b -> b 1')
        rows = [row]
        if self.L > 1:
            subdiag = subdiag_coeffs[0] * cos_theta
            subdiag = subdiag * diag_terms[0]
            subdiag = rearrange(subdiag, 'b -> b 1')
            diag = rearrange(diag_terms[1], 'b -> b 1')
            rows.append(torch.cat([subdiag, diag], dim=1))

        # P_l^m = alpha_lm cos(theta) P_{l-1}^m - beta_lm P_{l-2}^m.
        for deg in range(2, self.L):
            alpha_vals = rearrange(alpha[deg, : deg - 1], 'm -> 1 m')
            recurrence = alpha_vals * rearrange(cos_theta, 'b -> b 1')
            recurrence = recurrence * rows[deg - 1][:, : deg - 1]

            beta_vals = rearrange(beta[deg, : deg - 1], 'm -> 1 m')
            beta_term = beta_vals * rows[deg - 2][:, : deg - 1]
            recurrence = recurrence - beta_term

            subdiag = subdiag_coeffs[deg - 1] * cos_theta
            subdiag = subdiag * diag_terms[deg - 1]
            subdiag = subdiag.unsqueeze(1)
            diag = diag_terms[deg].unsqueeze(1)
            terms = [recurrence, subdiag, diag]
            rows.append(torch.cat(terms, dim=1))

        return rows

    def _real_spherical_harmonics(
        self, plm: list[Tensor], phi: Tensor, orders: Tensor
    ) -> Tensor:
        """Combine Legendre polynomials and longitude phases.

        Args:
            plm: Associated Legendre polynomials for each degree.
            phi: Longitude angle for each coordinate.
            orders: Spherical harmonic orders in the input dtype.

        Returns:
            Real spherical harmonic features of shape ``(B, L * L)``.
        """
        phases = rearrange(phi, 'b -> b 1')
        phases = phases * rearrange(orders, 'm -> 1 m')
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)

        # Y_l^0 = pi P_l^0; negative m use sin(m phi), positive m use cos(m phi).
        blocks: list[Tensor] = []
        for deg in range(self.L):
            row = plm[deg]
            center = math.pi * row[:, 0]
            center = center.unsqueeze(1)
            if deg == 0:
                blocks.append(center)
                continue

            base = math.sqrt(2) * row[:, 1:]
            negative = base * sin_terms[:, 1 : deg + 1]
            negative = torch.flip(negative, dims=(1,))
            positive = base * cos_terms[:, 1 : deg + 1]
            blocks.append(torch.cat([negative, center, positive], dim=1))

        return torch.cat(blocks, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Coordinates of shape ``(B, 2)`` as ``(longitude, latitude)`` in
                degrees.

        Returns:
            Spherical harmonic features of shape ``(B, L * L)``.
        """
        lon = x[:, 0]
        lat = x[:, 1]

        # Clamp latitudes off the poles to keep coordinate gradients finite.
        lat = lat.clamp(min=-89.95, max=89.95)
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        cos_theta = torch.cos(theta)

        sin_theta = 1 - cos_theta * cos_theta
        sin_theta = torch.clamp(sin_theta, min=0)
        sin_theta = torch.sqrt(sin_theta)

        orders = self.orders.to(device=x.device, dtype=x.dtype)
        plm = self._legendre_polynomials(x, cos_theta, sin_theta, orders)
        return self._real_spherical_harmonics(plm, phi, orders)


class SineActivation(nn.Module):
    """Sine activation with frequency scaling.

    .. versionadded:: 0.10
    """

    def __init__(self, w0: float = 1.0) -> None:
        """Initialize a new SineActivation instance.

        Args:
            w0: Frequency scaling factor applied before the sine activation.
                Higher values make the activation vary more rapidly with its input.
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

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2006.09661

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
            dim_in: Number of input features in tensors of shape ``(..., dim_in)``.
            dim_out: Number of output features in tensors of shape
                ``(..., dim_out)``.
            w0: Frequency scaling factor. Higher values let the layer represent
                higher-frequency variation, but also shrink the initialization range
                for non-first layers.
            c: Weight initialization scale constant. Higher values widen the
                initialization range for non-first layers.
            is_first: Whether this is the first SIREN layer. First layers use a
                wider initialization that preserves high-frequency coordinate
                information.
            use_bias: Whether to include a bias term.
            activation: Activation module applied after the linear layer. Defaults
                to :class:`SineActivation`.
            dropout: Whether to apply dropout to the linear output before the
                activation.
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
            dim_in: Number of input features in tensors of shape ``(B, dim_in)``.
            dim_hidden: Number of hidden features in each SIREN layer. Higher
                values increase network capacity and memory use.
            dim_out: Number of output features in tensors of shape ``(B, dim_out)``.
            num_layers: Number of hidden SIREN layers. Higher values increase
                network depth and the number of nonlinear coordinate transforms.
            w0: Frequency scaling for hidden layers. Higher values allow more
                rapidly varying hidden representations.
            w0_initial: Frequency scaling for the first layer. Higher values make
                the network more sensitive to fine-scale input-coordinate changes.
            use_bias: Whether to include bias terms.
            final_activation: Activation applied to the last layer.
        """
        super().__init__()
        layers = []
        for index in range(num_layers):
            is_first = index == 0
            layer_dim = dim_in if is_first else dim_hidden
            layer_w0 = w0_initial if is_first else w0
            layers.append(
                Siren(
                    dim_in=layer_dim,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=True,
                )
            )
        self.layers = nn.ModuleList(layers)
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

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1609/aaai.v39i4.32457

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
            legendre_polys: Number of Legendre polynomials used by the spherical
                harmonics encoder. Higher values preserve finer spatial variation
                and increase the positional encoding size as ``legendre_polys ** 2``.
                Must be positive.
            capacity: Number of hidden features in the SIREN network. Higher values
                increase model capacity and memory use. Must be positive.
            embed_dim: Number of output embedding features. Must be positive.
            num_hidden_layers: Number of hidden SIREN layers. Higher values increase
                network depth and coordinate-expression capacity. Must be positive.

        """
        super().__init__()
        assert legendre_polys > 0
        assert capacity > 0
        assert embed_dim > 0
        assert num_hidden_layers > 0

        self.posenc = SphericalHarmonics(legendre_polys).to(
            dtype=torch.get_default_dtype()
        )
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
                Coordinates are cast to the network parameter dtype.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        dtype = self.nnet.last_layer.weight.dtype
        x = x.to(dtype=dtype)
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
            'publication': 'https://doi.org/10.1609/aaai.v39i4.32457',
            'repo': 'https://github.com/microsoft/satclip',
            'ssl_method': 'clip',
            'image_encoder': 'vit16-l40',
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

    * https://doi.org/10.1609/aaai.v39i4.32457

    .. versionadded:: 0.10

    Args:
        weights: Pre-trained SatCLIP weights to load. If provided, the returned
            model is set to eval mode.
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
        checkpoint = weights.get_state_dict(
            progress=True, map_location='cpu', check_hash=True, weights_only=True
        )
        prefix = 'model.location.'
        state_dict = {
            key[len(prefix) :]: value
            for key, value in checkpoint['state_dict'].items()
            if key.startswith(prefix + 'nnet.')
        }
        model.load_state_dict(state_dict, strict=True)
        model.eval()

    return model
