# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# Copyright (c) 2021 VSainteuf (Vivien Sainte Fare Garnot)
# Adapted from https://github.com/VSainteuf/utae-paps
# This implementation differs significantly from the original: it uses einops
# for tensor operations, TorchGeo's LTAE2d, typed configuration validation, and
# explicit temporal-padding handling.

"""U-Net with Lightweight Temporal Attention Encoder (U-TAE)."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, cast

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn

from .ltae import LTAE2d


class UTAE(nn.Module):
    """U-Net with Lightweight Temporal Attention Encoder (U-TAE).

    Spatio-temporal encoder for satellite image time series. A shared L-TAE
    is applied at the U-Net bottleneck; per-head attention masks are
    propagated to the skip connections via a *TemporalAggregator*.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2107.07933

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        input_dim: int,
        encoder_widths: Sequence[int] = (64, 64, 64, 128),
        decoder_widths: Sequence[int] | None = (32, 32, 64, 128),
        out_conv: Sequence[int] = (32, 20),
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        encoder_norm: str = 'group',
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder: bool = False,
        return_maps: bool = False,
        pad_value: float = 0,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'reflect',
    ) -> None:
        """Initialize U-TAE.

        Args:
            input_dim: Number of channels in the input images.
            encoder_widths: Number of channels at each encoder stage, ordered
                from highest to lowest resolution.
            decoder_widths: Number of channels at each decoder stage. If
                ``None``, mirrors the encoder widths.
            out_conv: Channel widths for the final output convolutions.
            str_conv_k: Kernel size for strided up/down convolutions.
            str_conv_s: Stride for strided up/down convolutions.
            str_conv_p: Padding for strided up/down convolutions.
            encoder_norm: Normalisation layer for the encoder. One of
                ``'group'``, ``'batch'``, ``'instance'``, or ``'none'``.
            n_head: Number of attention heads in L-TAE.
            d_model: Projection dimension for L-TAE.
            d_k: Key/query dimension for L-TAE.
            encoder: If ``True``, return intermediate feature maps instead of
                class scores.
            return_maps: If ``True``, return intermediate feature maps
                alongside the output.
            pad_value: Value used by the dataloader for temporal padding.
            padding_mode: Spatial padding strategy for convolutions.

        Raises:
            AssertionError: If *encoder_widths* and *decoder_widths* have different
                lengths or different final widths.
        """
        super().__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = list(encoder_widths)
        self.decoder_widths = (
            list(decoder_widths) if decoder_widths is not None else list(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        assert len(self.encoder_widths) == len(self.decoder_widths), (
            'encoder_widths and decoder_widths must have the same length'
        )
        assert self.encoder_widths[-1] == self.decoder_widths[-1], (
            'encoder_widths and decoder_widths must have the same final width'
        )

        self.in_conv = ConvBlock(
            nkernels=[input_dim, self.encoder_widths[0], self.encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=self.encoder_widths[i],
                d_out=self.encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=self.decoder_widths[i],
                d_out=self.decoder_widths[i - 1],
                d_skip=self.encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm='batch',
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=self.encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=(d_model, self.encoder_widths[-1]),
            d_k=d_k,
        )
        self.temporal_aggregator = TemporalAggregator()
        self.out_conv = ConvBlock(
            nkernels=[self.decoder_widths[0], *out_conv], padding_mode=padding_mode
        )

    def forward(
        self, x: Tensor, batch_positions: Tensor | None = None, return_att: bool = False
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, C, H, W)``.
            batch_positions: Acquisition dates of shape ``(B, T)``.
            return_att: If ``True``, return attention masks alongside output.

        Returns:
            Output tensor of shape ``(B, out_conv[-1], H, W)``. If *return_att*
            is ``True``, also returns attention masks of shape
            ``(n_head, B, T, H_att, W_att)``. If *return_maps* or *encoder* is
            enabled, instead also returns the decoder feature maps.
        """
        pad_mask = reduce(x == self.pad_value, 'b t c h w -> b t', 'all')

        out = self.in_conv.smart_forward(x)
        feature_maps = [out]

        for i in range(self.n_stages - 1):
            out = cast(TemporallySharedBlock, self.down_blocks[i]).smart_forward(
                feature_maps[-1]
            )
            feature_maps.append(out)

        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )

        maps: list[Tensor] = []
        if self.return_maps:
            maps = [out]

        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps

        out = self.out_conv(out)
        if return_att:
            return out, att
        if self.return_maps:
            return out, maps
        return out


class TemporalAggregator(nn.Module):
    """Aggregate a temporal sequence of feature maps into a single frame.

    Builds the U-Net skip connection in U-TAE by collapsing the temporal
    dimension of encoder feature maps. The aggregation uses per-head L-TAE
    attention masks to weight the corresponding channel groups in the skip
    connection.

    """

    def forward(
        self, x: Tensor, pad_mask: Tensor | None = None, attn_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Feature maps of shape ``(B, T, C, H, W)``.
            pad_mask: Boolean mask of shape ``(B, T)``.
            attn_mask: Attention weights of shape
                ``(n_head, B, T, H_att, W_att)``.

        Returns:
            Aggregated feature map of shape ``(B, C, H, W)``.

        Raises:
            ValueError: If *attn_mask* is missing or the channel dimension of
                *x* is not divisible by the number of attention heads.
        """
        if attn_mask is None:
            raise ValueError('attn_mask is required for temporal aggregation')

        n_heads, batch_size, _, height, width = attn_mask.shape
        attn = rearrange(attn_mask, 'heads b t h w -> (heads b) t h w')
        if x.shape[-2:] != (height, width):
            attn = F.interpolate(
                attn, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
        attn = rearrange(
            attn, '(heads b) t h w -> heads b t h w', heads=n_heads, b=batch_size
        )

        if pad_mask is not None and pad_mask.any():
            # Ignore temporally padded frames so fixed-length batches do not
            # leak padding images into U-Net skip connections.
            valid = rearrange((~pad_mask).float(), 'b t -> 1 b t 1 1')
            attn = attn * valid

        self._check_att_group_channels(x, n_heads)
        out = rearrange(
            x, 'b t (heads channels) h w -> heads b t channels h w', heads=n_heads
        )
        attn = rearrange(attn, 'heads b t h w -> heads b t 1 h w')
        return reduce(
            attn * out, 'heads b t channels h w -> b (heads channels) h w', 'sum'
        )

    def _check_att_group_channels(self, x: Tensor, n_heads: int) -> None:
        """Validate channel grouping for ``att_group`` aggregation.

        Args:
            x: Feature maps of shape ``(B, T, C, H, W)``.
            n_heads: Number of temporal attention heads.

        Raises:
            ValueError: If the channel dimension of *x* is not divisible by *n_heads*.
        """
        if x.shape[2] % n_heads != 0:
            raise ValueError(
                'x.shape[2] must be divisible by n_heads for att_group aggregation'
            )


class TemporallySharedBlock(nn.Module):
    """Base for conv blocks shared across the temporal dimension.

    Adds :meth:`smart_forward`, which flattens the ``(B, T)`` dims of a 5-D
    input before applying the block and reshapes the output back.
    """

    def __init__(self, pad_value: float | None = None) -> None:
        """Initialize TemporallySharedBlock.

        Args:
            pad_value: If given, padded frames (all channels equal to this
                value) are skipped and their outputs are filled with this value.
        """
        super().__init__()
        self.pad_value = pad_value

    def smart_forward(self, x: Tensor) -> Tensor:
        """Apply this block to each timestep of a 5-D tensor.

        Args:
            x: Input of shape ``(B, T, C, H, W)``.

        Returns:
            Output matching the input rank.

        Raises:
            ValueError: If *x* does not have shape ``(B, T, C, H, W)``.
            ValueError: If every frame is padding.
        """
        if x.ndim != 5:
            raise ValueError('expected (B, T, C, H, W)')

        b, t = x.shape[:2]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        if self.pad_value is None:
            out = self(x)
        else:
            valid = ~reduce(x == self.pad_value, 'frames c h w -> frames', 'all')
            if not valid.any():
                raise ValueError('batch contains no valid frames')

            valid_out = self(x[valid])
            out = valid_out.new_full((b * t, *valid_out.shape[1:]), self.pad_value)
            out[valid] = valid_out

        return rearrange(out, '(b t) c h w -> b t c h w', b=b, t=t)


class ConvLayer(nn.Module):
    """Stack of Conv2d + norm + ReLU layers."""

    def __init__(
        self,
        nkernels: Sequence[int],
        norm: str = 'batch',
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'reflect',
    ) -> None:
        """Initialize ConvLayer.

        Args:
            nkernels: Channel widths ``[in, hidden..., out]``.
            norm: Normalisation type: ``'batch'``, ``'instance'``,
                ``'group'``, or ``'none'``.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            n_groups: Number of groups for GroupNorm.
            last_relu: If ``False``, omit ReLU after the last conv.
            padding_mode: Padding mode for :class:`~torch.nn.Conv2d`.
        """
        super().__init__()
        nl: Callable[[int], nn.Module] | None
        if norm == 'batch':
            nl = nn.BatchNorm2d
        elif norm == 'instance':
            nl = nn.InstanceNorm2d
        elif norm == 'group':
            nl = partial(nn.GroupNorm, n_groups)
        else:
            nl = None

        layers: list[nn.Module] = []
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            if last_relu or i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional stack.

        Args:
            x: Input tensor of shape ``(B, nkernels[0], H, W)``.

        Returns:
            Output tensor of shape ``(B, nkernels[-1], H_out, W_out)``.
        """
        return self.conv(x)


class ConvBlock(TemporallySharedBlock):
    """Temporally shared :class:`ConvLayer`."""

    def __init__(
        self,
        nkernels: Sequence[int],
        pad_value: float | None = None,
        norm: str = 'batch',
        last_relu: bool = True,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'reflect',
    ) -> None:
        """Initialize ConvBlock.

        Args:
            nkernels: Channel widths ``[in, hidden..., out]``.
            pad_value: See :class:`TemporallySharedBlock`.
            norm: Normalisation type.
            last_relu: If ``False``, omit ReLU after the final conv.
            padding_mode: Padding mode for convolutions.
        """
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the temporally shared convolutional block.

        Args:
            x: Input tensor of shape ``(B, nkernels[0], H, W)``.

        Returns:
            Output tensor of shape ``(B, nkernels[-1], H, W)``.
        """
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    """Strided downsampling followed by two conv layers with a residual."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        pad_value: float | None = None,
        norm: str = 'batch',
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'reflect',
    ) -> None:
        """Initialize DownConvBlock.

        Args:
            d_in: Input channels.
            d_out: Output channels.
            k: Stride conv kernel size.
            s: Stride.
            p: Padding.
            pad_value: See :class:`TemporallySharedBlock`.
            norm: Normalisation type.
            padding_mode: Padding mode for convolutions.
        """
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, x: Tensor) -> Tensor:
        """Downsample and transform a feature map.

        Args:
            x: Input tensor of shape ``(B, d_in, H, W)``.

        Returns:
            Downsampled tensor of shape ``(B, d_out, H_out, W_out)``.
        """
        out = self.down(x)
        out = self.conv1(out)
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    """Transposed-conv upsampling, skip fusion, and two conv layers."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        norm: str = 'batch',
        d_skip: int | None = None,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'reflect',
    ) -> None:
        """Initialize UpConvBlock.

        Args:
            d_in: Input channels from the lower resolution path.
            d_out: Output channels.
            k: Transposed conv kernel size.
            s: Transposed conv stride.
            p: Transposed conv padding.
            norm: Normalisation type.
            d_skip: Channels in the skip connection (defaults to ``d_out``).
            padding_mode: Padding mode for convolutions.
        """
        super().__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=1), nn.BatchNorm2d(d), nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Lower-resolution input of shape ``(B, d_in, H, W)``.
            skip: Skip connection of shape ``(B, d_skip, H', W')``.

        Returns:
            Output of shape ``(B, d_out, H', W')``.
        """
        out = self.up(x)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)
