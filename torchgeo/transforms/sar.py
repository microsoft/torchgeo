# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""SAR-specific transforms for synthetic aperture radar imagery."""

import torch
import torch.nn.functional as F
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor


def _box_filter(x: Tensor, window_size: int) -> Tensor:
    """Apply a per-channel box (mean) filter over the spatial dimensions.

    Args:
        x: Input tensor of shape ``(B, C, H, W)``.
        window_size: Odd integer side length of the smoothing window.

    Returns:
        Smoothed tensor of identical shape.
    """
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    kernel = torch.ones(
        1, 1, window_size, window_size, device=x.device, dtype=x.dtype
    ) / float(window_size * window_size)
    channels = x.shape[1]
    kernel = kernel.expand(channels, 1, window_size, window_size)
    return F.conv2d(x_padded, kernel, groups=channels)


def lee_filter(
    image: Tensor, window_size: int = 7, num_looks: float = 1.0, eps: float = 1e-8
) -> Tensor:
    r"""Apply the Lee filter to a SAR intensity image.

    The Lee (1980) filter assumes a multiplicative speckle model
    :math:`x = s \cdot v` where :math:`s` is the underlying signal and
    :math:`v` is unit-mean speckle with variance
    :math:`\sigma_v^2 = 1 / L` for an :math:`L`-look intensity image. The
    local linear minimum mean square error (LMMSE) estimator is

    .. math::

        \hat{s} = \mu + k \cdot (x - \mu),
        \quad
        k = \frac{\sigma_s^2}{\sigma_s^2 + \sigma_v^2 \mu^2}

    where :math:`\mu` is the local mean and
    :math:`\sigma_s^2 = \max(\sigma_x^2 - \sigma_v^2 \mu^2, 0)` is the
    estimated signal variance under the multiplicative-noise model. In
    homogeneous regions the filter behaves like a mean filter; near edges
    it preserves detail by giving the local mean less weight.

    If you use this method in your research, please cite the following paper:

    * https://doi.org/10.1109/TPAMI.1980.4766994

    Args:
        image: SAR intensity tensor of shape ``(B, C, H, W)``. Values are
            assumed to be non-negative intensities, not amplitudes or dB.
        window_size: Odd integer size of the local statistics window. Larger
            values produce more smoothing at the cost of detail.
        num_looks: Equivalent number of looks (ENL) of the input image.
            Single-look complex (SLC) intensity has ``num_looks=1``.
            Sentinel-1 GRDH typically has ``num_looks`` near 5.
        eps: Numerical floor to avoid division by zero in flat regions.

    Returns:
        Filtered tensor of the same shape and dtype as ``image``.

    Raises:
        ValueError: If ``window_size`` is not a positive odd integer.
        ValueError: If ``num_looks`` is not strictly positive.

    .. versionadded:: 0.10
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            f'window_size must be a positive odd integer, got {window_size}'
        )
    if num_looks <= 0:
        raise ValueError(f'num_looks must be > 0, got {num_looks}')

    sigma_v_sq = 1.0 / float(num_looks)

    mean_local = _box_filter(image, window_size)
    mean_sq_local = _box_filter(image * image, window_size)
    var_local = (mean_sq_local - mean_local * mean_local).clamp(min=0.0)

    var_signal = (var_local - sigma_v_sq * mean_local * mean_local).clamp(min=0.0)
    weight = var_signal / (var_signal + sigma_v_sq * mean_local * mean_local + eps)

    return mean_local + weight * (image - mean_local)


class LeeFilter(IntensityAugmentationBase2D):
    """Lee speckle reduction filter for SAR imagery.

    Applies the classic Lee (1980) adaptive filter to reduce multiplicative
    speckle noise while preserving edges and structural detail. Operates on
    SAR intensity imagery with non-negative values; amplitude and dB inputs
    should be converted to intensity beforehand.

    If you use this method in your research, please cite the following paper:

    * https://doi.org/10.1109/TPAMI.1980.4766994

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        window_size: int = 7,
        num_looks: float = 1.0,
        p: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize a new LeeFilter instance.

        Args:
            window_size: Odd integer size of the local statistics window.
            num_looks: Equivalent number of looks (ENL) of the input SAR data.
                Single-look complex (SLC) intensity has ``num_looks=1``.
            p: Probability of applying the filter to each sample.
            same_on_batch: Apply the same transformation across the batch.
            keepdim: Whether to keep the output shape the same as input (True)
                or broadcast it to the batch form (False).

        Raises:
            ValueError: If ``window_size`` is not a positive odd integer.
            ValueError: If ``num_looks`` is not strictly positive.
        """
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f'window_size must be a positive odd integer, got {window_size}'
            )
        if num_looks <= 0:
            raise ValueError(f'num_looks must be > 0, got {num_looks}')
        self.flags = {'window_size': window_size, 'num_looks': num_looks}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int | float],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the Lee filter to the input SAR image.

        Args:
            input: The input tensor.
            params: Generated parameters.
            flags: Static parameters.
            transform: The geometric transformation tensor.

        Returns:
            The filtered tensor.
        """
        return lee_filter(
            input,
            window_size=int(flags['window_size']),
            num_looks=float(flags['num_looks']),
        )


# Sub-window indices in the 9-channel directional-mean output (used by
# refined_lee_filter). Layout in the 7x7 frame:
#     NW  N  NE
#     W   C  E
#     SW  S  SE
_NW, _N, _NE = 0, 1, 2
_W, _C, _E = 3, 4, 5
_SW, _S, _SE = 6, 7, 8


def _build_subwindow_kernels(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build the 9 directional 3x3 sub-window kernels in a 7x7 frame.

    Returns a ``(9, 1, 7, 7)`` kernel where each channel has ``1/9`` in its
    3x3 sub-window position and 0 elsewhere. Sub-window order (channel
    index): ``[NW, N, NE, W, C, E, SW, S, SE]``.
    """
    kernels = torch.zeros(9, 1, 7, 7, device=device, dtype=dtype)
    centers = [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5)]
    for k, (r, c) in enumerate(centers):
        kernels[k, 0, r - 1 : r + 2, c - 1 : c + 2] = 1.0 / 9.0
    return kernels


def _select_subwindow(means: Tensor, image: Tensor) -> Tensor:
    """Pick the 3x3 sub-window aligned with the local edge direction.

    For each pixel, finds the dominant edge direction (out of 4: horizontal,
    vertical, two diagonals) by ``argmax`` over the absolute directional
    gradients of the 9 sub-window means, then selects the parallel
    sub-window whose mean is closest to the center pixel value (i.e. the
    homogeneous region the center pixel belongs to).

    Args:
        means: ``(B, C, 9, H, W)`` tensor of the 9 sub-window means.
        image: ``(B, C, H, W)`` tensor of the original pixel values.

    Returns:
        Index tensor of shape ``(B, C, H, W)`` giving the chosen sub-window
        index in the range ``[0, 9)`` for each pixel.
    """
    grad_horiz = means[:, :, _E] - means[:, :, _W]
    grad_vert = means[:, :, _S] - means[:, :, _N]
    grad_diag1 = means[:, :, _NE] - means[:, :, _SW]
    grad_diag2 = means[:, :, _NW] - means[:, :, _SE]
    grads = torch.stack([grad_horiz, grad_vert, grad_diag1, grad_diag2], dim=2)

    edge_dir = grads.abs().argmax(dim=2, keepdim=True).squeeze(2)

    pair_table = torch.tensor(
        [[_W, _E], [_N, _S], [_SW, _NE], [_SE, _NW]],
        device=means.device,
        dtype=torch.long,
    )

    pair_per_pixel = pair_table[edge_dir]
    side0_idx = pair_per_pixel[..., 0]
    side1_idx = pair_per_pixel[..., 1]

    side0_mean = means.gather(2, side0_idx.unsqueeze(2)).squeeze(2)
    side1_mean = means.gather(2, side1_idx.unsqueeze(2)).squeeze(2)

    pick_side1 = (image - side1_mean).abs() < (image - side0_mean).abs()
    return torch.where(pick_side1, side1_idx, side0_idx)


def refined_lee_filter(
    image: Tensor,
    num_looks: float = 1.0,
    edge_threshold_sigma: float = 2.0,
    eps: float = 1e-8,
) -> Tensor:
    r"""Apply the Refined Lee filter to a SAR intensity image.

    The Refined Lee (Lee, 1981) filter improves on the classic
    :func:`lee_filter` by estimating the local edge direction from 9
    overlapping 3x3 sub-windows within a 7x7 frame, then applying the
    LMMSE estimator using only the sub-window aligned along that edge —
    preserving edges far better than the basic Lee filter at moderate
    additional compute cost.

    To avoid spurious "edges" from speckle in homogeneous regions, the
    implementation follows the SNAP / Google Earth Engine convention of
    gating the directional refinement on edge magnitude: when the maximum
    directional gradient is below
    :math:`\text{edge\_threshold\_sigma} \cdot \sigma_v \cdot \mu`, the
    filter falls back to the standard 7x7 box statistics used by
    :func:`lee_filter`.

    If you use this method in your research, please cite the following paper:

    * https://doi.org/10.1016/S0146-664X(81)80018-4

    Args:
        image: SAR intensity tensor of shape ``(B, C, H, W)``. Values are
            assumed to be non-negative intensities, not amplitudes or dB.
        num_looks: Equivalent number of looks (ENL) of the input image.
            Single-look complex (SLC) intensity has ``num_looks=1``.
            Sentinel-1 GRDH typically has ``num_looks`` near 5.
        edge_threshold_sigma: Multiplier on the per-pixel speckle standard
            deviation used to decide whether a directional gradient is a
            real edge. Default ``2.0`` (~2 sigma rule).
        eps: Numerical floor to avoid division by zero in flat regions.

    Returns:
        Filtered tensor of the same shape and dtype as ``image``.

    Raises:
        ValueError: If ``num_looks`` is not strictly positive.

    .. versionadded:: 0.10
    """
    if num_looks <= 0:
        raise ValueError(f'num_looks must be > 0, got {num_looks}')

    batch_size, channels, height, width = image.shape
    sigma_v_sq = 1.0 / float(num_looks)
    sigma_v = sigma_v_sq**0.5

    # Sub-window means (and squared means) via grouped 7x7 conv
    sub_kernels = _build_subwindow_kernels(image.device, image.dtype)
    sub_kernels_per_channel = sub_kernels.repeat(channels, 1, 1, 1)

    image_padded = F.pad(image, (3, 3, 3, 3), mode='reflect')
    means_flat = F.conv2d(image_padded, sub_kernels_per_channel, groups=channels)
    means = means_flat.view(batch_size, channels, 9, height, width)

    image_sq_padded = F.pad(image * image, (3, 3, 3, 3), mode='reflect')
    means_sq_flat = F.conv2d(image_sq_padded, sub_kernels_per_channel, groups=channels)
    means_sq = means_sq_flat.view(batch_size, channels, 9, height, width)

    # Box (basic-Lee) statistics for the homogeneous-region fallback branch
    box_mean = _box_filter(image, 7)
    box_mean_sq = _box_filter(image * image, 7)
    box_var = (box_mean_sq - box_mean * box_mean).clamp(min=0.0)

    # Directional gradients + edge detection
    grad_horiz = means[:, :, _E] - means[:, :, _W]
    grad_vert = means[:, :, _S] - means[:, :, _N]
    grad_diag1 = means[:, :, _NE] - means[:, :, _SW]
    grad_diag2 = means[:, :, _NW] - means[:, :, _SE]
    grads = torch.stack([grad_horiz, grad_vert, grad_diag1, grad_diag2], dim=2)
    max_abs_grad = grads.abs().max(dim=2)[0]

    edge_threshold = edge_threshold_sigma * sigma_v * box_mean
    is_edge = max_abs_grad > edge_threshold

    # Refined-Lee branch: stats from the directionally-selected sub-window
    selected_subwindow = _select_subwindow(means, image)
    sub_mean = means.gather(2, selected_subwindow.unsqueeze(2)).squeeze(2)
    sub_mean_sq = means_sq.gather(2, selected_subwindow.unsqueeze(2)).squeeze(2)
    sub_var = (sub_mean_sq - sub_mean * sub_mean).clamp(min=0.0)

    # Hybrid: refined-Lee where edge, basic-Lee box stats elsewhere
    local_mean = torch.where(is_edge, sub_mean, box_mean)
    var_local = torch.where(is_edge, sub_var, box_var)

    # LMMSE estimator
    var_signal = (var_local - sigma_v_sq * local_mean * local_mean).clamp(min=0.0)
    weight = var_signal / (var_signal + sigma_v_sq * local_mean * local_mean + eps)

    return local_mean + weight * (image - local_mean)


class RefinedLeeFilter(IntensityAugmentationBase2D):
    """Refined Lee speckle reduction filter for SAR imagery.

    Edge-preserving variant of :class:`LeeFilter`. Estimates the local edge
    direction from 9 directional 3x3 sub-windows within a 7x7 frame and
    applies the LMMSE estimator using the sub-window aligned along that
    edge. Falls back to the standard 7x7 box statistics in homogeneous
    regions to avoid amplifying speckle as false edges.

    If you use this method in your research, please cite the following paper:

    * https://doi.org/10.1016/S0146-664X(81)80018-4

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        num_looks: float = 1.0,
        edge_threshold_sigma: float = 2.0,
        p: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize a new RefinedLeeFilter instance.

        Args:
            num_looks: Equivalent number of looks (ENL) of the input SAR data.
                Single-look complex (SLC) intensity has ``num_looks=1``.
            edge_threshold_sigma: Multiplier on the per-pixel speckle standard
                deviation used to decide whether a directional gradient is a
                real edge. Default ``2.0`` (~2 sigma rule).
            p: Probability of applying the filter to each sample.
            same_on_batch: Apply the same transformation across the batch.
            keepdim: Whether to keep the output shape the same as input (True)
                or broadcast it to the batch form (False).

        Raises:
            ValueError: If ``num_looks`` is not strictly positive.
        """
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if num_looks <= 0:
            raise ValueError(f'num_looks must be > 0, got {num_looks}')
        self.flags = {
            'num_looks': num_looks,
            'edge_threshold_sigma': edge_threshold_sigma,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int | float],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the Refined Lee filter to the input SAR image.

        Args:
            input: The input tensor.
            params: Generated parameters.
            flags: Static parameters.
            transform: The geometric transformation tensor.

        Returns:
            The filtered tensor.
        """
        return refined_lee_filter(
            input,
            num_looks=float(flags['num_looks']),
            edge_threshold_sigma=float(flags['edge_threshold_sigma']),
        )
