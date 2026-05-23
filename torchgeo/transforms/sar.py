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
