# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SAR utilities."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def inphase_quadrature_to_phase_magnitude(src: Tensor) -> Tensor:
    """Convert a SAR image in in-phase/quadrature format to phase/magnitude.

    Magnitude is in power scale.

    .. versionadded:: 0.5

    Args:
        src: A tensor containing in-phase and quadrature channels.

    Returns:
        A tensor in phase/magnitude representation
    """
    magnitude = _in_phase_to_magnitude(src)
    phase = _quadrature_phase_to_phase(src)

    magnitude = torch.unsqueeze(magnitude, dim=0)
    phase = torch.unsqueeze(phase, dim=0)

    return torch.concat((phase, magnitude), dim=0)


def _in_phase_to_magnitude(iq: Tensor) -> Tensor:
    """Detect an I/Q image to the power scale.

    .. versionadded:: 0.5

    Args:
        iq: A tensor in In-phase/Quadrature representation

    Returns:
        mag: A tensor with the magnitude of the SAR image in the power scale.
    """
    mag = iq[0] ** 2 + iq[1] ** 2
    return mag


def _quadrature_phase_to_phase(iq: Tensor) -> Tensor:
    """Convert the Quadrature-phase to phase.

    .. versionadded:: 0.5

    Args:
        iq: A tensor in In-phase/Quadrature representation.

    Returns:
        A tensor with the phase representation.
    """
    return torch.atan2(iq[1], iq[0])


def slc_to_iq(src: Tensor) -> Tensor:
    """Convert a Sentinel-1 SLC to In-phase/Quadrature components.

    .. versionadded:: 0.5

    Args:
        src: A (1, W, H) tensor from a Sentinel-1 style SLC.

    Returns:
        A (2, W, H) tensor in In-Phase/Quadrature representation.
    """
    in_phase = src.real
    quadrature = src.imag

    return torch.concat((in_phase, quadrature), dim=0)


def power_to_decibel(src: Tensor) -> Tensor:
    """Switch from power to decibel scales.

    .. versionadded:: 0.5

    Args:
        src: Tensor in power scale

    Returns:
        Tensor in dB scale
    """
    db = torch.log10(src)
    db[db.isinf()] = 0  # Clamp -inf values to 0
    return 10 * db


def power_to_amplitude(src: Tensor) -> Tensor:
    """Switch from power to amplitude scales.

    .. versionadded:: 0.5

    Args:
        src: Tensor in power scale

    Returns:
        Tensor in amplitude scale
    """
    return torch.sqrt(src)


def amplitude_to_power(src: Tensor) -> Tensor:
    """Switch from amplitude to power scales.

    .. versionadded:: 0.5

    Args:
        src: Tensor in amplitude scale

    Returns:
        Tensor in power scale
    """
    return src**2


def amplitude_to_decibel(src: Tensor) -> Tensor:
    """Switch from amplitude to decibel scales.

    .. versionadded:: 0.5

    Args:
        src: Tensor in amplitude scale

    Returns:
        Tensor in dB scale
    """
    return power_to_decibel(amplitude_to_power(src))


def enl(src: Tensor, window_size: int = 3) -> Tensor:
    """Calculate the estimated number of looks for a SAR image in power scale.

    .. versionadded:: 0.5

    Args:
        src: A tensor in power scale.
        window_size: The window size with which to calculate the mean and variance

    Returns:
        enl: The estimated number of looks for the image.
    """
    pad_size = math.ceil((window_size / 2) - 1)
    src_pad = F.pad(
        src.unsqueeze(dim=0), (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )

    mean_image = F.avg_pool2d(src_pad, kernel_size=3, stride=1)
    mean_sqr_image = F.avg_pool2d(src_pad**2, kernel_size=3, stride=1)
    var_image = mean_sqr_image - mean_image**2

    enl_image = mean_image / torch.sqrt(var_image)
    histogram = torch.histogram(enl_image)
    return histogram.bin_edges[torch.argmax(histogram.hist)]


def lee_filter(src: Tensor, window_size: int):
    """Lee filter for radar speckle removal.

    .. versionadded:: 0.5

    Args:
        src: input Tensor
        window_size: kernel window size, must be odd

    Returns:
        filtered_output: the filtered SAR image in amplitude scale.
    """
    assert window_size % 2 == 1, "The window size must be odd"

    pad_size = math.ceil((window_size / 2) - 1)
    src_pad = F.pad(src, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

    mean_image = F.avg_pool2d(
        src_pad, kernel_size=window_size, stride=1, count_include_pad=False
    )
    mean_sqr_image = F.avg_pool2d(
        src_pad**2, kernel_size=window_size, stride=1, count_include_pad=False
    )
    var_image = mean_sqr_image - mean_image**2

    variance = torch.var(src)

    img_weights = var_image / (var_image + variance)
    filtered_output = mean_image + img_weights * (src - mean_image)
    return filtered_output


def enhanced_lee_filter(src: Tensor, window_size: int, num_looks: int = 1):
    """Enhanced Lee filter for radar speckle removal.

    Implements the Enhanced Lee filter as defined in "Adaptive Speckle Filters
    and Scene Heterogeneity" by Lopes, Touzi, and Nezry (1990).

    .. versionadded:: 0.5

    Args:
        src: input Tensor
        window_size: kernel window size, must be odd
        num_looks: number of looks for the provided SAR image.

    Returns:
        filtered_output: the filtered SAR image in amplitude scale.
    """
    assert window_size % 2 == 1, "The window size must be odd"

    Cu = 0.523 / math.sqrt(num_looks)  # 0.523 is a magic number from Lopes et. al.

    pad_size = math.ceil((window_size / 2) - 1)
    src_pad = F.pad(src, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

    mean_image = F.avg_pool2d(
        src_pad, kernel_size=window_size, stride=1, count_include_pad=False
    )
    mean_sqr_image = F.avg_pool2d(
        src_pad**2, kernel_size=window_size, stride=1, count_include_pad=False
    )
    var_image = mean_sqr_image - mean_image**2

    # Lopes et. al. define C_max as the maximum coefficient of variation
    # value over the moving window within a homogeneous area
    Cmax = (torch.sqrt(var_image) / mean_image).max()

    noise_coeff_image = (
        torch.sqrt(var_image) / mean_image
    )  # Using the amplitude image as input
    thresh = noise_coeff_image.clone()

    mask_le = thresh <= Cu
    mask_ge = thresh >= Cmax
    mask_intermediate = (Cu < thresh) & (thresh < Cmax)

    # Set all values less than or equal to noise_var_coeff to 1.0
    thresh[mask_le] = 1.0

    # Set all values greater than or equal to var_max to 0.0
    thresh[mask_ge] = 0.0

    # Compute the threshold for values in between noise_var_coeff and var_max
    intermediate = -num_looks * (thresh - Cu) / (Cmax - thresh)
    thresh = (thresh * ~mask_intermediate) + (
        torch.exp(intermediate) * mask_intermediate
    )

    filtered_output = (mean_image * thresh) + (src * (1.0 - thresh))

    return filtered_output
