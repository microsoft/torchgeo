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
    """Convert a Sentinel-1 SLC ComplexFloatTensor/Complex64 to In-phase/Quadrature components.

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
