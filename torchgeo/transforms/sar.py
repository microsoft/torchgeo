# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

_EPSILON = 1e-10


def inphase_quadrature_to_phase_magnitude(src: Tensor) -> Tensor:
    magnitude = _in_phase_to_magnitude(src)
    phase = _quadrature_phase_to_phase(src)

    magnitude = torch.unsqueeze(magnitude, dim=0)
    phase = torch.unsqueeze(phase, dim=0)

    return torch.concat((phase, magnitude), dim=0)


def _in_phase_to_magnitude(iq: Tensor) -> Tensor:
    """Convert the In-phase channel to magnitude

    Args:
        iq: A tensor in In-phase/Quadrature representation

    Returns:
        A tensor with the magnitude representation.
    """
    return torch.log10((iq[0] + _EPSILON) ** 2 + (iq[1] + _EPSILON) ** 2)


def _quadrature_phase_to_phase(iq: Tensor) -> Tensor:
    """Convert the Quadrature-phase to phase

    Args:
        iq: A tensor in In-phase/Quadrature representation.

    Returns:
        A tensor with the phase representation.
    """
    return torch.atan2(iq[1], iq[0])


def slc_to_iq(src: Tensor) -> Tensor:
    """Convert an SLC ComplexFloatTensor/Complex64 to In-phase/Quadrature components

    Args:
        src: A (1, W, H) tensor from a Sentinel-1 style SLC.

    Returns:
        A (2, W, H) tensor in In-Phase/Quadrature representation.
    """
    in_phase = src.real
    quadrature = src.imag

    return torch.concat((in_phase, quadrature), dim=0)
