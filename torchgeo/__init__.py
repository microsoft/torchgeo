# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo: datasets, transforms, and models for geospatial data.

This library is part of the `PyTorch <http://pytorch.org/>`_ project. PyTorch is an open
source machine learning framework.

The :mod:`torchgeo` package consists of popular datasets, model architectures, and
common image transformations for geospatial data.
"""

# Fix circular import issue, see:
# https://github.com/microsoft/torchgeo/issues/276
import torchgeo.datasets  # noqa: F401

__author__ = "Adam J. Stewart"
__version__ = "0.1.1"
