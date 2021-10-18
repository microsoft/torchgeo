# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo specific losses."""

from .qr_losses import loss_on_prior_reversed_kl_simple, loss_on_prior_simple

__all__ = (
    "loss_on_prior_simple",
    "loss_on_prior_reversed_kl_simple",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.losses"
