# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.datamodules import PASTIS100DataModule, PASTISDataModule


def test_pastis_normalization_scale() -> None:
    for datamodule in (PASTISDataModule(), PASTIS100DataModule()):
        assert datamodule.mean == torch.tensor(0)
        assert datamodule.std == torch.tensor(10000)
