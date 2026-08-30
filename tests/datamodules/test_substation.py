# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.datamodules import SubstationDataModule


def test_substation_normalization_scale() -> None:
    datamodule = SubstationDataModule()

    assert datamodule.mean == torch.tensor(0)
    assert datamodule.std == torch.tensor(10000)
