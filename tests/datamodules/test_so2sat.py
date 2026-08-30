# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from torchgeo.datamodules import So2SatDataModule


def test_so2sat_all_uses_all_band_statistics() -> None:
    datamodule = So2SatDataModule()
    assert datamodule.mean.shape == (18,)
    assert datamodule.std.shape == (18,)
