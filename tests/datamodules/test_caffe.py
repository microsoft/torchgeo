# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import CaFFeDataModule


class TestCaFFeDataModule:
    @pytest.fixture
    def datamodule(self, test_data: Callable[[str], str]) -> CaFFeDataModule:
        root = test_data('caffe')
        batch_size = 2
        num_workers = 0
        dm = CaFFeDataModule(root=root, batch_size=batch_size, num_workers=num_workers)
        return dm

    def test_train_dataloader(self, datamodule: CaFFeDataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: CaFFeDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: CaFFeDataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: CaFFeDataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        sample = {
            'image': batch['image'][0],
            'mask_zones': batch['mask_zones'][0],
            'mask_front': batch['mask_front'][0],
        }
        datamodule.plot(sample)
        plt.close()
