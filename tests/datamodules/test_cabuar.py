# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import CaBuArDataModule
from torchgeo.datasets import unbind_samples

pytest.importorskip('h5py', minversion='3.6')


class TestCaBuArDataModule:
    @pytest.fixture
    def datamodule(self) -> CaBuArDataModule:
        root = os.path.join('tests', 'data', 'cabuar')
        batch_size = 1
        num_workers = 0
        dm = CaBuArDataModule(root=root, batch_size=batch_size, num_workers=num_workers)
        dm.prepare_data()
        return dm

    def test_train_dataloader(self, datamodule: CaBuArDataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: CaBuArDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: CaBuArDataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: CaBuArDataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample)
        plt.close()
