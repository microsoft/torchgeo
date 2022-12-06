# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import Vaihingen2DDataModule
from torchgeo.datasets import unbind_samples


class TestVaihingen2DDataModule:
    @pytest.fixture(scope="class", params=[0.0, 0.5])
    def datamodule(self, request: SubRequest) -> Vaihingen2DDataModule:
        root = os.path.join("tests", "data", "vaihingen")
        batch_size = 1
        num_workers = 0
        val_split_size = request.param
        dm = Vaihingen2DDataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=val_split_size,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: Vaihingen2DDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: Vaihingen2DDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: Vaihingen2DDataModule) -> None:
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: Vaihingen2DDataModule) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample)
        plt.close()
