# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import SpaceNet1DataModule
from torchgeo.datasets import unbind_samples


class TestSpaceNet1DataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> SpaceNet1DataModule:
        root = os.path.join("tests", "data", "spacenet")
        batch_size = 2
        num_workers = 0
        dm = SpaceNet1DataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=0.33,
            test_split_pct=0.33,
        )
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: SpaceNet1DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: SpaceNet1DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: SpaceNet1DataModule) -> None:
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: SpaceNet1DataModule) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample)
        plt.close()
