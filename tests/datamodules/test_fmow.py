# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import FMoWDataModule


class TestFMoWDataModule:
    @pytest.fixture
    def datamodule(self) -> FMoWDataModule:
        root = os.path.join('tests', 'data', 'fmow')
        return FMoWDataModule(root=root, batch_size=1, num_workers=0)

    def test_train_dataloader(self, datamodule: FMoWDataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: FMoWDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_plot(self, datamodule: FMoWDataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        sample = {
            'image': batch['image'][0],
            'bbox_xyxy': batch['bbox_xyxy'][0],
            'label': batch['label'][0],
        }
        datamodule.plot(sample)
        plt.close()
