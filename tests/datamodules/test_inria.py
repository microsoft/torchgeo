# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import InriaAerialImageLabelingDataModule


class TestInriaAerialImageLabelingDataModule:
    @pytest.fixture
    def datamodule(self) -> InriaAerialImageLabelingDataModule:
        root = os.path.join('tests', 'data', 'inria')
        dm = InriaAerialImageLabelingDataModule(
            root=root, batch_size=2, patch_size=2, num_workers=0
        )
        return dm

    def test_train_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        datamodule.setup('fit')
        batch = next(iter(datamodule.train_dataloader()))
        assert batch['image'].ndim == 4  # (B, C, H, W)
        assert batch['mask'].ndim == 3  # (B, H, W)
        assert batch['image'].shape[0] == 2  # batch_size

    def test_val_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        datamodule.setup('fit')
        batch = next(iter(datamodule.val_dataloader()))
        assert batch['image'].ndim == 4
        assert batch['mask'].ndim == 3
        assert batch['image'].shape[0] == 2

    def test_predict_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        datamodule.setup('predict')
        batch = next(iter(datamodule.predict_dataloader()))
        assert batch['image'].ndim == 4

    def test_plot(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        datamodule.setup('fit')
        batch = next(iter(datamodule.val_dataloader()))
        sample = {
            'image': batch['image'][0],
            'mask': batch['mask'][0],
        }
        datamodule.plot(sample)
        plt.close()
