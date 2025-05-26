# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import SolarPlantsBrazilDataModule


class TestSolarPlantsBrazilDataModule:
    @pytest.fixture
    def datamodule(self) -> SolarPlantsBrazilDataModule:
        root = os.path.join('tests', 'data', 'solar_plants_brazil')
        batch_size = 2
        num_workers = 0
        dm = SolarPlantsBrazilDataModule(
            root=root, batch_size=batch_size, num_workers=num_workers
        )
        return dm

    def test_train_dataloader(self, datamodule: SolarPlantsBrazilDataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: SolarPlantsBrazilDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: SolarPlantsBrazilDataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))

    def test_batch_shape(self, datamodule: SolarPlantsBrazilDataModule) -> None:
        datamodule.setup('fit')
        batch = next(iter(datamodule.train_dataloader()))

        # Apply post-processing manually (so the shape matches Lightning behavior)
        batch = datamodule.on_after_batch_transfer(batch, dataloader_idx=0)

        assert 'image' in batch
        assert 'mask' in batch
        assert batch['image'].ndim == 4  # [B, C, H, W]
        assert batch['mask'].ndim == 3  # [B, H, W]

    def test_plot(self, datamodule: SolarPlantsBrazilDataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        sample = {'image': batch['image'][0], 'mask': batch['mask'][0]}
        datamodule.plot(sample)
        plt.close()
