# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest
import torch
from lightning.pytorch import Trainer

from torchgeo.datamodules import BioMasstersDataModule


class TestBioMasstersDataModule:
    @pytest.fixture
    def fit_datamodule(self) -> BioMasstersDataModule:
        dm = BioMasstersDataModule(
            root=os.path.join('tests', 'data', 'biomassters'),
            batch_size=1,
            num_workers=0,
            val_split_pct=0.5,
            test_split_pct=0.0,
            padding_length=4,
            sensors=('S1', 'S2'),
        )
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    @pytest.fixture
    def test_datamodule(self) -> BioMasstersDataModule:
        dm = BioMasstersDataModule(
            root=os.path.join('tests', 'data', 'biomassters'),
            batch_size=1,
            num_workers=0,
            val_split_pct=0.0,
            test_split_pct=0.5,
            padding_length=4,
            sensors=('S1', 'S2'),
        )
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    @pytest.fixture
    def s2_datamodule(self) -> BioMasstersDataModule:
        dm = BioMasstersDataModule(
            root=os.path.join('tests', 'data', 'biomassters'),
            batch_size=1,
            num_workers=0,
            val_split_pct=0.5,
            test_split_pct=0.0,
            padding_length=3,
            sensors=('S2',),
        )
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    def test_train_dataloader(self, fit_datamodule: BioMasstersDataModule) -> None:
        fit_datamodule.setup('fit')
        batch = next(iter(fit_datamodule.train_dataloader()))

        assert batch['image'].shape == (1, 4, 15, 32, 32)
        assert batch['mask'].shape == (1, 1, 32, 32)
        assert torch.equal(batch['length'], torch.tensor([3]))
        assert not torch.all(batch['image'][0, 1, :4] == 0)
        assert torch.all(batch['image'][0, 1, 4:] == 0)

    def test_s2_only_padding(self, s2_datamodule: BioMasstersDataModule) -> None:
        s2_datamodule.setup('fit')
        batch = next(iter(s2_datamodule.train_dataloader()))

        assert batch['image'].shape == (1, 3, 11, 32, 32)
        assert torch.equal(batch['length'], torch.tensor([2]))
        assert torch.all(batch['image'][0, 2] == 0)

    def test_test_dataloader(self, test_datamodule: BioMasstersDataModule) -> None:
        test_datamodule.setup('test')
        batch = next(iter(test_datamodule.test_dataloader()))

        assert batch['image'].shape == (1, 4, 15, 32, 32)
        assert batch['mask'].shape == (1, 1, 32, 32)
        assert torch.equal(batch['length'], torch.tensor([3]))

    def test_predict_dataloader(self, test_datamodule: BioMasstersDataModule) -> None:
        test_datamodule.setup('predict')
        batch = next(iter(test_datamodule.predict_dataloader()))

        assert batch['image'].shape == (1, 4, 15, 32, 32)
        assert torch.equal(batch['length'], torch.tensor([3]))
        assert 'mask' not in batch

    def test_plot(self, fit_datamodule: BioMasstersDataModule) -> None:
        fit_datamodule.setup('fit')
        assert fit_datamodule.val_dataset is not None
        sample = fit_datamodule.val_dataset[0]
        fit_datamodule.plot(sample)
        plt.close()

    def test_plot_with_prediction(
        self, fit_datamodule: BioMasstersDataModule
    ) -> None:
        fit_datamodule.setup('fit')
        assert fit_datamodule.val_dataset is not None
        sample = fit_datamodule.val_dataset[0]
        sample['prediction'] = sample['mask']
        fit_datamodule.plot(sample)
        plt.close()

    def test_plot_invalid_image_shape(
        self, fit_datamodule: BioMasstersDataModule
    ) -> None:
        fit_datamodule.setup('fit')
        assert fit_datamodule.val_dataset is not None
        with pytest.raises(ValueError, match=r'Expected image tensor with shape'):
            fit_datamodule.val_dataset.dataset.plot({'image': torch.zeros(15, 32, 32)})
