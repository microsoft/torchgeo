# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import cast

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.trainers import SpatioTemporalSegmentationTask


class BinarySegDataset(NonGeoDataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> Sample:
        return {
            'image': torch.randn(4, 3, 16, 16),
            'mask': torch.randint(0, 2, (16, 16)),
            'length': torch.tensor(4),
        }


class TestSpatioTemporalSegmentationTask:
    @pytest.mark.parametrize('name', ['pastis', 'pastis_focal', 'pastis_jaccard'])
    def test_trainer(self, name: str, fast_dev_run: bool) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    def test_binary_task(self, fast_dev_run: bool) -> None:
        dataloader = DataLoader(BinarySegDataset(), batch_size=2)
        model = SpatioTemporalSegmentationTask(
            in_channels=3, task='binary', loss='bce', hidden_dim=8, num_layers=1
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
        trainer.test(model, dataloaders=dataloader)
        predictions = trainer.predict(model, dataloaders=dataloader)
        assert predictions is not None
        prediction = cast(torch.Tensor, predictions[0])
        assert prediction.shape == (2, 1, 16, 16)
        assert torch.all(prediction >= 0)
        assert torch.all(prediction <= 1)

    def test_multilabel_predict_step(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, num_labels=4, task='multilabel', hidden_dim=8, num_layers=1
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
