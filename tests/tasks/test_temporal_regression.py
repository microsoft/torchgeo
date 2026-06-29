# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch
from lightning.pytorch import Trainer

from torchgeo.datamodules import AirQualityDataModule, MisconfigurationException
from torchgeo.datasets import AirQuality
from torchgeo.main import main
from torchgeo.tasks import TemporalRegression


class TestTemporalRegression:
    @pytest.mark.parametrize(
        'name', ['air_quality_mse', 'air_quality_mae', 'air_quality_presto']
    )
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

    def test_predict_ltae(self) -> None:
        root = os.path.join('tests', 'data', 'air_quality')
        model = TemporalRegression(in_channels=17, num_outputs=17, len_max_seq=3)
        datamodule = AirQualityDataModule(root=root)
        datamodule.predict_dataset = AirQuality(root)
        trainer = Trainer(accelerator='cpu')
        trainer.predict(model=model, datamodule=datamodule)

    def test_predict_presto(self) -> None:
        root = os.path.join('tests', 'data', 'air_quality')
        model = TemporalRegression(
            model='presto',
            in_channels=17,
            num_outputs=17,
            encoder_embedding_size=16,
            channel_embed_ratio=0.25,
            month_embed_ratio=0.25,
            encoder_depth=1,
            mlp_ratio=2,
            encoder_num_heads=2,
            decoder_embedding_size=16,
            decoder_depth=1,
            decoder_num_heads=2,
            max_sequence_length=3,
        )
        datamodule = AirQualityDataModule(root=root)
        datamodule.predict_dataset = AirQuality(root)
        trainer = Trainer(accelerator='cpu')
        trainer.predict(model=model, datamodule=datamodule)

    def test_presto_channel_validation(self) -> None:
        match = 'Presto expected 17 input channels, got 3.'
        with pytest.raises(ValueError, match=match):
            TemporalRegression(model='presto', in_channels=3)

    def test_presto_optional_batch_fields(self) -> None:
        model = TemporalRegression(
            model='presto',
            in_channels=17,
            num_outputs=17,
            encoder_embedding_size=16,
            channel_embed_ratio=0.25,
            month_embed_ratio=0.25,
            encoder_depth=1,
            mlp_ratio=2,
            encoder_num_heads=2,
            decoder_embedding_size=16,
            decoder_depth=1,
            decoder_num_heads=2,
            max_sequence_length=3,
        )
        batch = {
            'input': torch.randn(2, 3, 17),
            'dynamic_world': torch.zeros(2, 3, dtype=torch.long),
            'latlons': torch.zeros(2, 2),
            'mask': torch.zeros(2, 3, 17),
            'month': torch.zeros(2, dtype=torch.long),
        }

        y_hat = model._forward_model(batch)

        assert y_hat.shape == torch.Size([2, 17])
