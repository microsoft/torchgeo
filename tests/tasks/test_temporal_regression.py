# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
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
