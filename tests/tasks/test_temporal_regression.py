# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable

import pytest
from lightning.pytorch import Trainer

from torchgeo.datamodules import AirQualityDataModule, MisconfigurationException
from torchgeo.datasets import AirQuality
from torchgeo.main import main
from torchgeo.tasks import TemporalRegression


class TestTemporalRegression:
    @pytest.mark.parametrize('name', ['air_quality_mse', 'air_quality_mae'])
    def test_trainer(
        self, name: str, fast_dev_run: bool, test_config: Callable[[str], str]
    ) -> None:
        config = test_config(name + '.yaml')

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

    def test_predict(self, test_data: Callable[[str], str]) -> None:
        root = test_data('air_quality')
        model = TemporalRegression(in_channels=17, num_outputs=17, len_max_seq=3)
        datamodule = AirQualityDataModule(root=root)
        datamodule.predict_dataset = AirQuality(root)
        trainer = Trainer(accelerator='cpu')
        trainer.predict(model=model, datamodule=datamodule)
