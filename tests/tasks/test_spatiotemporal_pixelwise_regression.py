# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable

import pytest
from lightning.pytorch import Trainer

from torchgeo.datamodules import (
    CopernicusBenchBiomassS3DataModule,
    MisconfigurationException,
)
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalPixelwiseRegression


class TestSpatioTemporalPixelwiseRegression:
    @pytest.mark.parametrize('name', ['copernicus_biomass_s3_ts'])
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
        root = test_data('copernicus/l3_biomass_s3')
        model = SpatioTemporalPixelwiseRegression(
            in_channels=3, hidden_dim=8, num_layers=1
        )
        datamodule = CopernicusBenchBiomassS3DataModule(
            root=root,
            batch_size=1,
            mode='time-series',
            bands=('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance'),
        )
        datamodule.setup('test')
        datamodule.predict_dataset = datamodule.test_dataset
        trainer = Trainer(accelerator='cpu')
        predictions = trainer.predict(model=model, datamodule=datamodule)

        assert predictions is not None
        assert predictions[0].shape == (1, 1, 282, 282)
