# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main


class TestClassificationTask:
    @pytest.mark.parametrize('name', ['iobench'])
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
