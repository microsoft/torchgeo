# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Literal

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.trainers import SpatioTemporalClassificationTask


class TestSpatioTemporalClassificationTask:
    @pytest.mark.parametrize('name', ['quakeset'])
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

    @pytest.mark.parametrize('task,num_classes', [('binary', 1), ('multiclass', 5)])
    def test_predict_step(
        self, task: Literal['binary', 'multiclass'], num_classes: int
    ) -> None:
        st_task = SpatioTemporalClassificationTask(
            in_channels=4,
            task=task,
            num_classes=num_classes,
            convlstm_hidden_dim=[16, 8],
            convlstm_kernel_size=[3, (1, 1)],
            convlstm_num_layers=2,
        )
        # (B=2, T=3, C=4, H=16, W=16)
        batch = {'image': torch.randn(2, 3, 4, 16, 16)}
        prediction = st_task.predict_step(batch, 0)
        assert prediction.shape == (2, num_classes)

    def test_forward_shape(self) -> None:
        task = SpatioTemporalClassificationTask(
            in_channels=10,
            task='multiclass',
            num_classes=20,
            convlstm_hidden_dim=[16, 8],
            convlstm_kernel_size=[3, (1, 1)],
            convlstm_num_layers=2,
        )
        x = torch.randn(2, 9, 10, 32, 32)
        y_hat = task(x)
        assert y_hat.shape == (2, 20)

    def test_unsupported_model(self) -> None:
        with pytest.raises(
            ValueError, match="Model type 'unsupported_model' is not supported"
        ):
            SpatioTemporalClassificationTask(
                in_channels=4,
                task='binary',
                num_classes=1,
                model='unsupported_model',  # type: ignore
            )
