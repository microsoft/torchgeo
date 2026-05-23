# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
import torch.nn as nn
from lightning.pytorch import Trainer

from torchgeo.datamodules import AirQualityDataModule
from torchgeo.trainers import TemporalRegressionTask

NUM_INPUT_STEPS = 3
NUM_TARGET_STEPS = 1
NUM_FEATURES = 12  


def make_datamodule(**kwargs: object) -> AirQualityDataModule:
    """Return a datamodule configured to match the test fixture."""
    defaults = dict(
        root='tests/data/air_quality',
        batch_size=4,
        num_workers=0,
        num_input_steps=NUM_INPUT_STEPS,
        num_target_steps=NUM_TARGET_STEPS,
    )
    defaults.update(kwargs)
    return AirQualityDataModule(**defaults)


def make_task(**kwargs: object) -> TemporalRegressionTask:
    """Return a task whose hyper-parameters are consistent with the fixture."""
    defaults = dict(
        in_channels=NUM_FEATURES,
        num_outputs=NUM_TARGET_STEPS * NUM_FEATURES,
        n_head=1,
        d_k=4,
        d_model=16,
        n_neurons=(16, 8),
        len_max_seq=NUM_INPUT_STEPS,
    )
    defaults.update(kwargs)
    return TemporalRegressionTask(**defaults)


class PredictAirQualityDataModule(AirQualityDataModule):
    """Subclass that also exposes a predict_dataset (mirrors the test fixture pattern)."""

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.predict_dataset = self.test_dataset


class TestTemporalRegressionTask:
    @pytest.mark.parametrize('loss', ['mse', 'mae'])
    def test_trainer(self, loss: str, fast_dev_run: bool) -> None:
        datamodule = make_datamodule()
        model = make_task(loss=loss)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictAirQualityDataModule(
            root='tests/data/air_quality',
            batch_size=4,
            num_workers=0,
            num_input_steps=NUM_INPUT_STEPS,
            num_target_steps=NUM_TARGET_STEPS,
        )
        model = make_task()
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_invalid_model(self) -> None:
        match = "Model 'invalid_model' is not supported."
        with pytest.raises(ValueError, match=match):
            TemporalRegressionTask(model='invalid_model')

    def test_invalid_loss(self) -> None:
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            TemporalRegressionTask(loss='invalid_loss')

    def test_unnormalise_no_stats(self) -> None:
        """_unnormalise is a no-op when mean/std are absent from the batch."""
        model = make_task()
        y_hat = torch.randn(4, NUM_TARGET_STEPS * NUM_FEATURES)
        y = torch.randn(4, NUM_TARGET_STEPS * NUM_FEATURES)
        out_hat, out_y = model._unnormalise(y_hat, y, batch={}, H=NUM_TARGET_STEPS)
        assert out_hat is y_hat
        assert out_y is y

    def test_unnormalise_with_stats(self) -> None:
        """_unnormalise correctly rescales when mean/std are present."""
        model = make_task()
        B = 4
        y_hat = torch.randn(B, NUM_TARGET_STEPS * NUM_FEATURES)
        y = torch.randn(B, NUM_TARGET_STEPS * NUM_FEATURES)
        batch = {'mean': torch.zeros(NUM_FEATURES), 'std': torch.ones(NUM_FEATURES)}
        out_hat, out_y = model._unnormalise(y_hat, y, batch, H=NUM_TARGET_STEPS)
        assert out_hat.shape == y_hat.shape
        assert out_y.shape == y.shape


class TestAirQualityDataModule:
    def test_setup_splits(self) -> None:
        dm = make_datamodule()
        dm.setup('fit')
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        assert len(dm.test_dataset) > 0

    def test_normalization_stats(self) -> None:
        """Mean and std are computed from training data only."""
        dm = make_datamodule()
        dm.setup('fit')
        assert dm.mean.shape == dm.std.shape
        assert (dm.std >= 0).all()

    def test_on_after_batch_transfer(self) -> None:
        """Batch is normalized and stats are injected."""
        dm = make_datamodule()
        dm.setup('fit')
        raw = next(iter(dm.train_dataloader()))
        batch = dm.on_after_batch_transfer(raw, dataloader_idx=0)
        assert 'mean' in batch
        assert 'std' in batch
